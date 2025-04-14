import random
import string
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, setup_chat_format
from trl.data_utils import maybe_apply_chat_template
from trl.models.modeling_base import create_reference_model

from arbor.server.api.models.schemas import GRPORequest, GRPOConfigRequest
from arbor.server.core.config import Settings
from arbor.server.services.job_manager import Job, JobEvent, JobStatus

class GRPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings


    def make_output_dir(self, request: GRPORequest):
        model_name = request.model.split('/')[-1].lower()
        suffix = request.suffix if request.suffix is not None else ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"grpo:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.settings.STORAGE_PATH).resolve() / "models" / name)

    def find_training_args(self, request: GRPOConfigRequest):

        name, output_dir = self.make_output_dir(request)

        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
            # Using the default TRL values here
            "beta": 0.04,
            "num_iterations": 1,
            "temperature": 0.9, # TODO: This should match vLLM
            "num_generations": 3,
            "scale_rewards": True,
            "epsilon_low": 0.2,
            "epsilon_high": 0.2,
        }

        train_kwargs = {'packing': False}
        train_kwargs={**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def initialize_config(self, request: GRPOConfigRequest):
        self.train_kwargs = self.find_training_args(request)
        if self.train_kwargs.get("device", None) is None:
            self.train_kwargs["device"] = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=request.model
        ).to(self.train_kwargs["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=request.model)

        # Set up the chat format; generally only for non-chat model variants, hence the try-except.
        try:
            self.model, self.tokenizer = setup_chat_format(model=self.model, tokenizer=self.tokenizer)
        except Exception:
            pass


        # Reference model
        self.beta = self.train_kwargs['beta']
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        else:
            # If PEFT configuration is not provided, create a reference model based on the initial model.
            self.ref_model = create_reference_model(self.model)

        # TODO: Temporary! This can be improved with a scheduler
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

        if "max_seq_length" not in self.train_kwargs or self.train_kwargs["max_seq_length"] is None:
            self.train_kwargs["max_seq_length"] = 4096


    def grpo_step(self, request: GRPORequest, job: Job):
        job.status = JobStatus.RUNNING
        job.add_event(JobEvent(level="info", message="Running GRPO step", data={}))

        try:
            job.add_event(JobEvent(level="info", message="Tokenizing batch", data={}))
            batch = dataset_from_json(request.batch)

            self.optimizer.zero_grad()
            total_loss = 0
            for example in batch:
                inputs = self.score_completions(example)
                loss = self.compute_loss(inputs)
                total_loss += loss

            total_loss = total_loss / len(batch)
            total_loss.backward()
            self.optimizer.step()

        except Exception as e:
            job.add_event(JobEvent(level="error", message=f"Training failed: {str(e)}", data={}))
            job.status = JobStatus.FAILED
            raise
        finally:
            pass

    def compute_loss(self, inputs):
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(self.model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        import pdb; pdb.set_trace()
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.train_kwargs["num_iterations"] > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.train_kwargs["epsilon_low"], 1 + self.train_kwargs["epsilon_high"])
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.train_kwargs['beta'] != 0.0:
            per_token_loss = per_token_loss + self.train_kwargs['beta'] * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()
        return loss


    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.train_kwargs["temperature"]
        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

    def score_completions(self, example):
        device = 'cuda'
        prompt_completion_texts = []
        for completion in example['completions']:
            prompt_completion_texts.append(
                maybe_apply_chat_template(
                    {
                        'prompt': example['input']['messages'],
                        'completion': [completion]
                    },
                    self.tokenizer
                )
            )
        prompt_texts = [prompt_completion_text['prompt'] for prompt_completion_text in prompt_completion_texts]
        prompt_inputs = self.tokenizer(prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(device)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        completion_texts = [prompt_completion_text['completion'] for prompt_completion_text in prompt_completion_texts]
        completion_ids = self.tokenizer(completion_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        completion_ids, completion_mask = completion_ids["input_ids"], completion_ids["attention_mask"]

        ## Masking is done differently in the GRPO trainer. Not sure if this is necessary.
        ## Only difference seems to be that this approach has a \n character after the eos token,
        ## while the GRPO trainer does not.
        # is_eos = completion_ids == self.processing_class.eos_token_id
        # eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        # eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        # sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        # completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens


        # TODO: Might need to throw an error here because the prompt has already been run through model
        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]


        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.train_kwargs["num_iterations"] > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.train_kwargs["beta"] == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                # with self.accelerator.unwrap_model(self.model).disable_adapter():
                ref_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )

        rewards = torch.tensor([completion['reward'] for completion in example['completions']], dtype=torch.float32).to(device)
        mean_grouped_rewards = rewards.view(-1, self.train_kwargs["num_generations"]).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.train_kwargs["num_generations"]).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.train_kwargs["num_generations"], dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.train_kwargs["num_generations"], dim=0)
        advantages = rewards - mean_grouped_rewards
        if self.train_kwargs["scale_rewards"]:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        # process_slice = slice(
        #     self.accelerator.process_index * len(prompts),
        #     (self.accelerator.process_index + 1) * len(prompts),
        # )
        # advantages = advantages[process_slice]
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

def dataset_from_json(json_data):
    from datasets import Dataset

    if json_data is not None:
        dataset = Dataset.from_list(json_data)

    return dataset

def selective_log_softmax(logits, index):
    """
    A memory-efficient implementation of the common `log_softmax -> gather` operation.

    This function is equivalent to the following naive implementation:
    ```python
    logps = torch.gather(logits.log_softmax(-1), dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
    ```

    Args:
        logits (`torch.Tensor`):
            Logits tensor of shape `(..., num_classes)`.
        index (`torch.Tensor`):
            Index tensor of shape `(...)`, specifying the positions to gather from the log-softmax output.

    Returns:
        `torch.Tensor`:
            Gathered log probabilities with the same shape as `index`.
    """
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)
        # loop to reduce peak mem consumption
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values  # log_softmax(x_i) = x_i - logsumexp(x)
    else:
        # logsumexp approach is unstable with bfloat16, fall back to slightly less efficent approach
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):  # loop to reduce peak mem consumption
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps