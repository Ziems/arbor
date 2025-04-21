# Borrowed from Will Brown's Verifiers Library

import warnings
from typing import Callable, Optional, Union, Any, List, Sized, Sampler
import json
import os
import time
import threading
import queue
from functools import lru_cache
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import load_dataset, Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch import nn
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
# from verifiers import RewardFunc
# from verifiers.envs.environment import Environment
# from verifiers.utils.logging_utils import print_prompt_completions_sample
# from verifiers.imports import LLM, SamplingParams
# from verifiers.inference.vllm_client import VLLMClient

# monkey patch vllm client
# import trl.extras.vllm_client
# trl.extras.vllm_client.VLLMClient = VLLMClient

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

# if is_wandb_available():
#     import wandb

tldr_dataset = load_dataset("trl-lib/tldr", split="train")

# Dummy writer process to simulate real-time data
def dummy_writer(output_file="data_stream.jsonl"):

    def _reward_func(prompts, completions):
        return [-abs(20 - len(completion)) if completion is not None else -300 for completion in completions]

    print(f"Starting dummy writer, writing to {output_file}")
    for i, item in enumerate(tldr_dataset):

        input_messages = [{"role": "user", "content": item["prompt"]}]
        # response = client.chat.completions.create(
        completions = [{
            "role": "assistant",
            "content": "This is a test completion"
        } for _ in range(2)] # 2 generations

        rewards = _reward_func(item["prompt"], [c["content"] for c in completions])
        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append({
                "messages": input_messages,
                "completion": completion,
                "reward": reward
            })
        with open(output_file, 'a') as f:
            f.write(json.dumps({'batch':batch}) + '\n')
        print(f"Wrote item {i} to {output_file}")
        time.sleep(5)  # Write a new item every 5 seconds


# torch.nanstd doesn't exist, so we define it here
def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard deviation of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`):
            Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`:
            Standard deviation of the tensor, ignoring NaNs.
    """
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)  # Compute variance ignoring NaNs
    count = torch.sum(~torch.isnan(tensor))  # Count of non-NaN values
    variance *= count / (count - 1)  # Bessel's correction
    return torch.sqrt(variance)

class GroupedDataSampler(Sampler):
    """
    Sampler that transforms grouped data into the format expected by GRPOTrainer.
    Each sample in the dataset is expected to be a dict containing:
    - 'messages': List of message dicts (the prompt)
    - 'completions': List of completion dicts
    - 'rewards': List of rewards for each completion

    Args:
        data_source (`Sized`):
            Dataset to sample from.
        num_generations (`int`):
            Number of generations per prompt (should match the number of completions in each sample).
        seed (`int` or `None`, *optional*, defaults to `None`):
            Random seed for reproducibility.
    """

    def __init__(
        self,
        data_source: Sized,
        num_generations: int,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.num_generations = num_generations
        self.num_samples = len(data_source)
        self.seed = seed
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):
        # Get random permutation of indices
        indexes = torch.randperm(self.num_samples, generator=self.generator).tolist()

        for idx in indexes:
            # Get the grouped sample
            sample = self.data_source[idx]

            # Transform the grouped data into individual examples
            for i in range(self.num_generations):
                yield {
                    'messages': sample['messages'],
                    'completion': sample['completions'][i],
                    'reward': sample['rewards'][i]
                }

    def __len__(self) -> int:
        return self.num_samples * self.num_generations

class ArborGRPOTrainer(GRPOTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            scale_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        # self.vllm_client = None
        # if not args.use_vllm: # type: ignore
        #     raise ValueError("vLLM must be enabled for GRPOEnvTrainer")

        super().__init__(
            model=model,
            reward_funcs=[],
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        self.scale_rewards = scale_rewards
        # self.sampling_params = SamplingParams(
        #     max_tokens=self.max_completion_length,
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     top_k=-1 if self.top_k is None else self.top_k,
        #     min_p=0.0 if self.min_p is None else self.min_p,
        #     repetition_penalty=self.repetition_penalty
        # )

    def _get_train_sampler(self) -> Sampler:
        # Use our custom sampler that transforms the grouped data format
        return GroupedDataSampler(
            data_source=self.train_dataset,
            num_generations=self.num_generations,
            seed=self.args.seed,
        )

    def _generate_and_score_completions(
         self, batch: List[dict[str, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device

        # Process prompts and completions
        prompt_completion_texts = []
        for example in batch:
            prompt_completion_texts.append(
                maybe_apply_chat_template(
                    {
                        'prompt': example['messages'],
                        'completion': [example['completion']]
                    },
                    self.tokenizer
                )
            )

        # Tokenize prompts
        prompt_texts = [prompt_completion_text['prompt'] for prompt_completion_text in prompt_completion_texts]
        prompt_inputs = self.processing_class(prompt_texts, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False).to(device)
        prompt_ids = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        # Tokenize completions
        completion_texts = [prompt_completion_text['completion'] for prompt_completion_text in prompt_completion_texts]
        completion_ids = self.processing_class(completion_texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        completion_ids, completion_mask = completion_ids["input_ids"], completion_ids["attention_mask"]


        # prompts = [x["prompt"] for x in inputs] # type: ignore
        # prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        # prompt_inputs = self.processing_class(
        #     prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        # ) # type: ignore
        # prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        # prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # if self.state.global_step != self._last_loaded_step:
        #     self._move_model_to_vllm()
        #     self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
        # all_prompts = gather_object(prompts)
        # if self.accelerator.is_main_process:
        #     env_result = self.env.generate(
        #         prompts=all_prompts,
        #         llm=self.vllm_client, # type: ignore
        #         sampling_params=self.sampling_params,
        #     )
        #     completion_ids = env_result['ids']
        #     completion_messages = env_result['messages']
        #     completion_mask = env_result['mask']

        # else:
        #     completion_ids = [None] * len(all_prompts)
        #     completion_messages = [None] * len(all_prompts)
        #     completion_mask = [None] * len(all_prompts)

        # completion_ids = broadcast_object_list(completion_ids, from_process=0)
        # completion_messages = broadcast_object_list(completion_messages, from_process=0)
        # completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(batch),
            (self.accelerator.process_index + 1) * len(batch),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)

        logits_to_keep = completion_ids.size(1)

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )

        # use message dicts for reward function inputs
        # completions = completion_messages
        # rewards_per_func = torch.zeros(len(batch), len(self.reward_funcs), device=device)
        # for i, reward_func in enumerate(self.reward_funcs):
        #     # Repeat all input columns (but "prompt" and "completion") to match the number of generations
        #     keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
        #     reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
        #     output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore

        #     output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]
        #     rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        # # If all reward functions return None for a given row, issue a detailed warning
        # if torch.isnan(rewards_per_func).all(dim=1).any():
        #     nan_row_idx = torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
        #     row_reward_kwargs = {key: value[nan_row_idx] for key, value in reward_kwargs.items()} # type: ignore
        #     row_reward_kwargs["prompt"] = prompts[nan_row_idx]
        #     row_reward_kwargs["completion"] = completions[nan_row_idx] # type: ignore
        #     warnings.warn(
        #         f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
        #         "Please ensure that at least one reward function returns a valid reward."
        #     )


        # rewards_per_func = gather(rewards_per_func)

        # # Apply weights to each reward function's output and sum
        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)

        # # Compute grouped-wise rewards
        # mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # # Normalize the rewards to compute the advantages
        # mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        # advantages = (rewards - mean_grouped_rewards)

        # std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        # std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        rewards = torch.tensor([example['reward'] for example in batch], dtype=torch.float32).to(device)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped_rewards

        if self.scale_rewards:
            # Scale the rewards to be between 0 and 1
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(batch),
            (self.accelerator.process_index + 1) * len(batch),
        )
        advantages = advantages[process_slice]

        # # Log the metrics
        # mode = "eval" if self.control.should_evaluate else "train"

        # completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        # self._metrics[mode]["completion_length"].append(completion_length)

        # # Calculate mean reward per function, but only for samples where the function was applied
        # for i, reward_func in enumerate(self.reward_funcs):
        #     reward_func_name = reward_func.__name__ # type: ignore
        #     # Only calculate mean for samples where this reward function was applied (non-NaN values)
        #     mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
        #     self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        #     std_rewards = nanstd(rewards_per_func[:, i]).item()
        #     self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        # self._metrics[mode]["reward"].append(rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item()) # type: ignore

        # if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
        #     prompts_to_log = gather_object(prompts)
        #     completions_to_log = gather_object(completions)
        #     rewards_to_log = rewards.tolist()

        #     if self.accelerator.is_main_process:
        #         if is_rich_available():
        #             print_prompt_completions_sample(
        #                 [str(prompts_to_log[0][-1]["content"])],
        #                 [completions_to_log[0]],
        #                 [rewards_to_log[0]],
        #                 self.state.global_step,
        #             )
        #         if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
        #             import pandas as pd

        #             # For logging
        #             table = {
        #                 "step": [str(self.state.global_step)] * len(rewards),
        #                 "prompt": prompts_to_log,
        #                 "completion": completions_to_log,
        #                 "reward": rewards.tolist(),
        #             }
        #             df = pd.DataFrame(table)
        #             wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

class JSONLStreamReader:
    def __init__(self, jsonl_file):
        self.jsonl_file = jsonl_file
        self.last_position = 0

    def read_new_entries(self):
        """Read any new entries from the file"""
        if not os.path.exists(self.jsonl_file):
            return []

        new_entries = []
        with open(self.jsonl_file, 'r') as f:
            f.seek(self.last_position)
            new_lines = f.readlines()

            if new_lines:
                self.last_position = f.tell()
                for line in new_lines:
                    try:
                        new_entries.append(json.loads(line.strip()))
                    except json.JSONDecodeError:
                        print(f"Warning: Skipping invalid JSON line: {line.strip()}")

        return new_entries

class DataProducer:
    def __init__(self, jsonl_file="data_stream.jsonl"):
        self.reader = JSONLStreamReader(jsonl_file)
        self.queue = queue.Queue()
        self.running = False
        self.thread = None

    def start(self):
        """Start the producer thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._produce, daemon=True)
            self.thread.start()

    def stop(self):
        """Stop the producer thread"""
        self.running = False
        if self.thread:
            self.thread.join()

    def _produce(self):
        """Main production loop"""
        while self.running:
            new_entries = self.reader.read_new_entries()
            for entry in new_entries:
                self.queue.put(entry)
            time.sleep(1)  # Configurable delay between checks

class BlockingDynamicDataset(Dataset):
    def __init__(self, accelerator, size=1000, maxsize=100, jsonl_file="data_stream.jsonl"):
        self.size = size
        self.accelerator = accelerator
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)

        # Create and start the producer in the main process only
        if accelerator is not None and accelerator.is_main_process:
            self.producer = DataProducer(jsonl_file)
            self.producer.start()

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        print(f"Item not cached, getting new data")
        if self.accelerator.is_main_process:
            print(f"Main process {self.accelerator.process_index} getting new data")
            new_data = self.producer.queue.get(block=True)
            return broadcast_object_list([new_data])[0]
        else:
            print(f"Other process {self.accelerator.process_index} waiting for data")
            return broadcast_object_list([None])[0]

    def __getitem__(self, idx):
        print(f"Process {self.accelerator.process_index} getting item {idx}")
        return self.get_cached_data(idx)

    def __del__(self):
        """Cleanup when the dataset is destroyed"""
        if hasattr(self, 'producer'):
            self.producer.stop()

training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
trainer = ArborGRPOTrainer(
    model="Qwen/Qwen2-0.5B-Instruct",
    args=training_args,
    train_dataset=BlockingDynamicDataset(None),
)

# Start the dummy writer only in the main process
if trainer.accelerator.is_main_process:
    dummy_writer_thread = threading.Thread(
        target=dummy_writer,
        args=("data_stream.jsonl",),
        daemon=True
    )
    dummy_writer_thread.start()
    # Give the writer a moment to start creating data
    time.sleep(2)

trainer.train_dataset = BlockingDynamicDataset(
    trainer.accelerator,
    jsonl_file="data_stream.jsonl"
)
trainer.train()