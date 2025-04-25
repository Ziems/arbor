# Borrowed from Will Brown's Verifiers Library
# TODO: Need proper attribution before release

import random
from typing import Callable, Optional, Union, Any, List, Sized
import json
import os
import time
import threading
from functools import lru_cache
from accelerate.utils import broadcast_object_list, gather, gather_object
from accelerate import Accelerator
from datasets import load_dataset, Dataset, IterableDataset
from peft import PeftConfig # type: ignore
import torch
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import argparse
from arbor.server.services.comms.comms import ArborServerCommsHandler, ArborScriptCommsHandler

from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad
import zmq

# if is_wandb_available():
#     import wandb
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

            comms_handler: Optional[ArborScriptCommsHandler] = None,
            update_interval: Optional[int] = 25,

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
        self.comms_handler = comms_handler
        self._last_loaded_step = 0
        self.update_interval = update_interval
        # self.sampling_params = SamplingParams(
        #     max_tokens=self.max_completion_length,
        #     temperature=self.temperature,
        #     top_p=self.top_p,
        #     top_k=-1 if self.top_k is None else self.top_k,
        #     min_p=0.0 if self.min_p is None else self.min_p,
        #     repetition_penalty=self.repetition_penalty
        # )


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
                    self.processing_class
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

        # if self.max_prompt_length is not None:
        #     prompt_ids = prompt_ids[:, -self.max_prompt_length :]
        #     prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Check if we need to update the inference model
        if self.comms_handler is not None and hasattr(self, '_last_loaded_step'):
            if self.state.global_step - self._last_loaded_step > self.update_interval - 1:
                if self.accelerator.is_main_process:
                    # SO I think this works if I make sure the saved model is
                    self.comms_handler.send_status({"status": f"saving model to {self.args.output_dir}"})
                    self.save_model()
                    self.comms_handler.send_status({"status": "update_inference_model"})
                    self._last_loaded_step = self.state.global_step

        # if self.state.global_step != self._last_loaded_step:
        #     self._move_model_to_vllm()
        #     self._last_loaded_step = self.state.global_step


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

        rewards = torch.tensor([example['reward'] for example in batch], dtype=torch.float32).to(device)
        rewards = gather(rewards)
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

# TODO: This should be in a separate file probably. It will be used by other training scripts as well.
class ArborTerminateTrainingCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that handles termination requests.
    """

    def __init__(self, comms_handler: ArborScriptCommsHandler, accelerator: Accelerator):
        self.comms_handler = comms_handler
        self.terminate_requested = False
        self.accelerator = accelerator

        if self.comms_handler is not None:
            self._command_thread = threading.Thread(target=self._monitor_commands, daemon=True)
            self._command_thread.start()


    def _monitor_commands(self):
        """Background thread that monitors for commands from the server."""
        if not self.comms_handler:
            return
            
        try:
            for command in self.comms_handler.receive_command():
                if command.get("command") == "terminate" and self.accelerator.is_main_process:
                    self.terminate_requested = True
                    self.comms_handler.send_status({"status": "Terminate requested. Training will stop at next step."})

        except Exception as e:
            print(f"Error in command monitor: {e}")
            self.comms_handler.send_status({"status": "error", "error": str(e)})

    def on_step_end(self, args, state, control, **kwargs):
        if self.terminate_requested:
            control.should_training_stop = True

class BlockingQueueDataset(Dataset):
    def __init__(self, accelerator: Accelerator, comms_handler: ArborScriptCommsHandler, size=1000, maxsize=100):
        self.size = size
        self.accelerator = accelerator
        self.comms_handler = comms_handler
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)
        self.completion_counters = {}

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        if self.accelerator.is_main_process:
            print(f"Main process {self.accelerator.process_index} getting new data")
            new_data = self.comms_handler.receive_data()  # This blocks until data is available
            # print(f"Main process {self.accelerator.process_index} got new data")
            if idx not in self.completion_counters:
                self.completion_counters[idx] = 0
            return broadcast_object_list([new_data])[0]
        else:
            # print(f"Other process {self.accelerator.process_index} waiting for data")
            return broadcast_object_list([None])[0]

    def __getitem__(self, idx):
        # print(f"Process {self.accelerator.process_index} getting item {idx}")
        data = self.get_cached_data(idx)

        if data is None:
            return None

        counter = self.completion_counters.get(idx, 0)
        item = data[counter]
        self.completion_counters[idx] = (counter + 1) % len(data)

        # print(f"Process {self.accelerator.process_index} got item {item['completion']['content'][:50]}")
        return item

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    pipe_args = parser.add_argument_group("Pipe arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int, required=True)
    pipe_args.add_argument("--status_port", type=int, required=True)
    pipe_args.add_argument("--data_port", type=int, required=True)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument("--trl_train_kwargs", type=json.loads, help="Training arguments as a JSON string")
    training_args.add_argument("--arbor_train_kwargs", type=json.loads, help="Training arguments as a JSON string")

    args = parser.parse_args()


    if args.debug:
        server_comms_handler = ArborServerCommsHandler(
            host=args.host,
        )

        args.command_port = server_comms_handler.command_port
        args.status_port = server_comms_handler.status_port
        args.data_port = server_comms_handler.data_port

        def debug_data_generator():
            tldr_dataset = load_dataset("trl-lib/tldr", split="train")
            while True:
                for item in tldr_dataset:
                    input_messages = [{"role": "user", "content": item["prompt"]}]
                    completions = [{
                        "role": "assistant",
                        "content": "This is a test completion" + hex(random.randint(0, 0xFFFFFF))[2:]
                    } for _ in range(8)]

                    rewards = [-abs(20 - len(c["content"])) for c in completions]
                    batch = []
                    for completion, reward in zip(completions, rewards):
                        batch.append({
                            "messages": input_messages,
                            "completion": completion,
                            "reward": reward
                        })
                    server_comms_handler.send_data(batch)
                    time.sleep(5)

        debug_thread = threading.Thread(target=debug_data_generator, daemon=True)
        debug_thread.start()

        def status_listener():
            # Need to set subscription for PUB/SUB pattern
            server_comms_handler.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            for status in server_comms_handler.receive_status():
                print(f"Status: {status}")

        status_listener_thread = threading.Thread(target=status_listener, daemon=True)
        status_listener_thread.start()

    # Create client handler
    comms_handler = ArborScriptCommsHandler(
        host=args.host,
        command_port=args.command_port,
        status_port=args.status_port,
        data_port=args.data_port
    )

    try:
        trl_train_args = {**(args.trl_train_kwargs or {})}
        arbor_train_args = {**(args.arbor_train_kwargs or {})}

        # TODO: These assertions should be done in some better way
        assert "output_dir" in trl_train_args, "output_dir is required"


        training_args = GRPOConfig(**trl_train_args)
        trainer = ArborGRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            args=training_args,
            train_dataset=BlockingQueueDataset(None, comms_handler),
            comms_handler=comms_handler,
            **arbor_train_args
        )

        # Initialize the dataset with the actual accelerator
        trainer.train_dataset = BlockingQueueDataset(
            trainer.accelerator,
            comms_handler
        )
        # Add a callback to handle termination requests
        trainer.add_callback(
            ArborTerminateTrainingCallback(
                comms_handler,
                trainer.accelerator
            )
        )

        print("Training...")
        trainer.train()

    except KeyboardInterrupt:
        print("\nReceived interrupt, shutting down...")
    except Exception as e:
        print(f"Error: {e}")
        comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        comms_handler.close()

if __name__ == "__main__":
    main()