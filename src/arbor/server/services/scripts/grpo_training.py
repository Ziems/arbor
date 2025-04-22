# Borrowed from Will Brown's Verifiers Library

import random
from typing import Callable, Optional, Union, Any, List, Sized
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
import argparse
import multiprocessing

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

            status_queue: Optional[multiprocessing.Queue] = None,

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
        self.status_queue = status_queue
        self._last_loaded_step = 0
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
        if self.status_queue is not None and hasattr(self, '_last_loaded_step'):
            if self.state.global_step - self._last_loaded_step > 25 - 1:
                self.save_model()
                self.status_queue.put({"status": "update_inference_model"})
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

class BlockingQueueDataset(Dataset):
    def __init__(self, accelerator, data_queue, size=1000, maxsize=100):
        self.size = size
        self.accelerator = accelerator
        self.data_queue = data_queue
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)
        self.completion_counters = {}

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        if self.accelerator.is_main_process:
            print(f"Main process {self.accelerator.process_index} getting new data")
            new_data = self.data_queue.get(block=True)
            # Initialize completion counters
            if idx not in self.completion_counters:
                self.completion_counters[idx] = 0
            return broadcast_object_list([new_data])[0]
        else:
            print(f"Other process {self.accelerator.process_index} waiting for data")
            return broadcast_object_list([None])[0]

    def __getitem__(self, idx):
        print(f"Process {self.accelerator.process_index} getting item {idx}")
        data = self.get_cached_data(idx)

        if data is None:
            return None

        counter = self.completion_counters.get(idx, 0)
        item = data[counter]
        self.completion_counters[idx] = (counter + 1) % len(data)

        print(f"Process {self.accelerator.process_index} got item {item['completion']['content'][:50]}")
        return item

class CommandListener:
    def __init__(self, command_queue, status_queue, trainer):
        self.command_queue = command_queue
        self.status_queue = status_queue
        self.trainer = trainer
        self.running = True
        self.thread = None

    def start(self):
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def _listen(self):
        while self.running:
            try:
                command = self.command_queue.get_nowait()
                if command["command"] == "terminate":
                    self.status_queue.put({"status": "terminated"})
                    self.running = False
                    # Signal to the trainer to stop
                    # self.trainer.control.should_training_stop = True
                    break
                # Handle other commands here...
            except multiprocessing.queues.Empty:
                time.sleep(0.1)  # Prevent busy waiting
            except Exception as e:
                self.status_queue.put({
                    "status": "error",
                    "error": str(e)
                })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--testing_mode", action="store_true")

    queue_args = parser.add_argument_group("Queue arguments")
    queue_args.add_argument("--command_queue", type=int)
    queue_args.add_argument("--status_queue", type=int)
    queue_args.add_argument("--data_queue", type=int)
    args = parser.parse_args()

    command_queue = None
    status_queue = None
    data_queue = None

    if not args.testing_mode:
        if not all([args.command_queue, args.status_queue, args.data_queue]):
            raise ValueError("All queues must be provided if not in testing mode")
        command_queue = multiprocessing.Queue(handle=args.command_queue)
        status_queue = multiprocessing.Queue(handle=args.status_queue)
        data_queue = multiprocessing.Queue(handle=args.data_queue)
    else:
        command_queue = multiprocessing.Queue()
        status_queue = multiprocessing.Queue()
        data_queue = multiprocessing.Queue()

    try:
        training_args = GRPOConfig(output_dir="Qwen2-0.5B-GRPO", logging_steps=10)
        trainer = ArborGRPOTrainer(
            model="Qwen/Qwen2-0.5B-Instruct",
            args=training_args,
            train_dataset=BlockingQueueDataset(None, data_queue),
            status_queue=status_queue,
        )

        # Initialize the dataset with the actual accelerator
        trainer.train_dataset = BlockingQueueDataset(
            trainer.accelerator,
            data_queue
        )
        # Set up and start the command listener
        command_listener = CommandListener(command_queue, status_queue, trainer)
        command_listener.start()

        if args.testing_mode:
            # Start a dummy writer
            dummy_writer_thread = threading.Thread(
                target=dummy_writer,
                args=(data_queue,),
                daemon=True
            )
            dummy_writer_thread.start()

            status_reader_thread = threading.Thread(
                target=status_reader,
                args=(status_queue,),
                daemon=True
            )
            status_reader_thread.start()
        # Start training
        try:
            print("Training...")
            trainer.train()

        except Exception as e:
            print(f"Error: {e}")
            status_queue.put({
                "status": "error",
                "error": str(e)
            })

    finally:
        command_queue.close()
        status_queue.close()
        data_queue.close()


# Dummy writer process to simulate real-time data
def dummy_writer(data_queue):
    tldr_dataset = load_dataset("trl-lib/tldr", split="train")

    def _reward_func(prompts, completions):
        return [-abs(20 - len(completion)) if completion is not None else -300 for completion in completions]

    print(f"Starting dummy writer")
    for i, item in enumerate(tldr_dataset):

        input_messages = [{"role": "user", "content": item["prompt"]}]
        # response = client.chat.completions.create(
        completions = [{
            "role": "assistant",
            "content": "This is a test completion" + hex(random.randint(0, 0xFFFFFF))[2:]
        } for _ in range(8)] # 8 generations

        rewards = _reward_func(item["prompt"], [c["content"] for c in completions])
        batch = []
        for completion, reward in zip(completions, rewards):
            batch.append({
                "messages": input_messages,
                "completion": completion,
                "reward": reward
            })
        data_queue.put(batch)
        # print(f"Wrote item {i} to {output_file}")
        time.sleep(5)  # Write a new item every 5 seconds

def status_reader(status_queue):
    while True:
        status = status_queue.get(block=True)
        print(f"status: {status}")
        time.sleep(1)

if __name__ == "__main__":
    main()