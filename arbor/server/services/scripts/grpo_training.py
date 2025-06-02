###############################################################################
# Initial Versions of this File Borrowed from Will Brown's Verifiers Library  #
# https://github.com/willccbb/verifiers                                       #
###############################################################################

import argparse
import shutil
import os
import json
import random
import threading
import time
from functools import lru_cache
from typing import Any, List, Optional, Union

import torch
import trl.extras.vllm_client
import zmq
from accelerate import Accelerator
from accelerate.utils import gather, gather_object
from datasets import Dataset, IterableDataset, load_dataset
from peft import AutoPeftModelForCausalLM, LoraConfig, PeftConfig  # type: ignore
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from trl import GRPOConfig, GRPOTrainer
from trl.data_utils import maybe_apply_chat_template

from arbor.server.services.comms.comms import (
    ArborScriptCommsHandler,
    ArborServerCommsHandler,
)
from arbor.server.services.inference.vllm_client import VLLMClient

trl.extras.vllm_client.VLLMClient = VLLMClient

if is_wandb_available():
    import wandb

last_step_time = None
last_queue_pop_time = None


def time_since_last_step():
    global last_step_time
    if last_step_time is None:
        return float("inf")
    return time.time() - last_step_time


def get_time_since_last_queue_pop():
    global last_queue_pop_time
    if last_queue_pop_time is None:
        return float("inf")
    return time.time() - last_queue_pop_time


class ArborGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        scale_rewards: bool = True,
        args: Optional[GRPOConfig] = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        comms_handler: Optional[ArborScriptCommsHandler] = None,
        lora: Optional[bool] = False,
        # We do nothing with max_context_length right now
        vllm_group_port: Optional[int] = None,
        max_context_length: Optional[int] = None,
        **kwargs,
    ):
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
        self.peft_config = peft_config
        self.scale_rewards = scale_rewards
        self.comms_handler = comms_handler

        self.vllm_client = None
        args.use_vllm = True
        self.use_vllm = True
        if self.accelerator.is_main_process:
            print(
                f"Initializing vLLM client with server port {args.vllm_server_port} and group port {vllm_group_port}"
            )
            self.vllm_client = VLLMClient(
                args.vllm_server_host,
                args.vllm_server_port,
                group_port=vllm_group_port,
                connection_timeout=args.vllm_server_timeout,
            )
            self.vllm_client.init_communicator()

        # vLLM specific sampling arguments
        self.guided_decoding_regex = args.vllm_guided_decoding_regex

        self._last_loaded_step = (
            -1
        )  # tag to avoid useless loading during grad accumulation

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

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
                        "prompt": example["messages"],
                        "completion": [example["completion"]],
                    },
                    self.processing_class,
                )
            )

        # Tokenize prompts
        prompt_texts = [
            prompt_completion_text["prompt"]
            for prompt_completion_text in prompt_completion_texts
        ]
        prompt_inputs = self.processing_class(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        ).to(device)
        prompt_ids = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        # Tokenize completions
        completion_texts = [
            prompt_completion_text["completion"]
            for prompt_completion_text in prompt_completion_texts
        ]
        completion_ids = self.processing_class(
            completion_texts,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        completion_ids, completion_mask = (
            completion_ids["input_ids"],
            completion_ids["attention_mask"],
        )

        if self.max_prompt_length is not None:
            if prompt_ids.shape[1] > self.max_prompt_length:
                print(f"Truncating prompt to {self.max_prompt_length} tokens")
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.max_completion_length is not None:
            if completion_ids.shape[1] > self.max_completion_length:
                print(f"Truncating completion to {self.max_completion_length} tokens")
            completion_ids = completion_ids[:, : self.max_completion_length]
            completion_mask = completion_mask[:, : self.max_completion_length]

        # Update the inference model if we've done a step
        # TODO: This should be done maybe at the end of the step
        if self.state.global_step != self._last_loaded_step:
            print("Updating inference model...")
            self.comms_handler.send_status({"status": "weight_update_start"})
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step
            self.comms_handler.send_status({"status": "weight_update_complete"})

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        print(
            f"prompt_completion_ids.shape (after truncation, if enabled): {prompt_completion_ids.shape}, prompt_ids.shape: {prompt_ids.shape}, completion_ids.shape: {completion_ids.shape}"
        )

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
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        rewards = torch.tensor(
            [example["reward"] for example in batch], dtype=torch.float32
        ).to(device)
        rewards = gather(rewards)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
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

        ## Logged Metrics Removed Here
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()  # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())  # type: ignore

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompt_texts)
            completions_to_log = gather_object(completion_texts)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None:  # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards_to_log,
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})  # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }


class LastStepTimeCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def on_step_end(self, args, state, control, **kwargs):
        global last_step_time
        print(f"Time since last step: {time_since_last_step()}")
        last_step_time = time.time()


class WeightUpdateCallback(TrainerCallback):
    """A callback that sends weight update completion status after each step"""
    def __init__(self):
        self.comms_handler = None

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def on_step_end(self, args, state, control, **kwargs):
        if self.comms_handler and self.comms_handler.is_main_process:
            print("[DEBUG] Sending weight update completion status")
            self.comms_handler.send_status({"status": "weight_update_complete"})


class BlockingQueueDataset(Dataset):
    def __init__(
        self,
        accelerator: Accelerator,
        comms_handler: ArborScriptCommsHandler,
        size=10_000,  # Just a random number
        maxsize=100,
    ):
        self.size = size
        self.accelerator = accelerator
        self.comms_handler = comms_handler
        self.get_cached_data = lru_cache(maxsize=maxsize)(self._get_data)
        self.completion_counters = {}

    def __len__(self):
        return self.size

    def _get_data(self, idx):
        rank = self.accelerator.process_index
        world_size = self.accelerator.num_processes

        if self.accelerator.is_main_process:
            global last_queue_pop_time
            last_queue_pop_time = time.time()

        if idx not in self.completion_counters:
            self.completion_counters[idx] = 0

        try:
            new_data = self.comms_handler.receive_data()

        except Exception as e:
            print(f"[rank {rank}] Error receiving data: {e}")
            new_data = None

        return new_data

    def __getitem__(self, idx):
        data = self.get_cached_data(idx)
        # Create hash of data to detect if processes are using the same idx for the same data
        data_hash = format(abs(hash(str(data))) % (16**8), "08x")

        if data is None:
            return None

        counter = self.completion_counters.get(idx, 0)
        item = data[counter]
        self.completion_counters[idx] = (counter + 1) % len(data)
        return item


class CommandMonitor:
    def __init__(
        self,
        comms_handler: ArborScriptCommsHandler,
        trainer: ArborGRPOTrainer,
        base_model_name: str,
    ):
        self.comms_handler = comms_handler
        self.trainer = trainer
        self.base_model_name = base_model_name
        self.command_thread = threading.Thread(
            target=self._monitor_commands, daemon=True
        )
        self.command_thread.start()

        self.broadcast_thread = threading.Thread(
            target=self._monitor_broadcasts, daemon=True
        )
        self.broadcast_thread.start()

    def _monitor_commands(self):
        """Background thread that monitors for commands from the server."""
        if not self.comms_handler:
            return
        try:
            for command in self.comms_handler.receive_command():
                print(f"Main process received command: {command}")
                if (
                    command.get("command") == "save_model"
                    and self.trainer.accelerator.is_main_process
                ):
                    print(
                        f"[Training Script] Instructed to save model at {self.trainer.args.output_dir}"
                    )
                    while (
                        time_since_last_step() <= 10
                        or get_time_since_last_queue_pop() <= 10
                    ):
                        print(f"Waiting for steps to finish")
                        print(
                            f"Time since last step: {time_since_last_step():.1f} (needs to be >= 10)"
                        )
                        print(
                            f"Time since last queue pop: {get_time_since_last_queue_pop():.1f} (needs to be >= 10)"
                        )
                        time.sleep(5)
                    print("[Training Script] Saving model...")
                    if self.trainer.peft_config:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir + "/adapter/"
                        )
                        _model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
                            self.trainer.args.output_dir + "/adapter/",
                            config=self.trainer.peft_config,
                        )
                        merged_model = _model_to_merge.merge_and_unload()
                        merged_model.save_pretrained(
                            self.trainer.args.output_dir,
                            safe_serialization=True,
                        )
                        self.trainer.processing_class.save_pretrained(
                            self.trainer.args.output_dir
                        )
                    else:
                        self.trainer.save_model()

                    print("[Training Script] Model saved")
                    self.comms_handler.send_status(
                        {
                            "status": "model_saved",
                            "output_dir": self.trainer.args.output_dir,
                        }
                    )
                elif command.get("command") == "save_checkpoint":
                    print(
                        f"[Training Script] Instructed to save checkpoint {command.get('checkpoint_name')}"
                    )
                    while (
                        time_since_last_step() <= 10
                        or get_time_since_last_queue_pop() <= 10
                    ):
                        print(f"Waiting for steps to finish")
                        print(
                            f"Time since last step: {time_since_last_step():.1f} (needs to be >= 10)"
                        )
                        print(
                            f"Time since last queue pop: {get_time_since_last_queue_pop():.1f} (needs to be >= 10)"
                        )
                        time.sleep(5)
                    if self.trainer.peft_config:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/adapter/"
                        )
                        _model_to_merge = AutoPeftModelForCausalLM.from_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/adapter/",
                            config=self.trainer.peft_config,
                        )
                        merged_model = _model_to_merge.merge_and_unload()
                        merged_model.save_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/",
                            safe_serialization=True,
                        )
                        self.trainer.processing_class.save_pretrained(
                            self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/"
                        )
                    else:
                        self.trainer.save_model(
                            output_dir=self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/"
                        )

                    # Copy checkpoint files to root output directory
                    checkpoint_dir = self.trainer.args.output_dir + f"/checkpoints/{command.get('checkpoint_name')}/"
                    root_dir = self.trainer.args.output_dir 
                    
                    # Copy all files from checkpoint dir to root dir, overwriting if they exist
                    # (effectively saves the checkpoint to the output directory)
                    for item in os.listdir(checkpoint_dir):
                        src = os.path.join(checkpoint_dir, item)
                        dst = os.path.join(root_dir, item)
                        if os.path.isdir(src):
                            if os.path.exists(dst):
                                shutil.rmtree(dst)
                            shutil.copytree(src, dst)
                        else:
                            shutil.copy2(src, dst)

                    self.comms_handler.send_status(
                        {
                            "status": "checkpoint_saved",
                            "checkpoint_name": command.get("checkpoint_name"),
                            "output_dir": self.trainer.args.output_dir
                            + f"/checkpoints/{command.get('checkpoint_name')}/",
                        }
                    )
                    self.comms_handler.send_status(
                        {
                            "status": "model_saved",
                            "output_dir": self.trainer.args.output_dir,
                        }
                    )


        except Exception as e:
            print(e)
            self.comms_handler.send_status({"status": "error", "error": str(e)})

    def _monitor_broadcasts(self):
        """Background thread that monitors for broadcasts from the server."""
        if not self.comms_handler:
            return
        try:
            for broadcast in self.comms_handler.receive_broadcast():
                print(f"!!!Received broadcast: {broadcast}")
                if broadcast.get("message") == "terminate":
                    # self.trainer.control.should_training_stop = True
                    # self.comms_handler.send_status(
                    #     {
                    #         "status": "Received termination command",
                    #         "process_id": self.trainer.accelerator.process_index,
                    #     }
                    # )
                    if self.trainer.accelerator.is_main_process:
                        self.trainer.accelerator.end_training()
        except Exception as e:
            self.comms_handler.send_status({"status": "error", "error": str(e)})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    pipe_args = parser.add_argument_group("Comms arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int, required=True)
    pipe_args.add_argument("--status_port", type=int, required=True)
    pipe_args.add_argument("--data_port", type=int, required=True)
    pipe_args.add_argument("--broadcast_port", type=int, required=True)
    pipe_args.add_argument("--handshake_port", type=int, required=True)
    pipe_args.add_argument("--vllm_group_port", type=int, required=True)
    pipe_args.add_argument("--vllm_port", type=int, required=True)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--model",
        type=str,
        help="Model to use for training",
    )
    training_args.add_argument(
        "--trl_train_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )
    training_args.add_argument(
        "--arbor_train_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )

    args = parser.parse_args()

    if args.debug:
        server_comms_handler = ArborServerCommsHandler(
            host=args.host,
        )

        args.command_port = server_comms_handler.command_port
        args.status_port = server_comms_handler.status_port
        args.data_port = server_comms_handler.data_port
        args.broadcast_port = server_comms_handler.broadcast_port
        args.handshake_port = server_comms_handler.handshake_port

        def debug_data_generator():
            tldr_dataset = load_dataset("trl-lib/tldr", split="train")
            idx = 0
            for item in tldr_dataset:
                input_messages = [{"role": "user", "content": item["prompt"]}]
                completions = [
                    {
                        "role": "assistant",
                        "content": "This is a test completion"
                        + hex(random.randint(0, 0xFFFFFF))[2:],
                    }
                    for _ in range(8)
                ]

                rewards = [-abs(20 - len(c["content"])) for c in completions]
                batch = []
                for completion, reward in zip(completions, rewards):
                    batch.append(
                        {
                            "messages": input_messages,
                            "completion": completion,
                            "reward": reward,
                        }
                    )
                server_comms_handler.send_data(batch)
                time.sleep(1)

                if idx >= 25:
                    server_comms_handler.send_command({"command": "save_model"})

        debug_thread = threading.Thread(target=debug_data_generator, daemon=True)
        debug_thread.start()

        def status_listener():
            # Need to set subscription for PUB/SUB pattern
            server_comms_handler.status_socket.setsockopt_string(zmq.SUBSCRIBE, "")
            for status in server_comms_handler.receive_status():
                print(f"Status: {status}")

        status_listener_thread = threading.Thread(target=status_listener, daemon=True)
        status_listener_thread.start()

    try:
        trl_train_args = {**(args.trl_train_kwargs or {})}
        arbor_train_args = {**(args.arbor_train_kwargs or {})}

        # TODO: These assertions should be done in some better way
        assert "output_dir" in trl_train_args, "output_dir is required"
        if "gradient_checkpointing_kwargs" in trl_train_args and arbor_train_args.get(
            "lora", False
        ):
            print(
                "Setting gradient_checkpointing_kwargs to use_reentrant=False for LORA training"
            )
            trl_train_args["gradient_checkpointing_kwargs"] = {
                **(trl_train_args.get("gradient_checkpointing_kwargs") or {}),
                "use_reentrant": False,
            }

        lora_config = None
        if arbor_train_args.get("lora", False):
            print("Using LORA for PEFT")
            lora_config = LoraConfig(
                r=16,
                lora_alpha=64,
                target_modules=[
                    "q_proj",
                    "k_proj",
                    "v_proj",
                    "o_proj",
                    "up_proj",
                    "down_proj",
                    "gate_proj",
                ],
                task_type="CAUSAL_LM",
                lora_dropout=0.05,
                inference_mode=False,
            )

        training_args = GRPOConfig(
            dataloader_num_workers=0,
            shuffle_dataset=False,
            vllm_server_port=args.vllm_port,
            **trl_train_args,
        )

        weight_update_callback = WeightUpdateCallback()
        trainer = ArborGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=BlockingQueueDataset(None, None),
            callbacks=[LastStepTimeCallback(), weight_update_callback],
            peft_config=lora_config,
            vllm_group_port=args.vllm_group_port,
            **arbor_train_args,
        )
        # Create client handler
        comms_handler = ArborScriptCommsHandler(
            host=args.host,
            command_port=args.command_port,
            status_port=args.status_port,
            data_port=args.data_port,
            broadcast_port=args.broadcast_port,
            handshake_port=args.handshake_port,
            is_main_process=trainer.accelerator.is_main_process,
        )
        weight_update_callback.set_comms_handler(comms_handler)
        trainer.comms_handler = comms_handler

        # Initialize the dataset with the actual accelerator
        trainer.train_dataset = BlockingQueueDataset(
            accelerator=trainer.accelerator,
            comms_handler=trainer.comms_handler,
        )

        command_monitor = CommandMonitor(
            comms_handler=comms_handler,
            trainer=trainer,
            base_model_name=args.model,
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
        trainer.accelerator.end_training()
        comms_handler.close()


if __name__ == "__main__":
    main()
