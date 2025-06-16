###############################################################################
# Initial Versions of this File Borrowed from Will Brown's Verifiers Library  #
# https://github.com/willccbb/verifiers                                       #
###############################################################################

import argparse
import json
import random
import signal
import sys
import threading
import time
from typing import Any, List, Optional, Union

import torch
import trl.extras.vllm_client
import zmq
from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset, load_dataset
from peft import LoraConfig, PeftConfig
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
from arbor.server.services.scripts.utils.callbacks import WeightUpdateCallback
from arbor.server.services.scripts.utils.comms_monitors import CommandMonitor
from arbor.server.services.scripts.utils.dataset import BlockingRotatingQueueDataset
from arbor.server.services.scripts.utils.ingestion_monitor import IngestionMonitor

trl.extras.vllm_client.VLLMClient = VLLMClient

from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)

if is_wandb_available():
    import wandb


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
        vllm_group_port: Optional[int] = None,
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
            logger.info(
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
        mode = "train" if self.model.training else "eval"

        # Process prompts and completions
        prompt_completion_texts = []
        for example in batch:
            prompt_completion_texts.append(
                maybe_apply_chat_template(
                    {
                        "prompt": example["messages"],
                        "completion": (
                            example["completion"]
                            if isinstance(example["completion"], list)
                            else [example["completion"]]
                        ),
                    },
                    self.processing_class,
                )
            )

        # Tokenize prompts
        prompts_text = [
            prompt_completion_text["prompt"]
            for prompt_completion_text in prompt_completion_texts
        ]
        prompt_inputs = self.processing_class(
            prompts_text,
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
        completions_text = [
            prompt_completion_text["completion"]
            for prompt_completion_text in prompt_completion_texts
        ]
        completion_inputs = self.processing_class(
            completions_text,
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(device)
        completion_ids, completion_mask = (
            completion_inputs["input_ids"],
            completion_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            if prompt_ids.shape[1] > self.max_prompt_length:
                logger.info(f"Truncating prompt to {self.max_prompt_length} tokens")
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.max_completion_length is not None:
            if completion_ids.shape[1] > self.max_completion_length:
                logger.info(
                    f"Truncating completion to {self.max_completion_length} tokens"
                )
            completion_ids = completion_ids[:, : self.max_completion_length]
            completion_mask = completion_mask[:, : self.max_completion_length]

        prompt_ids = broadcast_object_list(prompt_ids)
        prompt_mask = broadcast_object_list(prompt_mask)
        completion_ids = broadcast_object_list(completion_ids)
        completion_mask = broadcast_object_list(completion_mask)

        process_slice = slice(
            self.accelerator.process_index * len(batch),
            (self.accelerator.process_index + 1) * len(batch),
        )

        prompt_ids = prompt_ids[process_slice]
        prompt_mask = prompt_mask[process_slice]
        completion_ids = completion_ids[process_slice]
        completion_mask = completion_mask[process_slice]

        is_eos = completion_ids == self.processing_class.eos_token_id

        # Sum along sequence dimension (dim=1) to get completion length per sequence, used for logging
        completion_lengths = completion_mask.sum(1)

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logger.info(
            f"prompt_completion_ids.shape (after truncation, if enabled): {prompt_completion_ids.shape}, prompt_ids.shape: {prompt_ids.shape}, completion_ids.shape: {completion_ids.shape}"
        )

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if (
                self.num_iterations > 1
                or self.args.steps_per_generation
                > self.args.gradient_accumulation_steps
            ):
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

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
        is_std_zero = torch.isclose(
            std_grouped_rewards, torch.zeros_like(std_grouped_rewards)
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
        all_process_advantages = (
            advantages.clone()
        )  # keep the aggregated advantages for logging
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self.state.num_input_tokens_seen += (
                self.accelerator.gather(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths, mean, min, max
        agg_completion_lengths = self.accelerator.gather(completion_lengths)
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_lengths.float().max().item()
        )

        # Identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather(is_eos.any(dim=1))
        term_completion_lengths = agg_completion_lengths[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_lengths) / len(
            agg_completion_lengths
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if (
            len(term_completion_lengths) == 0
        ):  # edge case where no terminated sequences are found
            term_completion_lengths = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_lengths.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_lengths.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_lengths.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())
        self._metrics[mode]["frac_reward_zero_std"].append(
            is_std_zero.float().mean().item()
        )

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        self._textual_logs["advantages"].extend(all_process_advantages.tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
        }


class LastStepTimeCallback(TrainerCallback):
    "A callback that prints a message at the beginning of training"

    def __init__(self, ingestion_monitor: IngestionMonitor):
        self.ingestion_monitor = ingestion_monitor

    def on_step_end(self, args, state, control, **kwargs):
        logger.info(
            f"Time since last step: {self.ingestion_monitor.time_since_last_step()}"
        )
        self.ingestion_monitor.set_last_step_time()


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
        # python grpo_training.py --debug
        #  --command_port 0 --status_port 0
        #  --data_port 0 --broadcast_port 0
        #  --handshake_port 0 --model Qwen/Qwen3-0.6B
        #  --trl_train_kwargs '{"output_dir": ".", "report_to": "none"}'
        server_comms_handler = ArborServerCommsHandler(
            host=args.host,
        )

        args.command_port = server_comms_handler.command_port
        args.status_port = server_comms_handler.status_port
        args.data_port = server_comms_handler.data_port
        args.broadcast_port = server_comms_handler.broadcast_port
        args.handshake_port = server_comms_handler.handshake_port

        handshake_thread = threading.Thread(
            target=server_comms_handler.wait_for_clients, args=(1,), daemon=True
        )
        handshake_thread.start()

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
                logger.info(f"Status: {status}")

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
            logger.info(
                "Setting gradient_checkpointing_kwargs to use_reentrant=False for LORA training"
            )
            trl_train_args["gradient_checkpointing_kwargs"] = {
                **(trl_train_args.get("gradient_checkpointing_kwargs") or {}),
                "use_reentrant": False,
            }

        lora_config = None
        if arbor_train_args.get("lora", False):
            logger.info("Using LORA for PEFT")
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

        if "report_to" in trl_train_args and trl_train_args["report_to"] == "wandb":
            import wandb

            if "wandb_kwargs" in arbor_train_args and arbor_train_args["wandb_kwargs"]:
                wandb.init(**arbor_train_args["wandb_kwargs"])

        training_args = GRPOConfig(
            dataloader_num_workers=0,
            shuffle_dataset=False,
            vllm_server_port=args.vllm_port,
            **trl_train_args,
        )

        # Create ingestion monitor
        ingestion_monitor = IngestionMonitor()

        train_dataset = BlockingRotatingQueueDataset(
            ingestion_monitor=ingestion_monitor,
        )

        weight_update_callback = WeightUpdateCallback(
            ingestion_monitor=ingestion_monitor,
        )

        trainer = ArborGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=train_dataset,
            callbacks=[LastStepTimeCallback(ingestion_monitor), weight_update_callback],
            peft_config=lora_config,
            vllm_group_port=args.vllm_group_port,
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

        train_dataset.set_comms_handler(comms_handler)
        train_dataset.set_accelerator(trainer.accelerator)

        weight_update_callback.set_comms_handler(comms_handler)
        weight_update_callback.set_trainer(trainer)
        trainer.comms_handler = comms_handler

        command_monitor = CommandMonitor(
            comms_handler=comms_handler,
            trainer=trainer,
            base_model_name=args.model,
            ingestion_monitor=ingestion_monitor,
        )
        command_monitor.start()

        # Add signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            logger.info(f"\nReceived signal {signum}. Initiating graceful shutdown...")
            logger.info("Ending training...")
            trainer.accelerator.end_training()
            logger.info("Closing communications...")
            comms_handler.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Starting training...")
        try:
            trainer.train()
        except Exception as e:
            logger.error(f"Error during training: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            raise

    except KeyboardInterrupt:
        logger.info("\nReceived interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        logger.info("Cleaning up resources...")
        trainer.accelerator.end_training()
        comms_handler.close()
        logger.info("Cleanup complete")


if __name__ == "__main__":
    main()
