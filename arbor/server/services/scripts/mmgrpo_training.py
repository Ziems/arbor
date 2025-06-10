import argparse
import json
import signal
import sys
import time
from typing import Any, Optional, Union

import torch
import trl.extras.vllm_client
from datasets import Dataset, IterableDataset, load_dataset
from peft import LoraConfig, PeftConfig
from torch.utils.data import Dataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from trl.data_utils import maybe_apply_chat_template
from trl.trainer.grpo_trainer import GRPOConfig, GRPOTrainer

from arbor.server.services.comms.comms import ArborScriptCommsHandler
from arbor.server.services.inference.vllm_client import VLLMClient
from arbor.server.services.scripts.utils.comms_monitors import CommandMonitor
from arbor.server.services.scripts.utils.dataset import BlockingQueueDataset

trl.extras.vllm_client.VLLMClient = VLLMClient


class MMGRPOTrainer(GRPOTrainer):
    def __init__(
        self,
        model: Union[str, PreTrainedModel],
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
        self.comms_handler = comms_handler

        # self.vllm_client = None
        # args.use_vllm = True
        # self.use_vllm = True
        # if self.accelerator.is_main_process:
        #     print(
        #         f"Initializing vLLM client with server port {args.vllm_server_port} and group port {vllm_group_port}"
        #     )
        #     self.vllm_client = VLLMClient(
        #         args.vllm_server_host,
        #         args.vllm_server_port,
        #         group_port=vllm_group_port,
        #         connection_timeout=args.vllm_server_timeout,
        #     )
        #     self.vllm_client.init_communicator()

        # # vLLM specific sampling arguments
        # self.guided_decoding_regex = args.vllm_guided_decoding_regex

        # self._last_loaded_step = (
        #     -1
        # )  # tag to avoid useless loading during grad accumulation

        # # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # # synchronize all processes after vLLM has been fully initialized.
        # self.accelerator.wait_for_everyone()

    def get_train_dataloader(self):
        return Trainer.get_train_dataloader(self)

    def _get_train_sampler(self, dataset: Optional[Dataset] = None):
        return Trainer._get_train_sampler(self, dataset)

    def _prepare_inputs(
        self, generation_batch: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:

        return Trainer._prepare_inputs(self, generation_batch)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, Union[torch.Tensor, Any]]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        raise NotImplementedError(
            "_generate_and_score_completions must be implemented by subclass"
        )

    def _compute_loss(self, model, inputs, return_outputs=False):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = (
            inputs["completion_ids"],
            inputs["completion_mask"],
        )
        inputs_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)

        per_token_logps = self._get_per_token_logps(
            model, inputs_ids, attention_mask, logits_to_keep
        )

        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, inputs_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, inputs_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        advantages = inputs["advantages"]
        # When using num_iterations == 1 and steps_per_generation <= gradient_accumulation_steps
        # old_per_token_logps == per_token_logps, so we can skip it's computation
        # (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.args.delta is not None:
            coef_1 = torch.clamp(coef_1, max=self.args.delta)

        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl

        if self.loss_type == "grpo":
            loss = (
                (per_token_loss * completion_mask).sum(-1)
                / completion_mask.sum(-1).clamp(min=1.0)
            ).mean()
        elif self.loss_type == "bnpo":
            loss = (
                per_token_loss * completion_mask
            ).sum() / completion_mask.sum().clamp(min=1.0)
        elif self.loss_type == "dr_grpo":
            loss = (per_token_loss * completion_mask).sum() / (
                per_token_loss.size(0) * self.max_completion_length
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Log the metrics
        mode = "train" if self.model.training else "eval"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather(mean_kl).nanmean().item()
            )

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        # gathered_low_clip = self.accelerator.gather(low_clip)
        # self._metrics[mode]["clip_ratio/low_mean"].append(
        #     gathered_low_clip.nanmean().item()
        # )
        # self._metrics[mode]["clip_ratio/low_min"].append(
        #     nanmin(gathered_low_clip).item()
        # )
        # gathered_high_clip = self.accelerator.gather(high_clip)
        # self._metrics[mode]["clip_ratio/high_mean"].append(
        #     gathered_high_clip.nanmean().item()
        # )
        # self._metrics[mode]["clip_ratio/high_max"].append(
        #     nanmax(gathered_high_clip).item()
        # )
        # gathered_clip_ratio = self.accelerator.gather(clip_ratio)
        # self._metrics[mode]["clip_ratio/region_mean"].append(
        #     gathered_clip_ratio.nanmean().item()
        # )

        return loss


class WeightUpdateCallback(TrainerCallback):
    """A callback that sends weight update completion status after each step"""

    def __init__(self):
        self.comms_handler = None
        self.trainer = None

    def set_comms_handler(self, comms_handler: ArborScriptCommsHandler):
        self.comms_handler = comms_handler

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_step_end(self, args, state, control, **kwargs):
        if self.comms_handler and self.comms_handler.is_main_process and self.trainer:
            if state.global_step != self.trainer._last_loaded_step:
                print("Updating inference model...")
                self.comms_handler.send_status({"status": "weight_update_start"})


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
        pass

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
        trainer = MMGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=BlockingQueueDataset(None, None),
            callbacks=[weight_update_callback],
            peft_config=lora_config,
            vllm_group_port=args.vllm_group_port,
            **arbor_train_args,
        )

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
        weight_update_callback.set_trainer(trainer)
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
        command_monitor.start()

        # Add signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            print(f"\nReceived signal {signum}. Initiating graceful shutdown...")
            print("Ending training...")
            trainer.accelerator.end_training()
            print("Closing communications...")
            comms_handler.close()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        print("Training...")
        trainer.train()

    except Exception as e:
        print(f"Error: {e}")
        comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        print("Cleaning up resources...")
        trainer.accelerator.end_training()
        comms_handler.close()
        print("Cleanup complete")


# Example usage:
if __name__ == "__main__":
    main()
