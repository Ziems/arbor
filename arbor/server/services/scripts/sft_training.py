import argparse
import json
import random
import threading
import time

import torch
import zmq
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format

from arbor.server.services.scripts.utils.arg_parser import get_training_arg_parser
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = get_training_arg_parser()
    args = parser.parse_args()

    try:
        trl_train_args = {**(args.trl_config_kwargs or {})}
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
        if args.lora:
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

        training_args = SFTConfig(
            output_dir=trl_train_args["output_dir"],
            num_train_epochs=trl_train_args["num_train_epochs"],
            per_device_train_batch_size=trl_train_args["per_device_train_batch_size"],
            gradient_accumulation_steps=trl_train_args["gradient_accumulation_steps"],
            learning_rate=trl_train_args["learning_rate"],
            max_grad_norm=2.0,  # note that the current SFTConfig default is 1.0
            logging_steps=20,
            warmup_ratio=0.03,
            lr_scheduler_type="constant",
            save_steps=10_000,
            bf16=trl_train_args["bf16"],
            max_seq_length=trl_train_args["max_seq_length"],
            packing=trl_train_args["packing"],
            dataset_kwargs={  # We need to pass dataset_kwargs because we are processing the dataset ourselves
                "add_special_tokens": False,  # Special tokens handled by template
                "append_concat_token": False,  # No additional separator needed
            },
        )

        trainer = SFTTrainer(
            model=args.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            peft_config=lora_config,
        )

        logger.info("Starting training...")
        trainer.train()

        trainer.save_model()
        logger.info("Model saved")

        MERGE = True
        if USE_PEFT and MERGE:
            from peft import AutoPeftModelForCausalLM

            # Load PEFT model on CPU
            model_ = AutoPeftModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=sft_config.output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            merged_model = model_.merge_and_unload()
            merged_model.save_pretrained(
                sft_config.output_dir, safe_serialization=True, max_shard_size="5GB"
            )

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Training error: {e}")
        comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        trainer.accelerator.end_training()
        comms_handler.close()


if __name__ == "__main__":
    main()
