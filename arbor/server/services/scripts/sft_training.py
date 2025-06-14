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
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--lora", type=bool, default=False)
    args = parser.parse_args()

    try:
        trl_train_kwargs = {**(args.trl_config_kwargs or {})}

        # TODO: These assertions should be done in some better way
        assert "output_dir" in trl_train_kwargs, "output_dir is required"
        if "gradient_checkpointing_kwargs" in trl_train_kwargs and args.lora:
            logger.info(
                "Setting gradient_checkpointing_kwargs to use_reentrant=False for LORA training"
            )
            trl_train_kwargs["gradient_checkpointing_kwargs"] = {
                **(trl_train_kwargs.get("gradient_checkpointing_kwargs") or {}),
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

        training_args = GRPOConfig(
            dataloader_num_workers=0,
            shuffle_dataset=False,
            **trl_train_args,
        )

        trainer = ArborGRPOTrainer(
            model=args.model,
            args=training_args,
            train_dataset=BlockingQueueDataset(None, None),
            callbacks=[LastStepTimeCallback()],
            peft_config=lora_config,
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

        logger.info("Starting training...")
        trainer.train()

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
