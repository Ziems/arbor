import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from arbor.server.services.comms.comms import (
    ArborScriptCommsHandler,
)
from arbor.server.services.scripts.utils.arg_parser import get_training_arg_parser
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


def main():
    parser = get_training_arg_parser()
    args = parser.parse_args()

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

        if "max_seq_length" not in trl_train_args:
            trl_train_args["max_seq_length"] = 4096
            logger.info(
                f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {trl_train_args['max_seq_length']}"
            )

        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model
        )

        train_data = load_train_data(arbor_train_args["train_data_path"])
        hf_dataset = Dataset.from_list(train_data)

        def tokenize_function(example):
            return encode_sft_example(
                example, tokenizer, trl_train_args["max_seq_length"]
            )

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
        tokenized_dataset.set_format(type="torch")
        tokenized_dataset = tokenized_dataset.filter(
            lambda example: (example["labels"] != -100).any()
        )

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
        else:
            trl_train_args["report_to"] = "none"

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

        logger.info("Starting training...")
        trainer.train()

        trainer.save_model()
        logger.info("Model saved")

        MERGE = True
        if arbor_train_args.get("lora", False) and MERGE:
            from peft import AutoPeftModelForCausalLM

            # Load PEFT model on CPU
            model_ = AutoPeftModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=training_args.output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            merged_model = model_.merge_and_unload()
            merged_model.save_pretrained(
                training_args.output_dir, safe_serialization=True, max_shard_size="5GB"
            )

    except KeyboardInterrupt:
        logger.info("Received interrupt, shutting down...")
    except Exception as e:
        logger.error(f"Training error: {e}")
        comms_handler.send_status({"status": "error", "error": str(e)})
        raise e
    finally:
        if trainer:
            trainer.accelerator.end_training()
        if comms_handler:
            comms_handler.close()


def encode_sft_example(example, tokenizer, max_seq_length):
    """
    This function encodes a single example into a format that can be used for sft training.
    Here, we assume each example has a 'messages' field. Each message in it is a dict with 'role' and 'content' fields.
    We use the `apply_chat_template` function from the tokenizer to tokenize the messages and prepare the input and label tensors.

    Code obtained from the allenai/open-instruct repository: https://github.com/allenai/open-instruct/blob/4365dea3d1a6111e8b2712af06b22a4512a0df88/open_instruct/finetune.py
    """
    import torch

    messages = example["messages"]
    if len(messages) == 0:
        raise ValueError("messages field is empty.")
    input_ids = tokenizer.apply_chat_template(
        conversation=messages,
        tokenize=True,
        return_tensors="pt",
        padding=False,
        truncation=True,
        max_length=max_seq_length,
        add_generation_prompt=False,
    )
    labels = input_ids.clone()
    # mask the non-assistant part for avoiding loss
    for message_idx, message in enumerate(messages):
        if message["role"] != "assistant":
            # we calculate the start index of this non-assistant message
            if message_idx == 0:
                message_start_idx = 0
            else:
                message_start_idx = tokenizer.apply_chat_template(
                    conversation=messages[
                        :message_idx
                    ],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if (
                message_idx < len(messages) - 1
                and messages[message_idx + 1]["role"] == "assistant"
            ):
                # for intermediate messages that follow with an assistant message, we need to
                # set `add_generation_prompt=True` to avoid the assistant generation prefix being included in the loss
                # (e.g., `<|assistant|>`)
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=True,
                ).shape[1]
            else:
                # for the last message or the message that doesn't follow with an assistant message,
                # we don't need to add the assistant generation prefix
                message_end_idx = tokenizer.apply_chat_template(
                    conversation=messages[: message_idx + 1],
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # set the label to -100 for the non-assistant part
            labels[:, message_start_idx:message_end_idx] = -100
            if max_seq_length and message_end_idx >= max_seq_length:
                break
    attention_mask = torch.ones_like(input_ids)
    return {
        "input_ids": input_ids.flatten(),
        "labels": labels.flatten(),
        "attention_mask": attention_mask.flatten(),
    }


def load_train_data(data_path):
    """
    Load training data from a file path.

    Args:
        data_path (str): Path to the training data file (JSONL format)

    Returns:
        list: List of training examples with 'messages' field
    """
    import json

    train_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                example = json.loads(line)
                train_data.append(example)

    return train_data


if __name__ == "__main__":
    main()
