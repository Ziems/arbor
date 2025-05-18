import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer, setup_chat_format


def dataset_from_file(data_path):
    """
    Creates a HuggingFace Dataset from a JSONL file.
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset


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


def _cleanup(model, tokenizer, trainer):
    import gc

    import torch

    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


class ArborSFTTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    pipe_args = parser.add_argument_group("Comms arguments")
    pipe_args.add_argument("--host", default="localhost")
    pipe_args.add_argument("--command_port", type=int, required=True)
    pipe_args.add_argument("--status_port", type=int, required=True)
    pipe_args.add_argument("--data_port", type=int, required=True)
    pipe_args.add_argument("--broadcast_port", type=int, required=True)

    training_args = parser.add_argument_group("Training arguments")
    training_args.add_argument(
        "--model",
        type=str,
        help="Model to use for training",
    )
    training_args.add_argument(
        "--train_data_path",
        type=str,
        help="Path to the training data",
    )
    training_args.add_argument(
        "--trl_config_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )
    training_args.add_argument(
        "--trainer_kwargs",
        type=json.loads,
        help="Training arguments as a JSON string",
    )

    args = parser.parse_args()

    try:
        trl_config_args = {**(args.trl_config_kwargs or {})}
        trainer_args = {**(args.trainer_kwargs or {})}
        trl_config_args["bf16"] = True

        # TODO: These assertions should be done in some better way
        assert "output_dir" in trl_config_args, "output_dir is required"
        assert "train_data_path" in trainer_args, "train_data_path is required"

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=args.model
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=args.model
        )

        # Set up the chat format; generally only for non-chat model variants, hence the try-except.
        try:
            model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
        except Exception:
            pass

        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

        if (
            "max_seq_length" not in trainer_args
            or trainer_args["max_seq_length"] is None
        ):
            trainer_args["max_seq_length"] = 4096

        hf_dataset = dataset_from_file(trainer_args["train_data_path"])

        def tokenize_function(example):
            return encode_sft_example(
                example, tokenizer, trainer_args["max_seq_length"]
            )

        tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
        tokenized_dataset.set_format(type="torch")
        tokenized_dataset = tokenized_dataset.filter(
            lambda example: (example["labels"] != -100).any()
        )

        USE_PEFT = trainer_args.get("use_peft", False)
        peft_config = None

        if USE_PEFT:
            from peft import LoraConfig

            rank_dimension = 32
            lora_alpha = 64
            lora_dropout = 0.05

            peft_config = LoraConfig(
                r=rank_dimension,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )

        config = SFTConfig(**trl_config_args)
        trainer = ArborSFTTrainer(
            model=args.model,
            trl_config=config,
            **trainer_args,
            train_dataset=tokenized_dataset,
            peft_config=peft_config,
        )
        trainer.train()
        trainer.save_model()

        MERGE = True
        if USE_PEFT and MERGE:
            from peft import AutoPeftModelForCausalLM

            # Load PEFT model on CPU
            model_ = AutoPeftModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=config.output_dir,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
            )

            merged_model = model_.merge_and_unload()
            merged_model.save_pretrained(
                config.output_dir, safe_serialization=True, max_shard_size="5GB"
            )

        _cleanup(model, tokenizer, trainer)

    except Exception as e:
        print(f"Error: {e}")
        raise e
