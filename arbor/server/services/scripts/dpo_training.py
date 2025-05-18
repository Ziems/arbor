import argparse
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOConfig, DPOTrainer, setup_chat_format


def dataset_from_file(data_path):
    """
    Creates a HuggingFace Dataset from a JSONL file.
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset


def _cleanup(model, tokenizer, trainer):
    import gc

    import torch

    del model
    del tokenizer
    del trainer
    gc.collect()
    torch.cuda.empty_cache()


class ArborDPOTrainer(DPOTrainer):
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

        # tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
        # tokenized_dataset.set_format(type="torch")
        # tokenized_dataset = tokenized_dataset.filter(
        #     lambda example: (example["labels"] != -100).any()
        # )

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

        config = DPOConfig(**trl_config_args)
        trainer = ArborDPOTrainer(
            model=args.model,
            trl_config=config,
            **trainer_args,
            train_dataset=hf_dataset,
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
