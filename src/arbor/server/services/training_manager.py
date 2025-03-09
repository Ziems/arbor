from arbor.server.api.models.schemas import FineTuneRequest
from arbor.server.services.job_manager import Job, JobStatus
from arbor.server.services.file_manager import FileManager

class TrainingManager:
    def __init__(self):
        pass

    def find_train_args(self, request: FineTuneRequest, file_manager: FileManager):
        file = file_manager.get_file(request.training_file)
        if file is None:
            raise ValueError(f"Training file {request.training_file} not found")

        data_path = file["path"]
        output_dir = f"models/{request.model_name}" # TODO: This should be updated to be unique in some way


        default_train_kwargs = {
            "device": None,
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "max_seq_length": None,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
            "train_data_path": data_path,
        }
        train_kwargs = {'packing': False}
        train_kwargs={**default_train_kwargs, **(train_kwargs or {})}
        output_dir = train_kwargs["output_dir"]  # user might have changed the output_dir

        return train_kwargs


    def fine_tune(self, request: FineTuneRequest, job: Job, file_manager: FileManager):
        # Get logger for this job
        logger = job.setup_logger("training")

        job.status = JobStatus.RUNNING
        logger.info("Starting fine-tuning job")

        try:
            train_kwargs = self.find_train_args(request, file_manager)

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
            from trl import SFTConfig, SFTTrainer, setup_chat_format

            device = train_kwargs.get("device", None)
            if device is None:
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            logger.info(f"Using device: {device}")

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=request.model_name
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=request.model_name)

            # Set up the chat format; generally only for non-chat model variants, hence the try-except.
            try:
                model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
            except Exception:
                pass

            if tokenizer.pad_token_id is None:
                logger.info("Adding pad token to tokenizer")
                tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

            logger.info("Creating dataset")
            if "max_seq_length" not in train_kwargs or train_kwargs["max_seq_length"] is None:
                train_kwargs["max_seq_length"] = 4096
                logger.info(f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {train_kwargs['max_seq_length']}")


            hf_dataset = dataset_from_file(train_kwargs["train_data_path"])
            def tokenize_function(example):
                return encode_sft_example(example, tokenizer, train_kwargs["max_seq_length"])
            tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
            tokenized_dataset.set_format(type="torch")
            tokenized_dataset = tokenized_dataset.filter(lambda example: (example["labels"] != -100).any())

            USE_PEFT = train_kwargs.get("use_peft", False)
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

            sft_config = SFTConfig(
                output_dir=train_kwargs["output_dir"],
                num_train_epochs=train_kwargs["num_train_epochs"],
                per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],
                gradient_accumulation_steps=train_kwargs["gradient_accumulation_steps"],
                learning_rate=train_kwargs["learning_rate"],
                max_grad_norm=2.0,  # note that the current SFTConfig default is 1.0
                logging_steps=20,
                warmup_ratio=0.03,
                lr_scheduler_type="constant",
                save_steps=10_000,
                bf16=train_kwargs["bf16"],
                max_seq_length=train_kwargs["max_seq_length"],
                packing=train_kwargs["packing"],
                dataset_kwargs={  # We need to pass dataset_kwargs because we are processing the dataset ourselves
                    "add_special_tokens": False,  # Special tokens handled by template
                    "append_concat_token": False,  # No additional separator needed
                },
            )

            logger.info("Starting training")
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=tokenized_dataset,
                peft_config=peft_config,

            )

            # Train!
            trainer.train()

            # Save the model!
            trainer.save_model()

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

            # Clean up!
            import gc

            del model
            del tokenizer
            del trainer
            gc.collect()
            torch.cuda.empty_cache()

            logger.info("Training completed successfully")
            job.status = JobStatus.COMPLETED
        except Exception as e:
            logger.error(f"Training failed: {str(e)}")
            job.status = JobStatus.FAILED
            raise
        finally:
            job.cleanup_logger()

        return sft_config.output_dir

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
                    conversation=messages[:message_idx],  # here marks the end of the previous messages
                    tokenize=True,
                    return_tensors="pt",
                    padding=False,
                    truncation=True,
                    max_length=max_seq_length,
                    add_generation_prompt=False,
                ).shape[1]
            # next, we calculate the end index of this non-assistant message
            if message_idx < len(messages) - 1 and messages[message_idx + 1]["role"] == "assistant":
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
        "attention_mask": attention_mask.flatten()
    }