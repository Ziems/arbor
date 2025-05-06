import os
import random
import string
import time
from datetime import datetime
from pathlib import Path

from arbor.server.api.models.schemas import FineTuneRequest
from arbor.server.core.config import Settings
from arbor.server.services.file_manager import FileManager
from arbor.server.services.job_manager import Job, JobEvent, JobStatus


class TrainingManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def make_output_dir(self, request: FineTuneRequest):
        model_name = request.model.split("/")[-1].lower()
        suffix = (
            request.suffix
            if request.suffix is not None
            else "".join(random.choices(string.ascii_letters + string.digits, k=6))
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"ft:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.settings.STORAGE_PATH).resolve() / "models" / name)

    def find_train_args_sft(self, request: FineTuneRequest, file_manager: FileManager):
        file = file_manager.get_file(request.training_file)
        if file is None:
            raise ValueError(f"Training file {request.training_file} not found")

        data_path = file["path"]
        file_manager.validate_file_format_sft(data_path)

        name, output_dir = self.make_output_dir(request)

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
        train_kwargs = {"packing": False}
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def find_train_args_dpo(self, request: FineTuneRequest, file_manager: FileManager):
        file = file_manager.get_file(request.training_file)
        if file is None:
            raise ValueError(f"Training file {request.training_file} not found")

        data_path = file["path"]
        file_manager.validate_file_format_dpo(data_path)

        name, output_dir = self.make_output_dir(request)

        default_train_kwargs = {
            "device": "cuda:2",
            "use_peft": False,
            "num_train_epochs": 5,
            "per_device_train_batch_size": 1,
            "gradient_accumulation_steps": 8,
            "learning_rate": 1e-5,
            "packing": True,
            "bf16": True,
            "output_dir": output_dir,
            "train_data_path": data_path,
            "prompt_length": 1024,
            "max_seq_length": 1512,
            "use_peft": False,
        }

        # https://www.philschmid.de/dpo-align-llms-in-2024-with-trl#3-align-llm-with-trl-and-the-dpotrainer

        train_kwargs = request.model_dump(exclude_unset=True)
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def fine_tune(self, request: FineTuneRequest, job: Job, file_manager: FileManager):

        job.status = JobStatus.RUNNING
        job.add_event(
            JobEvent(level="info", message="Starting fine-tuning job", data={})
        )

        fine_tune_type = request.method["type"]
        if fine_tune_type == "dpo":
            self.dpo_fine_tune(request, job, file_manager)
        else:
            self.sft_fine_tune(request, job, file_manager)

    def dpo_fine_tune(
        self, request: FineTuneRequest, job: Job, file_manager: FileManager
    ):
        try:

            job.status = JobStatus.RUNNING
            job.add_event(
                JobEvent(level="info", message="Starting fine-tuning job", data={})
            )

            train_kwargs = self.find_train_args_dpo(request, file_manager)
            import torch
            from transformers import (
                AutoModelForCausalLM,
                AutoTokenizer,
                TrainingArguments,
            )
            from trl import DPOConfig, DPOTrainer, setup_chat_format

            device = train_kwargs.get("device", None)
            if device is None:
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )

            job.add_event(
                JobEvent(level="info", message=f"Using device: {device}", data={})
            )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=request.model, device_map="auto"
            )
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=request.model
            )

            try:
                model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
            except Exception:
                pass

            if tokenizer.pad_token_id is None:
                job.add_event(
                    JobEvent(
                        level="info", message="Adding pad token to tokenizer", data={}
                    )
                )
                tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

            hf_dataset = dataset_from_file(train_kwargs["train_data_path"])
            train_dataset = hf_dataset

            use_peft = train_kwargs.get("use_peft", False)
            peft_config = None

            if use_peft:
                from peft import LoraConfig

                peft_config = LoraConfig(
                    lora_alpha=128,
                    lora_dropout=0.05,
                    r=256,
                    bias="none",
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                )

            dpo_args = DPOConfig(
                output_dir=train_kwargs["output_dir"],
                num_train_epochs=train_kwargs["num_train_epochs"],
            )

            trainer = DPOTrainer(
                model=model,
                args=dpo_args,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                peft_config=peft_config,
            )

            trainer.train()

            if job.status == JobStatus.PENDING_PAUSE:
                trainer.accelerator.save_state(output_dir=dpo_args.output_dir)
                current_step = trainer.state.global_step
                job.status = JobStatus.PAUSED
                job.add_event(
                    JobEvent(
                        level="info",
                        message="Training paused",
                        data={"step": current_step},
                    )
                )

            while job.status == JobStatus.PAUSED:
                time.sleep(1)  # Sleep to avoid busy waiting
                if job.status == JobStatus.PENDING_RESUME:
                    job.status = JobStatus.RUNNING
                    job.add_event(JobEvent(level="info", message="Resuming training"))
                    trainer.accelerator.load_state(input_dir=dpo_args.output_dir)
                    trainer.train(resume_from_checkpoint=True)

            if job.status == JobStatus.PENDING_CANCEL:
                job.status = JobStatus.CANCELLED
                job.add_event(JobEvent(level="info", message="Training cancelled"))

                _cleanup(model, tokenizer, trainer)
                raise Exception(
                    "Training cancelled"
                )  # not sure if this should be raised or just return None

            job.add_event(
                JobEvent(level="info", message="Training completed successfully")
            )

            job.add_event(JobEvent(level="info", message="Saving model", data={}))
            # Save the model!
            trainer.save_model()
            job.add_event(
                JobEvent(
                    level="info",
                    message="Model saved",
                    data={"location": dpo_args.output_dir},
                )
            )

            MERGE = True
            if use_peft and MERGE:
                from peft import AutoPeftModelForCausalLM

                # Load PEFT model on CPU
                model_ = AutoPeftModelForCausalLM.from_pretrained(
                    pretrained_model_name_or_path=dpo_args.output_dir,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                )

                merged_model = model_.merge_and_unload()
                merged_model.save_pretrained(
                    dpo_args.output_dir, safe_serialization=True, max_shard_size="5GB"
                )

            _cleanup(model, tokenizer, trainer)

            job.status = JobStatus.SUCCEEDED
            job.fine_tuned_model = dpo_args.output_dir
        except Exception as e:
            job.add_event(
                JobEvent(level="error", message=f"Training failed: {str(e)}", data={})
            )
            job.status = JobStatus.FAILED
            raise
        finally:
            pass

        return dpo_args.output_dir

    def sft_fine_tune(
        self, request: FineTuneRequest, job: Job, file_manager: FileManager
    ):

        try:
            train_kwargs = self.find_train_args_sft(request, file_manager)

            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from trl import SFTConfig, SFTTrainer, setup_chat_format

            device = train_kwargs.get("device", None)
            if device is None:
                device = (
                    "cuda"
                    if torch.cuda.is_available()
                    else "mps" if torch.backends.mps.is_available() else "cpu"
                )
            job.add_event(
                JobEvent(level="info", message=f"Using device: {device}", data={})
            )

            model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=request.model
            ).to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=request.model
            )

            # Set up the chat format; generally only for non-chat model variants, hence the try-except.
            try:
                model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
            except Exception:
                pass

            if tokenizer.pad_token_id is None:
                job.add_event(
                    JobEvent(
                        level="info", message="Adding pad token to tokenizer", data={}
                    )
                )
                tokenizer.add_special_tokens({"pad_token": "[!#PAD#!]"})

            job.add_event(JobEvent(level="info", message="Creating dataset", data={}))
            if (
                "max_seq_length" not in train_kwargs
                or train_kwargs["max_seq_length"] is None
            ):
                train_kwargs["max_seq_length"] = 4096
                job.add_event(
                    JobEvent(
                        level="info",
                        message=f"The 'train_kwargs' parameter didn't include a 'max_seq_length', defaulting to {train_kwargs['max_seq_length']}",
                        data={},
                    )
                )

            job.add_event(JobEvent(level="info", message="Tokenizing dataset", data={}))
            hf_dataset = dataset_from_file(train_kwargs["train_data_path"])

            def tokenize_function(example):
                return encode_sft_example(
                    example, tokenizer, train_kwargs["max_seq_length"]
                )

            tokenized_dataset = hf_dataset.map(tokenize_function, batched=False)
            tokenized_dataset.set_format(type="torch")
            tokenized_dataset = tokenized_dataset.filter(
                lambda example: (example["labels"] != -100).any()
            )

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

            job.add_event(JobEvent(level="info", message="Starting training"))
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=tokenized_dataset,
                peft_config=peft_config,
            )

            # Train!
            trainer.train()

            if job.status == JobStatus.PENDING_PAUSE:
                trainer.accelerator.save_state(output_dir=sft_config.output_dir)
                current_step = trainer.state.global_step
                job.status = JobStatus.PAUSED
                job.add_event(
                    JobEvent(
                        level="info",
                        message="Training paused",
                        data={"step": current_step},
                    )
                )

            while job.status == JobStatus.PAUSED:
                time.sleep(1)  # Sleep to avoid busy waiting
                if job.status == JobStatus.PENDING_RESUME:
                    job.status = JobStatus.RUNNING
                    job.add_event(JobEvent(level="info", message="Resuming training"))
                    trainer.accelerator.load_state(input_dir=sft_config.output_dir)
                    trainer.train(resume_from_checkpoint=True)

            if job.status == JobStatus.PENDING_CANCEL:
                job.status = JobStatus.CANCELLED
                job.add_event(JobEvent(level="info", message="Training cancelled"))

                _cleanup(model, tokenizer, trainer)

                raise Exception(
                    "Training cancelled"
                )  # not sure if this should be raised or just return None

            job.add_event(
                JobEvent(level="info", message="Training completed successfully")
            )

            job.add_event(JobEvent(level="info", message="Saving model", data={}))
            # Save the model!
            trainer.save_model()
            job.add_event(
                JobEvent(
                    level="info",
                    message="Model saved",
                    data={"location": sft_config.output_dir},
                )
            )

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

            _cleanup(model, tokenizer, trainer)

            job.status = JobStatus.SUCCEEDED
            job.fine_tuned_model = sft_config.output_dir
        except Exception as e:
            job.add_event(
                JobEvent(level="error", message=f"Training failed: {str(e)}", data={})
            )
            job.status = JobStatus.FAILED
            raise
        finally:
            pass

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
