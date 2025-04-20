import time
import random
import string
from datetime import datetime
from pathlib import Path
from arbor.server.api.models.schemas import DPORequest
from arbor.server.core.config import Settings
from arbor.server.services.job_manager import Job, JobStatus, JobEvent
from arbor.server.services.file_manager import FileManager

class DPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings

    def make_output_dir(self, request: DPORequest):
        model_name = request.model.split('/')[-1].lower()
        suffix = request.suffix if request.suffix is not None else ''.join(random.choices(string.ascii_letters + string.digits, k=6))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name = f"dpo:{model_name}:{suffix}:{timestamp}"
        return name, str(Path(self.settings.STORAGE_PATH).resolve() / "models" / name)
    
    def find_train_args(self, request: DPORequest, file_manager: FileManager):
        file = file_manager.get_file(request.training_file)
        if file is None:
            raise ValueError(f"Training file {request.training_file} not found")

        data_path = file["path"]

        name, output_dir = self.make_output_dir(request)

        default_train_kwargs = {
            "device": None,
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
            "use_peft": False
        }
        # https://www.philschmid.de/dpo-align-llms-in-2024-with-trl#3-align-llm-with-trl-and-the-dpotrainer

        train_kwargs = request.model_dump(exclude_unset=True)
        train_kwargs = {**default_train_kwargs, **(train_kwargs or {})}

        return train_kwargs

    def run_dpo(self, request: DPORequest, job: Job, file_manager: FileManager):
        job.status = JobStatus.RUNNING
        job.add_event(JobEvent(level="info", message="Starting fine-tuning job", data={}))

        train_kwargs = self.find_train_args(request, file_manager)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from trl import setup_chat_format, DPOConfig, DPOTrainer

        device = train_kwargs.get("device", None)
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps" if torch.backends.mps.is_available() else "cpu"
            )

        job.add_event(JobEvent(level="info", message=f"Using device: {device}", data={}))

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=request.model,
            device_map='auto'
        ).to(device)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=request.model)

        try:
            model, tokenizer = setup_chat_format(model=model, tokenizer=tokenizer)
        except Exception:
            pass

        if tokenizer.pad_token_id is None:
            job.add_event(JobEvent(level="info", message="Adding pad token to tokenizer", data={}))
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

 
        # args = DPOConfig(
        #     output_dir=train_kwargs["output_dir"],               # directory to save and repository id
        #     num_train_epochs=train_kwargs["num_train_epochs"],                     # number of training epochs
        #     per_device_train_batch_size=train_kwargs["per_device_train_batch_size"],         # batch size per device during training
        #     per_device_eval_batch_size=4,           # batch size for evaluation
        #     gradient_accumulation_steps=1,          # number of steps before performing a backward/update pass
        #     gradient_checkpointing=True,            # use gradient checkpointing to save memory
        #     optim="adamw_torch_fused",              # use fused adamw optimizer
        #     learning_rate=train_kwargs["learning_rate"],                     # 10x higher LR than QLoRA paper
        #     max_grad_norm=0.3,                      # max gradient norm based on QLoRA paper
        #     warmup_ratio=0.1,                       # warmup ratio based on QLoRA paper
        #     lr_scheduler_type="cosine",             # use cosine learning rate scheduler
        #     logging_steps=25,                       # log every 25 steps
        #     save_steps=500,                         # when to save checkpoint
        #     save_total_limit=2,                     # limit the total amount of checkpoints
        #     evaluation_strategy="steps",            # evaluate every 1000 steps
        #     eval_steps=700,                         # when to evaluate
        #     bf16=train_kwargs["bf16"],                              # use bfloat16 precision
        #     tf32=True,                              # use tf32 precision
        # )
        
        # dpo_args = {
        #     "beta": 0.1,                            # The beta factor in DPO loss. Higher beta means less divergence
        #     "loss_type": "sigmoid"                  # The loss type for DPO.
        # }

        training_args = DPOConfig(
            output_dir=train_kwargs["output_dir"],
            num_train_epochs=train_kwargs["num_train_epochs"],
        )

        trainer = DPOTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            peft_config=peft_config,
        )

        trainer.train()

        

def dataset_from_file(data_path):
    """
    Creates a HuggingFace Dataset from a JSONL file.
    """
    from datasets import load_dataset

    dataset = load_dataset("json", data_files=data_path, split="train")
    return dataset


