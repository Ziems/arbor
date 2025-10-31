import argparse
import json
import logging
import math
import os
import shutil
import threading
import time
from collections import defaultdict, deque
from contextlib import nullcontext
from typing import Any, Optional

import datasets
import deepspeed
import torch
import transformers
import wandb
from accelerate.utils import broadcast_object_list, is_peft_model
from peft import LoraConfig
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoConfig,
    AutoProcessor,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    ProcessorMixin,
)
from transformers.trainer import Trainer
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker
from trl.models import prepare_deepspeed, prepare_peft_model
from trl.trainer.callbacks import SyncRefModelCallback
from trl.trainer.utils import (
    disable_dropout_in_model,
    identity,
    nanmax,
    nanmin,
    pad,
    selective_log_softmax,
)

from arbor.core.logging import get_logger, setup_logging
from arbor.server.services.comms.async_batch_requester import (
    AsyncBatchRequester,
    BatchRequest,
)
from arbor.server.services.comms.async_dataloader_wrapper import AsyncDataLoaderWrapper
from arbor.server.services.comms.control_client import TrainerControlClient
from arbor.server.services.inference.vllm_client import VLLMClient
from arbor.training.grpo.config import ArborGRPOConfig


class _ExternalBatchDataset(Dataset):
    """Minimal dataset used when batches come from external sources."""

    def __init__(self, length: int):
        self._length = max(1, int(length))

    def __len__(self) -> int:  # type: ignore[override]
        return self._length

    def __getitem__(self, index: int) -> dict[str, int]:  # type: ignore[override]
        return {"index": index}


def shuffle_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]],
) -> dict[str, Optional[torch.Tensor]]:
    """
    Shuffles a dictionary of tensors along the first dimension in unison.
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    batch_size = first_tensor.shape[0]
    permutation = torch.randperm(batch_size)
    return {
        key: tensor[permutation] if tensor is not None else None
        for key, tensor in tensor_dict.items()
    }


def split_tensor_dict(
    tensor_dict: dict[str, Optional[torch.Tensor]], num_chunks: int
) -> list[dict[str, Optional[torch.Tensor]]]:
    """
    Splits a dictionary of tensors along the first dimension into `num_chunks` equal parts.
    """
    first_tensor = next(tensor for tensor in tensor_dict.values() if tensor is not None)
    chunk_size = first_tensor.shape[0] // num_chunks
    return [
        {
            key: tensor[i * chunk_size : (i + 1) * chunk_size]
            if tensor is not None
            else None
            for key, tensor in tensor_dict.items()
        }
        for i in range(num_chunks)
    ]


class ArborGRPOTrainer(Trainer):
    def __init__(
        self,
        model: str,
        args: ArborGRPOConfig,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
    ):
        self.logger = get_logger(__name__)
        self.logger.debug("Starting __init__")
        self._checkpoint_condition = threading.Condition()
        self._checkpoint_request_data: Optional[dict[str, Any]] = None
        self._checkpoint_result: Optional[dict[str, Any]] = None
        self._checkpoint_records: list[dict[str, Any]] = []
        self._last_checkpoint_record: Optional[dict[str, Any]] = None
        self._checkpoint_records_lock = threading.Lock()
        self._control_client: Optional[TrainerControlClient] = None

        # Trained model
        model_init_kwargs: dict[str, Any] = args.model_init_kwargs or {}  # type: ignore
        model_id = model
        dtype = model_init_kwargs.get("dtype")
        if isinstance(dtype, torch.dtype) or dtype == "auto" or dtype is None:
            pass  # dtype is already a torch.dtype or "auto" or None
        elif isinstance(dtype, str):  # it's a str, but not "auto"
            dtype = getattr(torch, dtype)
            model_init_kwargs["dtype"] = dtype
        else:
            raise ValueError(
                "Invalid `dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                f"a `torch.dtype` (e.g., 'float32'), but got {dtype}."
            )
        # Disable caching if gradient checkpointing is enabled (not supported)
        config = AutoConfig.from_pretrained(model_id)
        architecture = getattr(transformers, config.architectures[0])
        model: PreTrainedModel = architecture.from_pretrained(
            model_id, **model_init_kwargs
        )

        if args.lora_config is not None:
            model = prepare_peft_model(model, args.lora_config, args)
            # Override sync_ref_model if PEFT is used since ref_model will be None
            if args.sync_ref_model:
                self.logger.warning(
                    "sync_ref_model=True is not compatible with PEFT. Setting sync_ref_model=False."
                )
                args.sync_ref_model = False

                # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)  # type: ignore

        # Processing class
        processing_class = AutoProcessor.from_pretrained(model.config._name_or_path)

        # Handle pad token for processors or tokenizers
        if isinstance(processing_class, ProcessorMixin):
            tokenizer = processing_class.tokenizer
        elif isinstance(processing_class, PreTrainedTokenizerBase):
            tokenizer = processing_class
        else:
            raise TypeError(
                "The `processing_class` must be either a `PreTrainedTokenizerBase` or a `ProcessorMixin`"
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.pad_token = tokenizer.pad_token
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id

        self.per_device_train_batch_size = args.per_device_train_batch_size
        self.max_prompt_length = args.max_prompt_length
        self.max_seq_len = args.max_seq_len  # max sequence length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper
        self.soft_completion_penalty_length = args.soft_completion_penalty_length
        if self.max_completion_length is not None:
            self.logger.warning(
                "max_completion_length is deprecated. Use max_seq_len instead."
            )
            if self.max_seq_len is None and self.max_prompt_length is not None:
                self.max_seq_len = self.max_prompt_length + self.max_completion_length
                self.logger.info(
                    f"max_seq_len is set to {self.max_seq_len} (max_prompt_length={self.max_prompt_length} + max_completion_length={self.max_completion_length})"
                )
            else:
                self.max_seq_len = self.max_completion_length
                self.logger.info(
                    f"max_seq_len is set to {self.max_seq_len} (max_completion_length={self.max_completion_length})"
                )
        if self.max_seq_len is None:
            raise ValueError(
                "max_seq_len is required when max_completion_length is not provided"
            )
        if self.max_prompt_length is None:
            self.max_prompt_length = self.max_seq_len
            self.logger.info(
                f"max_prompt_length is set to {self.max_prompt_length} (max_seq_len={self.max_seq_len})"
            )
        self.max_tokens = args.max_tokens  # max tokens per generation
        if self.max_tokens is None:
            self.max_tokens = self.max_seq_len
            self.logger.info(
                f"max_tokens is set to {self.max_tokens} (max_seq_len={self.max_seq_len})"
            )
        self.num_generations = args.num_generations  # = G in the GRPO paper
        self.max_data_workers = args.max_data_workers
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        # self.use_transformers_paged = args.use_transformers_paged
        self.vllm_mode = args.vllm_mode
        self.vllm_gpu_memory_utilization = (
            args.vllm_gpu_memory_utilization
        )  # only applies to colocation mode
        self.vllm_tensor_parallel_size = (
            args.vllm_tensor_parallel_size
        )  # only applies to colocation mode
        self.vllm_importance_sampling_correction = (
            args.vllm_importance_sampling_correction
        )
        self.vllm_importance_sampling_cap = args.vllm_importance_sampling_cap
        self.use_liger_loss = args.use_liger_loss
        self.loss_type = args.loss_type
        self.scale_rewards = args.scale_rewards
        self.delta = args.delta
        self.importance_sampling_level = args.importance_sampling_level
        self.mask_truncated_completions = args.mask_truncated_completions
        self.top_entropy_quantile = args.top_entropy_quantile
        if self.use_liger_loss and self.top_entropy_quantile < 1.0:
            raise NotImplementedError(
                "Liger Kernels don't currently support masking token positions based on entropy."
            )
        if self.use_liger_loss and not self.importance_sampling_level == "token":
            raise NotImplementedError(
                "Liger Kernels currently only support token-level importance sampling. Please set"
                "`importance_sampling_level` to 'token'."
            )

        # Multi-step
        self.num_iterations = args.num_iterations  # = 𝜇 in the GRPO paper
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )
        # Tracks the number of iterations (forward + backward passes), including those within a grad accum cycle
        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self._step = 0
        # Buffer the batch to reuse generated outputs across multiple updates. For more details, see
        # `_get_train_sampler` and `_prepare_inputs`.
        self._buffered_inputs = None

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        if args.max_steps <= 0:
            raise ValueError(
                "ArborGRPOTrainer requires a finite `max_steps` when batches are supplied externally."
            )
        total_items = args.max_steps * args.gradient_accumulation_steps
        train_dataset = _ExternalBatchDataset(total_items)
        self.logger.debug(
            "Initialized dummy dataset with %s items for external batch results",
            total_items,
        )

        self.logger.debug("Starting super().__init__")

        super().__init__(
            model=model,
            args=args,
            data_collator=identity,  # No data collation is needed in GRPO
            train_dataset=train_dataset,
            eval_dataset=None,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            # In Trainer, `training_step` scales the loss by `gradient_accumulation_steps` only if `compute_loss_func`
            # is None. For DAPO, loss scaling instead depends on the total number of completions tokens across the
            # global accumulated batch. To control scaling ourselves, we must disable Trainer’s built-in scaling. The
            # simplest (though a bit hacky) way is to set `compute_loss_func` to any non-None value, which bypasses
            # that behavior without rewriting `training_step`.
            compute_loss_func="non-None value to disable scaling",
        )
        self.logger.debug("Done with super().__init__")

        # Validate batch sizing to avoid empty per-process slices
        try:
            world_size = self.accelerator.num_processes
        except Exception:
            world_size = 1
        min_required_generations = world_size * int(self.gradient_accumulation_steps)
        if self.num_generations < min_required_generations:
            raise ValueError(
                "num_generations must be >= world_size * gradient_accumulation_steps. "
                f"Got num_generations={self.num_generations}, world_size={world_size}, "
                f"gradient_accumulation_steps={self.gradient_accumulation_steps} → "
                f"min_required={min_required_generations}."
            )

        # Reference model
        self.logger.debug("Starting reference model")
        self.beta = args.beta
        if self.beta == 0.0:
            # If beta is 0.0, the reference model is not needed
            self.ref_model = None
        elif is_peft_model(self.model):
            # If PEFT is used, the reference model is not needed since the adapter can be disabled
            # to revert to the initial model.
            self.ref_model = None
        else:
            # For deepspeed, fsdp or non-distributed models, create a reference model from scratch
            config = AutoConfig.from_pretrained(model_id)
            architecture = getattr(transformers, config.architectures[0])
            self.ref_model = architecture.from_pretrained(model_id, **model_init_kwargs)  # type: ignore

        self.logger.debug("Done with reference model")

        # Disable dropout in the models
        if args.disable_dropout:
            disable_dropout_in_model(self.model)
            if self.ref_model is not None:
                disable_dropout_in_model(self.ref_model)

        # Liger loss
        # TODO: Commented out for now for testing --Noah 9/26/2025
        # if self.use_liger_loss:
        #     if not is_liger_kernel_available():
        #         raise ImportError(
        #             "Liger is required to use `liger_loss` as the GRPO loss. Run `pip install liger-kernel`."
        #         )
        #     # redirect the model.module forward to the model forward to ensure pre-forward hooks are called
        #     self._forward_redirection = _ForwardRedirection()
        #
        #     self.liger_grpo_loss = LigerFusedLinearGRPOLoss(
        #         beta=self.beta,
        #         epsilon_low=self.epsilon_low,
        #         epsilon_high=self.epsilon_high,
        #         temperature=self.temperature,
        #         use_ref_model=self.beta != 0.0,
        #         loss_type=self.loss_type,
        #         max_completion_length=self.max_completion_length,
        #     )

        # Initialize the metrics
        self._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
        self._total_train_tokens = 0
        self.log_completions = args.log_completions
        self.wandb_log_unique_prompts = args.wandb_log_unique_prompts
        self.num_completions_to_print = args.num_completions_to_print

        # maxlen is set to the total number of forward passes per step. This value of `maxlen` ensures we log only the
        # final optimization step.
        maxlen = (
            self.accelerator.num_processes
            * args.per_device_train_batch_size
            * args.gradient_accumulation_steps
        )
        self._textual_logs = {
            "prompt": deque(maxlen=maxlen),
            "completion": deque(maxlen=maxlen),
            "rewards": defaultdict(lambda: deque(maxlen=maxlen)),
        }
        self.vllm_client = None
        self.logger.debug("Configuring vllm client")
        if self.vllm_mode == "server":
            if self.accelerator.is_main_process:
                self.vllm_client = VLLMClient(
                    host=args.vllm_server_host,
                    port=args.vllm_server_port,
                    connection_timeout=args.vllm_server_timeout,
                )
                self.vllm_client.init_communicator()

        elif self.vllm_mode == "colocate":
            # TODO: Implement colocation mode
            raise NotImplementedError(
                "Colocation mode is not supported for ArborGRPO at this time."
            )
        else:
            raise ValueError(
                f"vllm_mode must be either 'server' or 'colocate', got '{self.vllm_mode}'."
            )
        self.logger.debug("Done with configuring vllm client")

        self._last_loaded_step = (
            0  # tag to avoid useless loading during grad accumulation
        )

        # When using vLLM, the main process is responsible for loading the model weights. This can cause process
        # desynchronization and seems to lead to DeepSpeed hanging during initialization. To prevent this, we
        # synchronize all processes after vLLM has been fully initialized.
        self.accelerator.wait_for_everyone()

        # Reference model
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )
        if args.sync_ref_model:
            self.add_callback(
                SyncRefModelCallback(
                    ref_model=self.ref_model,  # type: ignore
                    accelerator=self.accelerator,
                )
            )

        self._next_batch_id: int = 0
        self._async_started: bool = False
        self.num_batches_ahead: int = args.num_batches_ahead

        self.async_requester = AsyncBatchRequester(
            # client_config=self.client_config,
            model_name=self.model.config._name_or_path,
            num_batches_ahead=self.num_batches_ahead,
            max_queue_size=args.async_max_queue_size,
            generation_timeout=args.async_generation_timeout,
        )
        self.logger.debug(
            "Async requester initialized",
            context={
                "num_batches_ahead": self.async_requester.num_batches_ahead,
                "max_queue_size": self.async_requester.max_queue_size,
            },
        )
        if self.accelerator.is_main_process and args.control_endpoint:
            _control_client = TrainerControlClient(self, args.control_endpoint)
            _control_client.start()
            self._control_client = _control_client
        self.logger.debug("Done with __init__")

    def get_train_dataloader(self):
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        batch_size = self._train_batch_size * self.gradient_accumulation_steps  # type: ignore

        dataloader_params = {
            "batch_size": batch_size,  # type: ignore (None case handled by config __post_init__)
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        dataloader = DataLoader(train_dataset, **dataloader_params)  # type: ignore

        # Always wrap with AsyncDataLoaderWrapper for consistent behavior
        # Store the wrapped dataloader for async access
        self._async_dataloader = AsyncDataLoaderWrapper(
            dataloader, buffer_size=max(5, self.num_batches_ahead * 2)
        )
        return self.accelerator.prepare(self._async_dataloader)

    def handle_checkpoint_request(
        self,
        checkpoint_name: Optional[str],
        metadata: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Checkpoint metadata must be a mapping if provided")
        with self._checkpoint_condition:
            self.logger.debug(
                "Received checkpoint request",
                context={
                    "checkpoint_name": checkpoint_name,
                    "has_metadata": metadata is not None,
                    "in_progress": self._checkpoint_request_data is not None,
                },
            )
            if self._checkpoint_request_data is not None:
                raise RuntimeError("A checkpoint request is already in progress")
            self._checkpoint_request_data = {
                "checkpoint_name": checkpoint_name,
                "metadata": metadata,
            }
            self._checkpoint_result = None
            self._checkpoint_condition.notify_all()
            self.logger.debug(
                "Waiting for checkpoint result",
                context={"checkpoint_name": checkpoint_name},
            )
            while self._checkpoint_result is None:
                self._checkpoint_condition.wait()
            result = dict(self._checkpoint_result)
            self._checkpoint_result = None
            self.logger.debug(
                "Checkpoint result ready",
                context={
                    "checkpoint_name": result.get("checkpoint_name"),
                    "checkpoint_dir": result.get("checkpoint_dir"),
                    "global_step": result.get("global_step"),
                },
            )
            return result

    def _service_checkpoint_requests(self) -> None:
        payload = [None]
        if self.accelerator.is_main_process:
            with self._checkpoint_condition:
                if self._checkpoint_request_data is not None:
                    payload[0] = dict(self._checkpoint_request_data)
                    self.logger.debug(
                        "Main process broadcasting checkpoint request",
                        context={
                            "checkpoint_name": payload[0].get("checkpoint_name"),
                            "has_metadata": payload[0].get("metadata") is not None,
                        },
                    )
        broadcast_object_list(payload, from_process=0)
        request = payload[0]
        if request is None:
            self.logger.debug("No checkpoint request to process")
            return

        self.logger.debug(
            "Processing checkpoint request",
            context={
                "checkpoint_name": request.get("checkpoint_name"),
                "has_metadata": request.get("metadata") is not None,
                "is_main_process": self.accelerator.is_main_process,
            },
        )

        record = self._execute_checkpoint(
            checkpoint_name=request.get("checkpoint_name"),
            metadata=request.get("metadata"),
        )

        if record is None:
            self.logger.debug("No checkpoint record to process")
            return

        if self.accelerator.is_main_process:
            with self._checkpoint_records_lock:
                self._checkpoint_records.append(dict(record))
                self._last_checkpoint_record = dict(record)
                self.logger.debug(
                    "Recorded checkpoint result",
                    context={
                        "checkpoint_name": record.get("checkpoint_name"),
                        "checkpoint_dir": record.get("checkpoint_dir"),
                        "global_step": record.get("global_step"),
                    },
                )
            with self._checkpoint_condition:
                self._checkpoint_result = dict(record)
                self._checkpoint_request_data = None
                self._checkpoint_condition.notify_all()
                self.logger.debug(
                    "Notified waiting threads about checkpoint completion",
                    context={"checkpoint_name": record.get("checkpoint_name")},
                )

    def _is_checkpoint_pending_main(self) -> bool:
        pending = False
        if self.accelerator.is_main_process:
            with self._checkpoint_condition:
                pending = self._checkpoint_request_data is not None
        return pending

    def _tick_checkpoints(self) -> None:
        """Barrier + service checkpoint requests + barrier, in lockstep across ranks."""
        self.accelerator.wait_for_everyone()
        self._service_checkpoint_requests()
        self.accelerator.wait_for_everyone()

    def _execute_checkpoint(
        self, checkpoint_name: Optional[str], metadata: Optional[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if metadata is not None and not isinstance(metadata, dict):
            raise TypeError("Checkpoint metadata must be a mapping if provided")
        self.logger.debug(
            "Entering checkpoint execution",
            context={
                "checkpoint_name": checkpoint_name,
                "is_main_process": self.accelerator.is_main_process,
            },
        )
        if self.accelerator.is_main_process:
            self.logger.info(
                "Checkpoint requested; pausing batch generation to save state."
            )
        self.accelerator.wait_for_everyone()
        self.logger.debug(
            "All processes synchronized before checkpoint save",
            context={"checkpoint_name": checkpoint_name},
        )

        # All ranks must participate in checkpoint saving with DeepSpeed ZeRO-3
        self.logger.debug(
            "Saving checkpoint state on all ranks",
            context={"global_step": int(self.state.global_step)},
        )
        super()._save_checkpoint(self.model, trial=None)
        self.accelerator.wait_for_everyone()

        # Finalize (rename/metadata) on main process only
        if self.accelerator.is_main_process:
            self.logger.debug(
                "Finalizing checkpoint on main process",
                context={
                    "checkpoint_name": checkpoint_name,
                    "global_step": int(self.state.global_step),
                },
            )
            record = self._finalize_checkpoint_record(checkpoint_name, metadata)
        else:
            record = None

        self.accelerator.wait_for_everyone()
        self.logger.debug(
            "All processes synchronized after checkpoint save",
            context={"checkpoint_name": checkpoint_name},
        )
        record_list = [record]
        broadcast_object_list(record_list, from_process=0)
        record = record_list[0]

        if record and record.get("checkpoint_dir"):
            self.state.best_model_checkpoint = record["checkpoint_dir"]
            self.logger.debug(
                "Updated best model checkpoint",
                context={
                    "checkpoint_name": record.get("checkpoint_name"),
                    "checkpoint_dir": record.get("checkpoint_dir"),
                },
            )

        return record

    def _finalize_checkpoint_record(
        self, checkpoint_name: Optional[str], metadata: Optional[dict[str, Any]]
    ) -> dict[str, Any]:
        """Finalize a checkpoint after state has been saved by all ranks.

        This performs directory renaming to the requested checkpoint name and writes
        metadata. Only safe to call on the main process.
        """
        default_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=None)
        default_dir = os.path.abspath(os.path.join(run_dir, default_folder))
        requested_name = checkpoint_name or default_folder
        target_dir = os.path.abspath(os.path.join(run_dir, requested_name))

        if requested_name != default_folder:
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            if os.path.exists(default_dir):
                shutil.move(default_dir, target_dir)
            else:
                target_dir = os.path.abspath(os.path.join(run_dir, requested_name))
        else:
            target_dir = default_dir

        if self.state.best_model_checkpoint:
            best_path = os.path.abspath(self.state.best_model_checkpoint)
            if best_path == os.path.abspath(default_dir):
                self.state.best_model_checkpoint = target_dir

        if metadata:
            metadata_path = os.path.join(target_dir, "metadata.json")
            with open(metadata_path, "w", encoding="utf-8") as fh:
                json.dump(metadata, fh, indent=2, sort_keys=True)

        record = {
            "checkpoint_name": requested_name,
            "checkpoint_dir": target_dir,
            "global_step": int(self.state.global_step),
            "requested": True,
            "metadata": metadata,
            "timestamp": time.time(),
        }
        return record

    def get_checkpoint_records(self) -> list[dict[str, Any]]:
        with self._checkpoint_records_lock:
            return [dict(record) for record in self._checkpoint_records]

    def get_last_checkpoint_record(self) -> Optional[dict[str, Any]]:
        with self._checkpoint_records_lock:
            if self._last_checkpoint_record is None:
                return None
            return dict(self._last_checkpoint_record)

    def _enable_gradient_checkpointing(
        self, model: PreTrainedModel, args: ArborGRPOConfig
    ) -> PreTrainedModel:
        """Enables gradient checkpointing for the model."""
        # Ensure use_cache is disabled
        model.config.use_cache = False

        # Enable gradient checkpointing on the base model for PEFT
        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()  # type: ignore
        # Enable gradient checkpointing for non-PEFT models
        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        assert isinstance(gradient_checkpointing_kwargs, dict)
        use_reentrant = gradient_checkpointing_kwargs.get("use_reentrant", True)

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, logits_to_keep, batch_size=None
    ) -> torch.Tensor:
        batch_size = batch_size or input_ids.size(
            0
        )  # Chunk inputs into smaller batches to reduce memory peak
        all_logps = []
        for i in range(0, input_ids.size(0), batch_size):
            input_ids_batch = input_ids[i : i + batch_size]
            attention_mask_batch = attention_mask[i : i + batch_size]
            logits = model(
                input_ids=input_ids_batch,
                attention_mask=attention_mask_batch,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
            input_ids_batch = input_ids_batch[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature
            logps = selective_log_softmax(
                logits, input_ids_batch
            )  # compute logprobs for the input tokens
            all_logps.append(logps)
        return torch.cat(all_logps, dim=0)

    def _move_model_to_vllm(self):
        assert self.model is not None
        is_generating = False
        if self.accelerator.is_main_process:
            pending_count = self.async_requester.get_pending_count()
            # If a checkpoint is being addressed and the pending queue equals the async pipeline size,
            # treat as not actively generating so we don't block checkpoint handling.
            if self._is_checkpoint_pending_main() and (
                pending_count == self.async_requester.num_batches_ahead
            ):
                is_generating = False
            else:
                is_generating = pending_count > 0
        is_generating_list = [is_generating]
        broadcast_object_list(is_generating_list, from_process=0)
        is_generating = is_generating_list[0]

        waits = 0
        while is_generating:
            time.sleep(0.5)
            waits += 1
            if waits % 10 == 0:
                self.logger.info("Waiting for generation to finish before syncing.")
            if self.accelerator.is_main_process:
                pending_count = self.async_requester.get_pending_count()
                if self._is_checkpoint_pending_main() and (
                    pending_count == self.async_requester.num_batches_ahead
                ):
                    is_generating = False
                else:
                    is_generating = pending_count > 0
            is_generating_list = [is_generating]
            broadcast_object_list(is_generating_list, from_process=0)
            is_generating = is_generating_list[0]

        if self.state.global_step > 0:  # skip first step
            deepspeed_plugin = self.accelerator.state.deepspeed_plugin
            zero_stage_3 = (
                deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
            )
            if zero_stage_3:
                gather_if_zero3 = deepspeed.zero.GatheredParameters
            else:
                gather_if_zero3 = nullcontext
            self.accelerator.wait_for_everyone()
            self.logger.info("Starting weight sync to vLLM")

            if is_peft_model(self.model):
                # PEFT: gather + merge, then update each parameter
                with gather_if_zero3(list(self.model.parameters())):
                    self.model.merge_adapter()  # type: ignore :(
                    for name, param in self.model.named_parameters():
                        # recover original parameter names
                        name = name.removeprefix("base_model.model.").replace(
                            ".base_layer", ""
                        )
                        if self.model.prefix in name:  # type: ignore :(
                            continue  # discard some parameters
                        if "original_module" in name:  # from modules_to_save
                            continue
                        name = name.replace("modules_to_save.default.", "")
                        if self.vllm_client:
                            self.vllm_client.update_named_param(name, param.data)
                    self.model.unmerge_adapter()  # type: ignore :(
            else:
                # non-PEFT models: gather + update each parameter individually
                for name, param in self.model.named_parameters():  # type: ignore :(
                    with gather_if_zero3([param]):
                        if self.vllm_client:
                            self.vllm_client.update_named_param(name, param.data)

            # reset cache + wait for background tasks to complete
            if self.vllm_client:
                self.vllm_client.reset_prefix_cache()
                while self.vllm_client.get_num_background_tasks() > 0:
                    time.sleep(0.5)
                    self.logger.info("Resetting prefix cache.")

        self.accelerator.wait_for_everyone()

    def _inner_training_loop(self, *args, **kwargs):
        """Override to ensure async generator is stopped when training ends"""
        try:
            return super()._inner_training_loop(*args, **kwargs)
        finally:
            if self.accelerator.is_main_process and self._control_client is not None:
                self._control_client.stop()
                self._control_client = None
            # Clean up async generator on all processes
            if (
                self.async_requester
                and self._async_started
                and self.accelerator.is_main_process
            ):
                self.async_requester.stop()
            self._async_started = False

    def _prepare_inputs(  # type: ignore
        self, inputs: list[dict[str, Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        inputs: raw inputs from the dataloader (per process)

        This method implements async batch generation with priming:
        1. Always maintain num_batches_ahead batches in the pipeline
        2. On first calls, prime by submitting num_batches_ahead batches before retrieving any
        3. On subsequent calls, submit new batches to maintain the pipeline
        """
        # Ensure all processes are synchronized at the start
        self.accelerator.wait_for_everyone()
        # Service any pending checkpoint requests in lockstep at entry
        self._tick_checkpoints()
        # inputs = list of dicts for all gradient accumulation steps
        generate_every = (
            self.gradient_accumulation_steps
        )  # * self.num_iterations # num iterations unused
        self.logger.debug(
            "prepare_inputs entry",
            context={
                "step": self._step,
                "global_step": int(self.state.global_step),
                "buffer_empty": self._buffered_inputs is None,
                "generate_every": generate_every,
                "step_mod": self._step % generate_every if generate_every else 0,
                "next_batch_id": self._next_batch_id,
            },
        )

        # Check if we need to generate new completions
        if self._step % generate_every == 0 or self._buffered_inputs is None:
            # Update weights to vLLM if needed
            if self.state.global_step > self._last_loaded_step:
                self.logger.debug(
                    f"Syncing weights to vLLM at step {self.state.global_step}"
                )
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Start async requester on main process only, then sync
            if self.accelerator.is_main_process and not self._async_started:
                self.async_requester.start()
                self._async_started = True
            self.accelerator.wait_for_everyone()

            # Calculate which batch we need for this step
            batch_id_to_retrieve = self._step // generate_every

            # Calculate the target: we want to always be num_batches_ahead batches ahead
            # This means we should have submitted up to batch_id_to_retrieve + num_batches_ahead
            target_batch_id = (
                batch_id_to_retrieve + self.async_requester.num_batches_ahead
            )

            # Submit any missing batches to maintain the pipeline
            # On first call, this submits batches 0 through num_batches_ahead
            # On subsequent calls, this submits new batches to stay ahead
            batches_requested = 0
            steps_per_batch = generate_every
            max_batch_id = (
                self.state.max_steps * self.gradient_accumulation_steps - 1
            ) // steps_per_batch
            target_batch_id = min(target_batch_id, max_batch_id)

            for batch_id in range(self._next_batch_id, target_batch_id + 1):
                first_grad_step_for_batch = batch_id * steps_per_batch
                if (
                    first_grad_step_for_batch
                    >= self.state.max_steps * self.gradient_accumulation_steps
                ):
                    self.logger.info(
                        f"Reached max global steps ({self.state.max_steps}), stopping batch generation"
                    )
                    break

                if self.accelerator.is_main_process:
                    if self.async_requester.request_batch(
                        BatchRequest(
                            batch_id=batch_id,
                            soft_completion_penalty_length=self.soft_completion_penalty_length,
                            # TODO: Masking and truncation should be handled here.
                        )
                    ):
                        batches_requested += 1
                        self.logger.debug(
                            "Requested async batch",
                            context={
                                "batch_id": batch_id,
                                "next_batch_id": self._next_batch_id,
                            },
                        )
                    else:
                        self.logger.debug(
                            "Async requester queue is full; batch %s not requested",
                            batch_id,
                        )
                self.accelerator.wait_for_everyone()

            # Update next batch id
            if self.accelerator.is_main_process:
                self._next_batch_id = self._next_batch_id + batches_requested
                self.logger.debug(
                    "Updated next_batch_id",
                    context={
                        "next_batch_id": self._next_batch_id,
                        "requested": batches_requested,
                        "step": self._step,
                    },
                )
            self.accelerator.wait_for_everyone()
            # Synchronize next_batch_id across all processes
            next_batch_id_list = [
                self._next_batch_id if self.accelerator.is_main_process else 0
            ]
            broadcast_object_list(next_batch_id_list, from_process=0)
            self._next_batch_id = next_batch_id_list[0]
            self.accelerator.wait_for_everyone()

            # Now retrieve the batch we need for this step
            broadcast_data = None
            while broadcast_data is None:
                # Synchronize and service checkpoint requests in lockstep across all ranks
                self.accelerator.wait_for_everyone()
                self._service_checkpoint_requests()
                self.accelerator.wait_for_everyone()

                if self.accelerator.is_main_process:
                    try:
                        batch_result = self.async_requester.get_batch(
                            batch_id_to_retrieve, timeout=1.0
                        )
                        if batch_result.batch_id != batch_id_to_retrieve:
                            raise ValueError(
                                f"Retrieved batch {batch_result.batch_id} but expected {batch_id_to_retrieve}"
                            )
                        processed_results = batch_result.processed_results

                        broadcast_data = {
                            "prompt_ids": processed_results.prompt_ids,
                            "prompt_mask": processed_results.prompt_mask,
                            "completion_ids": processed_results.completion_ids,
                            "completion_mask": processed_results.completion_mask,
                            "rewards": processed_results.rewards,
                            "all_reward_dict": batch_result.all_reward_dict,
                            "completions": batch_result.completions,
                            "prompts": batch_result.prompts,
                            "metrics": batch_result.metrics,
                        }
                    except TimeoutError:
                        broadcast_data = None
                # Broadcast processed data (or None) to all ranks
                broadcast_list = [broadcast_data]
                broadcast_object_list(broadcast_list, from_process=0)
                broadcast_data = broadcast_list[0]
                self.accelerator.wait_for_everyone()

                if broadcast_data is None:
                    if self.accelerator.is_main_process:
                        self.logger.debug(
                            "No broadcast data for batch %s yet; retrying.",
                            batch_id_to_retrieve,
                        )
                    continue
                break
            # Each process takes its slice based on the total broadcast batch size
            total_sequences = len(broadcast_data["prompt_ids"])
            world_size = max(1, self.accelerator.num_processes)
            per_process = (
                math.ceil(total_sequences / world_size) if total_sequences else 0
            )
            slice_start = per_process * self.accelerator.process_index
            slice_stop = min(slice_start + per_process, total_sequences)
            process_slice = slice(slice_start, slice_stop)

            # Create rewards tensor and compute advantages using full batch
            assert (
                broadcast_data is not None
            )  # After broadcast, all processes have data
            all_rewards = torch.tensor(
                broadcast_data["rewards"], device=self.accelerator.device
            )
            all_advantages = self._compute_advantages(all_rewards)

            # Now create tensors only for this process's slice
            input_ids_list = []
            attention_mask_list = []

            slice_size = max(0, process_slice.stop - process_slice.start)
            self.logger.debug(
                "prepare_inputs process=%s start=%s stop=%s per_process=%s total_sequences=%s len_inputs=%s len_prompt_ids=%s len_completion_ids=%s len_prompt_mask=%s len_completion_mask=%s",
                self.accelerator.process_index,
                process_slice.start,
                process_slice.stop,
                per_process,
                total_sequences,
                len(inputs),
                len(broadcast_data["prompt_ids"]),
                len(broadcast_data["completion_ids"]),
                len(broadcast_data["prompt_mask"]),
                len(broadcast_data["completion_mask"]),
            )

            for i in range(process_slice.start, process_slice.stop):
                input_ids_list.append(
                    torch.tensor(
                        broadcast_data["prompt_ids"][i]
                        + broadcast_data["completion_ids"][i],
                        device=self.accelerator.device,
                    )
                )
                attention_mask_list.append(
                    torch.tensor(
                        broadcast_data["prompt_mask"][i]
                        + broadcast_data["completion_mask"][i],
                        device=self.accelerator.device,
                    )
                )

            if not input_ids_list:
                self.logger.debug(
                    "prepare_inputs empty input_ids_list; process=%s slice_size=%s per_process=%s total_sequences=%s",
                    self.accelerator.process_index,
                    slice_size,
                    per_process,
                    total_sequences,
                )
            else:
                self.logger.debug(
                    "prepare_inputs padding %s sequences; input_lengths=%s mask_lengths=%s",
                    len(input_ids_list),
                    [tuple(t.shape) for t in input_ids_list],
                    [tuple(t.shape) for t in attention_mask_list],
                )

            input_ids = pad(
                input_ids_list,
                padding_value=self.processing_class.pad_token_id,  # type: ignore
                padding_side="right",
            )  # type: ignore
            attention_mask = pad(attention_mask_list, padding_side="right")  # type: ignore

            # Truncate if needed
            if self.max_seq_len is not None and input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len :]
                attention_mask = attention_mask[:, -self.max_seq_len :]

            # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
            if self.mask_truncated_completions:
                eos_and_pad = [self.eos_token_id, self.pad_token_id]
                is_truncated = [
                    ids[-1] not in eos_and_pad
                    for ids in broadcast_data["completion_ids"]
                ]
                # completion_mask is a ragged list of lists (not a tensor). Apply per-sample zeroing.
                broadcast_data["completion_mask"] = [
                    ([0] * len(cm) if truncated else cm)
                    for cm, truncated in zip(
                        broadcast_data["completion_mask"], is_truncated
                    )
                ]
            # Take this process's slice of advantages
            advantages = all_advantages[process_slice]

            # Log metrics on main process only
            if self.accelerator.is_main_process:
                self._log_reward_metrics_primary(
                    mode="train",
                    all_reward_dict=broadcast_data["all_reward_dict"],
                    all_rewards=all_rewards,
                    generation_batch_size=len(all_rewards),
                )

                self._log_textual_data_primary(
                    all_prompts=broadcast_data["prompts"],
                    all_completions=broadcast_data["completions"],
                    all_reward_dict=broadcast_data["all_reward_dict"],
                )

                # Log completion metrics using full batch data on CPU to save memory
                self._log_completion_metrics_primary(
                    mode="train",
                    all_completion_mask=broadcast_data["completion_mask"],
                    all_completion_ids=broadcast_data["completion_ids"],
                    all_prompt_mask=broadcast_data["prompt_mask"],
                )

                self._log_external_metrics_primary(
                    mode="train",
                    external_metrics=broadcast_data["metrics"],
                )
            with torch.no_grad():
                completion_mask = attention_mask[:, 1:]
                logits_to_keep = completion_mask.size(1)
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    input_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size=self.per_device_train_batch_size,
                )

            # Compute global total completion tokens from broadcasted masks (same on all ranks)
            total_completion_tokens_global = sum(
                int(sum(cm)) for cm in broadcast_data["completion_mask"]
            )
            total_completion_tokens_tensor = torch.tensor(
                float(total_completion_tokens_global), device=self.accelerator.device
            )

            # Concatenate all data for shuffling
            full_batch = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "old_per_token_logps": old_per_token_logps,
                "advantages": advantages,
            }

            # Shuffle and split for gradient accumulation
            full_batch = shuffle_tensor_dict(full_batch)
            self._buffered_inputs = split_tensor_dict(
                full_batch, self.gradient_accumulation_steps
            )
            # Attach scalar metadata to each split so compute_loss can access it
            for i in range(len(self._buffered_inputs)):
                self._buffered_inputs[i]["total_completion_tokens"] = (
                    total_completion_tokens_tensor
                )
            self.accelerator.wait_for_everyone()
        # Return appropriate slice from buffer
        result = self._buffered_inputs[self._step % self.gradient_accumulation_steps]
        self._step += 1
        self.logger.debug(
            "prepare_inputs exit",
            context={
                "next_step": self._step,
                "global_step": int(self.state.global_step),
                "buffer_remaining": len(self._buffered_inputs),
            },
        )
        self.accelerator.wait_for_everyone()
        return result

    def _compute_advantages(
        self,
        rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advantages from rewards with normalization using full batch statistics."""
        # Always use full batch statistics
        mean_grouped = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute advantages
        mean_grouped = mean_grouped.repeat_interleave(self.num_generations, dim=0)
        std_grouped = std_grouped.repeat_interleave(self.num_generations, dim=0)
        advantages = rewards - mean_grouped

        if self.scale_rewards:
            advantages = advantages / (std_grouped + 1e-4)

        return advantages

    def compute_loss(  # type: ignore
        self,  # type: ignore
        model: PreTrainedModel,
        inputs: dict[str, torch.Tensor],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor:  # type: ignore
        mode = "train"
        # Compute the per-token log probabilities for the model
        input_ids, attention_mask = inputs["input_ids"], inputs["attention_mask"]

        # prompt is at least 1 token
        completion_mask = attention_mask[:, 1:]
        logits_to_keep = completion_mask.size(1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep
        )
        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps,
        # so we can skip it's computation (see _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = (
            per_token_logps.detach()
            if inputs["old_per_token_logps"] is None
            else inputs["old_per_token_logps"]
        )
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)

        if self.delta is not None:
            # Use clamp instead of min to handle tensor-float comparison
            per_token_loss1 = torch.clamp(
                coef_1, max=self.delta
            ) * advantages.unsqueeze(1)
        else:
            # Original GRPO clipping (only lower bound implicitly applied by the final min)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)

        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            with torch.no_grad():
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model, input_ids, attention_mask, logits_to_keep
                    )
                else:
                    with self.accelerator.unwrap_model(self.model).disable_adapter():  # type: ignore
                        ref_per_token_logps = self._get_per_token_logps(
                            self.model, input_ids, attention_mask, logits_to_keep
                        )
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )
            per_token_loss = per_token_loss + self.beta * per_token_kl
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).nanmean().item()  # type: ignore
            )
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
                per_token_loss.size(0) * self.max_seq_len
            )  # type: ignore
        elif self.loss_type == "dapo":
            normalizer = (
                inputs["total_completion_tokens"] / self.accelerator.num_processes
            )
            loss = (per_token_loss * completion_mask).sum() / normalizer
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Compute the clipped probability ratios
        is_low_clipped = (coef_1 < 1 - self.epsilon_low) & (advantages.unsqueeze(1) < 0)
        is_high_clipped = (coef_1 > 1 + self.epsilon_high) & (
            advantages.unsqueeze(1) > 0
        )
        is_region_clipped = is_low_clipped | is_high_clipped

        low_clip = (is_low_clipped * completion_mask).sum() / completion_mask.sum()
        high_clip = (is_high_clipped * completion_mask).sum() / completion_mask.sum()
        clip_ratio = (is_region_clipped * completion_mask).sum() / completion_mask.sum()

        gathered_low_clip = self.accelerator.gather_for_metrics(low_clip)
        self._metrics[mode]["clip_ratio/low_mean"].append(
            gathered_low_clip.nanmean().item()  # type: ignore
        )
        self._metrics[mode]["clip_ratio/low_min"].append(
            nanmin(gathered_low_clip).item()  # type: ignore
        )
        gathered_high_clip = self.accelerator.gather_for_metrics(high_clip)
        self._metrics[mode]["clip_ratio/high_mean"].append(
            gathered_high_clip.nanmean().item()  # type: ignore
        )
        self._metrics[mode]["clip_ratio/high_max"].append(
            nanmax(gathered_high_clip).item()  # type: ignore
        )
        gathered_clip_ratio = self.accelerator.gather_for_metrics(clip_ratio)
        self._metrics[mode]["clip_ratio/region_mean"].append(
            gathered_clip_ratio.nanmean().item()  # type: ignore
        )
        return loss

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        mode = "train" if self.model is not None and self.model.training else "eval"  # type: ignore
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics[mode].items()
        }  # average the metrics

        # This method can be called both in training and evaluation. When called in evaluation, the keys in `logs`
        # start with "eval_". We need to add the prefix "eval_" to the keys in `metrics` to match the format.
        if mode == "eval":
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if start_time is not None:
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics[mode].clear()

        if self.accelerator.is_main_process and self.log_completions:
            if (
                self.args.report_to
                and "wandb" in self.args.report_to
                and wandb.run is not None
            ):
                import pandas as pd

                table = {
                    "step": [str(self.state.global_step)]
                    * len(self._textual_logs["prompt"]),
                    "prompt": list(self._textual_logs["prompt"]),
                    "completion": list(self._textual_logs["completion"]),
                    **{k: list(v) for k, v in self._textual_logs["rewards"].items()},
                }
                if len(table["prompt"]) > 0:
                    df = pd.DataFrame(table)
                    if self.wandb_log_unique_prompts:
                        df = df.drop_duplicates(subset=["prompt"])
                    wandb.log({"completions": wandb.Table(dataframe=df)})

            # Clear the textual logs after logging
            self._textual_logs["prompt"].clear()
            self._textual_logs["completion"].clear()
            for key in self._textual_logs["rewards"]:
                self._textual_logs["rewards"][key].clear()

    def _log_reward_metrics_primary(
        self,
        mode: str,
        all_reward_dict: dict[str, Any],
        all_rewards: torch.Tensor,
        generation_batch_size: int,
    ) -> None:
        """
        Log generation metrics (PRIMARY PROCESS ONLY).
        This handles reward statistics and per-reward-function metrics using the full batch data.
        """
        # Log reward statistics using full batch
        mean_rewards = all_rewards.view(-1, self.num_generations).mean(dim=1)
        std_rewards = all_rewards.view(-1, self.num_generations).std(dim=1)
        self._metrics[mode]["reward"].append(mean_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_rewards.mean().item())

        # Log individual reward function scores as metrics
        for reward_key in all_reward_dict:
            if reward_key != "reward":  # Skip the consolidated reward
                reward_values = all_reward_dict[reward_key]
                if isinstance(reward_values, list):
                    reward_tensor = torch.tensor(
                        reward_values, device=all_rewards.device
                    )
                else:
                    reward_tensor = reward_values
                mean_reward = reward_tensor.mean().item()
                self._metrics[mode][f"rewards/{reward_key}"].append(mean_reward)

    def _log_textual_data_primary(
        self,
        all_prompts: list[str | list[dict[str, Any]]],
        all_completions: list[str | list[dict[str, Any]]],
        all_reward_dict: dict[str, Any],
    ) -> None:
        """
        Log textual data for wandb (PRIMARY PROCESS ONLY).
        This logs the full batch of prompts, completions, and rewards.
        """
        self._textual_logs["prompt"].extend(all_prompts)
        self._textual_logs["completion"].extend(all_completions)

        # Log all reward scores - both individual functions and consolidated
        for reward_key in all_reward_dict:
            reward_values = all_reward_dict[reward_key]
            self._textual_logs["rewards"][reward_key].extend(
                reward_values.tolist()
                if isinstance(reward_values, torch.Tensor)
                else reward_values
            )

    def _log_completion_metrics_primary(
        self,
        mode: str,
        all_completion_mask: list[list[int]],
        all_completion_ids: list[list[int]],
        all_prompt_mask: list[list[int]],
    ) -> None:
        """
        Log completion-related metrics (PRIMARY PROCESS ONLY).
        This handles completion length statistics using the full batch data.
        """
        # Log token count
        if mode == "train":
            total_tokens = sum(
                len(pm) + len(cm)
                for pm, cm in zip(all_prompt_mask, all_completion_mask)
            )
            self.state.num_input_tokens_seen += total_tokens
        self._metrics[mode]["num_tokens"] = [self.state.num_input_tokens_seen]

        # Log completion lengths
        completion_lengths = [sum(mask) for mask in all_completion_mask]
        self._metrics[mode]["completions/mean_length"].append(
            float(sum(completion_lengths)) / len(completion_lengths)
        )
        self._metrics[mode]["completions/min_length"].append(
            float(min(completion_lengths))
        )
        self._metrics[mode]["completions/max_length"].append(
            float(max(completion_lengths))
        )

        # Check for EOS tokens
        term_lengths = []
        for comp_ids, comp_mask in zip(all_completion_ids, all_completion_mask):
            has_eos = any(
                token == self.processing_class.eos_token_id  # type: ignore
                for token, mask in zip(comp_ids, comp_mask)
                if mask
            )
            if has_eos:
                term_lengths.append(sum(comp_mask))

        clipped_completions_ratio = 1 - len(term_lengths) / len(completion_lengths)
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )

        if len(term_lengths) == 0:
            term_lengths = [0]
        self._metrics[mode]["completions/mean_terminated_length"].append(
            float(sum(term_lengths)) / len(term_lengths)
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            float(min(term_lengths))
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            float(max(term_lengths))
        )

    def _log_external_metrics_primary(
        self,
        mode: str,
        external_metrics: dict[str, Any],
    ) -> None:
        """
        Log external metrics (PRIMARY PROCESS ONLY).
        This handles external metrics using the full batch data.
        """
        for key, value in external_metrics.items():
            self._metrics[mode][key].append(value)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer_config_json", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--command_port", type=int, required=True)
    parser.add_argument("--vllm_server_port", type=int, required=True)
    return parser.parse_args()


def build_trainer_config(args: argparse.Namespace) -> ArborGRPOConfig:
    cfg = json.loads(args.trainer_config_json)
    lora_cfg = cfg.get("lora_config")
    if lora_cfg is not None:
        cfg["lora_config"] = LoraConfig(**lora_cfg)
    return ArborGRPOConfig(**cfg)


def main():
    setup_logging(
        log_level="INFO",
        enable_console_logging=True,
        enable_file_logging=False,
        show_colors=False,
    )
    # Reduce asyncio log verbosity
    logging.getLogger("asyncio").setLevel(logging.ERROR)
    args = parse_args()
    trainer_config = build_trainer_config(args)
    trainer = ArborGRPOTrainer(
        model=args.model,
        args=trainer_config,
    )

    stop_flag = threading.Event()
    try:
        trainer.train()
    finally:
        stop_flag.set()


if __name__ == "__main__":
    main()
