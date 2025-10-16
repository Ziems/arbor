from enum import Enum
from typing import Any, Generic, List, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict

# Generic type for list items
T = TypeVar("T")


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaginatedResponse(StrictBaseModel, Generic[T]):
    object: str = "list"
    data: List[T]
    has_more: bool = False


class FileModel(StrictBaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str
    format: str = "unknown"  # Detected format: "sft", "dpo", or "unknown"


class WandbConfig(StrictBaseModel):
    project: str
    name: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List[str]] = None


class IntegrationModel(StrictBaseModel):
    type: str
    wandb: WandbConfig


class FineTuneRequest(StrictBaseModel):
    model: str
    training_file: str  # id of uploaded jsonl file
    method: Optional[dict] = None
    suffix: Optional[str] = None
    num_gpus: Optional[int] = 1  # Number of GPUs to request for training
    # UNUSED
    validation_file: Optional[str] = None
    integrations: Optional[List[IntegrationModel]] = []
    seed: Optional[int] = None


class ErrorModel(StrictBaseModel):
    code: str
    message: str
    param: str | None = None


class SupervisedHyperparametersModel(StrictBaseModel):
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    n_epochs: int | str = "auto"


class DPOHyperparametersModel(StrictBaseModel):
    beta: float | str = "auto"
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    n_epochs: int | str = "auto"


class SupervisedModel(StrictBaseModel):
    hyperparameters: SupervisedHyperparametersModel


class DpoModel(StrictBaseModel):
    hyperparameters: DPOHyperparametersModel


class MethodModel(StrictBaseModel):
    type: Literal["supervised"] | Literal["dpo"]
    supervised: SupervisedModel | None = None
    dpo: DpoModel | None = None


# https://platform.openai.com/docs/api-reference/fine-tuning/object
class JobStatus(Enum):
    PENDING = "pending"  # Not in OAI
    PENDING_PAUSE = "pending_pause"  # Not in OAI
    PENDING_RESUME = "pending_resume"  # Not in OAI
    PAUSED = "paused"  # Not in OAI
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PENDING_CANCEL = "pending_cancel"
    CREATED = "created"


# https://platform.openai.com/docs/api-reference/fine-tuning/object
class JobStatusModel(StrictBaseModel):
    object: str = "fine_tuning.job"
    id: str
    fine_tuned_model: str | None = None
    status: JobStatus
    training_file: str | None = None
    model: str | None = None

    # UNUSED so commented out
    # model: str
    # created_at: int
    # error: ErrorModel | None = None
    # details: str = ""
    # finished_at: int
    # hyperparameters: None # deprecated in OAI
    # organization_id: str
    # result_files: list[str]
    # trained_tokens: int | None = None # None if not finished
    # training_file: str
    # validation_file: str
    # integrations: list[Integration]
    # seed: int
    # estimated_finish: int | None = None # The Unix timestamp (in seconds) for when the fine-tuning job is estimated to finish. The value will be null if the fine-tuning job is not running.
    # method: MethodModel
    # metadata: dict[str, str]


class JobEventModel(StrictBaseModel):
    object: str = "fine_tuning.job_event"
    id: str
    created_at: int
    level: str
    message: str
    data: dict[str, Any]
    type: str


class MetricsModel(StrictBaseModel):
    step: int
    train_loss: float
    train_mean_token_accuracy: float
    valid_loss: float
    valid_mean_token_accuracy: float
    full_valid_loss: float
    full_valid_mean_token_accuracy: float


class JobCheckpointModel(StrictBaseModel):
    object: str = "fine_tuning.job_checkpoint"
    id: str
    created_at: int
    fine_tuned_model_checkpoint: str
    step_number: int
    metrics: MetricsModel
    fine_tuning_job_id: str


class ChatCompletionMessage(StrictBaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class ChatCompletionRequest(StrictBaseModel):
    model: str
    messages: List[ChatCompletionMessage]
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None


class ChatCompletionChoice(StrictBaseModel):
    message: ChatCompletionMessage
    index: int
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionModel(StrictBaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]


class InferenceJobRequest(StrictBaseModel):
    model: str
    max_context_length: Optional[int] = None


### GRPO
class MultiGPUConfig(StrictBaseModel):
    num_inference_gpus: int
    num_training_gpus: int  # Number of GPUs to use for training


class GRPOGPUConfig(StrictBaseModel):
    type: Literal["multi"]
    multi: MultiGPUConfig


class GRPOStatus(StrictBaseModel):
    job_id: str
    status: Optional[str] = None
    current_model: str
    checkpoints: dict[str, str]
    last_checkpoint: Optional[str] = None
    pending_batch_ids: List[int] = []


class LoRAConfigRequest(StrictBaseModel):
    r: int
    lora_alpha: int
    target_modules: List[str]
    lora_dropout: float


class WandbConfigRequest(StrictBaseModel):
    project: str
    name: Optional[str] = None


class ArborGRPOConfigRequest(StrictBaseModel):
    """Only the settings we want to feed straight into ArborGRPOConfig."""

    temperature: Optional[float] = None
    beta: Optional[float] = None
    num_generations: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    gradient_accumulation_steps: Optional[int] = None
    gradient_checkpointing: Optional[bool] = None
    lr_scheduler_type: Optional[str] = None
    max_prompt_length: Optional[int] = None
    max_completion_length: Optional[int] = None
    gradient_checkpointing_kwargs: Optional[dict[str, Any]] = None
    bf16: Optional[bool] = None
    scale_rewards: Optional[bool] = None
    max_grad_norm: Optional[float] = None
    report_to: Optional[str] = None
    log_completions: Optional[bool] = None
    logging_steps: Optional[int] = None
    generation_batch_size: Optional[int] = None
    mask_truncated_completions: Optional[bool] = None
    async_generation_timeout: Optional[float] = None
    max_seq_len: Optional[int] = None
    lora_config: Optional[LoRAConfigRequest] = None
    max_steps: Optional[int] = None
    warmup_steps: Optional[int] = None
    # ...and any other ArborGRPOConfig fields you care about.
    # Thanks to extra="allow" you can keep this list short and still pass others.


class GRPOInitializeRequest(StrictBaseModel):
    run_name: Optional[str] = None
    model: str
    trainer_config: ArborGRPOConfigRequest
    inference_config: InferenceJobRequest
    wandb_config: Optional[WandbConfigRequest] = None
    gpu_config: GRPOGPUConfig = GRPOGPUConfig(
        type="multi", multi=MultiGPUConfig(num_inference_gpus=1, num_training_gpus=1)
    )


# Base class for all GRPO requests except initialize
class GRPOBaseRequest(StrictBaseModel):
    job_id: str


class GRPOStepRequest(GRPOBaseRequest):
    model: str
    batch_id: int
    batch: List[dict] | List[List[dict]]


class GRPOCheckpointRequest(GRPOBaseRequest):
    checkpoint_name: str


class GRPOTerminateRequest(GRPOBaseRequest):
    pass