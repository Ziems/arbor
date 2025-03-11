from typing import Optional, List, Literal, Generic, TypeVar, Dict, Any
from enum import Enum
from pydantic import BaseModel, ConfigDict

class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

class WandbConfig(BaseModel):
    project: str
    name: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List[str]] = None

class Integration(BaseModel):
    type: str
    wandb: WandbConfig

class FineTuneRequest(BaseModel):
    model: str
    training_file: str  # id of uploaded jsonl file
    suffix: Optional[str] = None
    # UNUSED
    validation_file: Optional[str] = None
    integrations: Optional[List[Integration]] = []
    seed: Optional[int] = None

class ErrorModel(BaseModel):
    code: str
    message: str
    param: str | None = None

class SupervisedHyperparametersModel(BaseModel):
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    n_epochs: int | str = "auto"

class DPOHyperparametersModel(BaseModel):
    beta: float | str = "auto"
    batch_size: int | str = "auto"
    learning_rate_multiplier: float | str = "auto"
    n_epochs: int | str = "auto"

class SupervisedModel(BaseModel):
    hyperparameters: SupervisedHyperparametersModel

class DpoModel(BaseModel):
    hyperparameters: DPOHyperparametersModel

class MethodModel(BaseModel):
    type: Literal["supervised"] | Literal["dpo"]
    supervised: SupervisedModel | None = None
    dpo: DpoModel | None = None

# https://platform.openai.com/docs/api-reference/fine-tuning/object
class JobStatus(Enum):
    PENDING = "pending" # Not in OAI
    VALIDATING_FILES = "validating_files"
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

# https://platform.openai.com/docs/api-reference/fine-tuning/object
class JobStatusResponse(BaseModel):
    object: str = "fine_tuning.job"
    id: str
    fine_tuned_model: str | None = None
    status: JobStatus

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

# Generic type for list items
T = TypeVar('T')

class PaginatedResponse(BaseModel, Generic[T]):
    object: str = "list"
    data: List[T]
    has_more: bool = False

    model_config = ConfigDict(exclude_none=True)