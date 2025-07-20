from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceLaunchConfig:
    max_context_length: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
    is_grpo: Optional[bool] = False
    grpo_job_id: Optional[str] = None
