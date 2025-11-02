from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceLaunchConfig:
    max_context_length: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
    num_gpus: Optional[int] = 1
    grpo_job_id: Optional[str] = None
    log_file_path: Optional[str] = None
    hf_token: Optional[str] = None
