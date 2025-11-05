from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceLaunchConfig:
    max_seq_len: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
    num_gpus: Optional[int] = 1
    grpo_job_id: Optional[str] = None
    log_file_path: Optional[str] = None
