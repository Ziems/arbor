from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceLaunchConfig:
    max_context_length: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
    is_grpo: Optional[bool] = False
    grpo_job_id: Optional[str] = None
    # GPU memory sharing configuration
    gpu_memory_utilization: Optional[float] = (
        0.9  # Default 90%, set to 0.45 for sharing
    )
    enable_gpu_sharing: Optional[bool] = False  # Enable sharing mode
