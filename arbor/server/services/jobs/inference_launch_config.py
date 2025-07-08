from dataclasses import dataclass
from typing import Optional


@dataclass
class InferenceLaunchConfig:
    max_context_length: Optional[int] = None
    gpu_ids: Optional[list[int]] = None
