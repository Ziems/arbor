import threading
import time
from collections import deque
from typing import Any

from pydantic import BaseModel, Field


class ProcessedOutputs(BaseModel):
    """Pydantic model for processed outputs."""

    prompt_ids: list[list[int]]
    prompt_mask: list[list[int]]
    completion_ids: list[list[int]]
    completion_mask: list[list[int]]
    rewards: list[float]


class BatchRequest(BaseModel):
    """Descriptor for a batch that will be provided to the requester."""

    batch_id: int
    env_inputs: dict[str, list[Any]] = Field(default_factory=dict)
    processing_class: Any | None = None
    mask_env_responses: bool = False
    max_seq_len: int = -1
    mask_truncated_completions: bool = False
    zero_truncated_completions: bool = False
    max_concurrent: int = 1


class BatchResult(BaseModel):
    """Result from batch generation"""

    batch_id: int
    processed_results: ProcessedOutputs
    generation_time: float = 0.0
    all_reward_dict: dict[str, list[float]] = Field(
        default_factory=dict
    )  # All reward scores
    completions: list[Any] = Field(
        default_factory=list
    )  # Store completions for logging
    prompts: list[Any] = Field(default_factory=list)  # Store prompts for logging


class AsyncBatchRequester:
    """Coordinates externally produced batches for GRPO training."""

    def __init__(
        self,
        model_name: str,
        num_batches_ahead: int = 1,
        max_queue_size: int | None = None,
        generation_timeout: float = 300.0,
    ):
        self.model_name = model_name
        self.num_batches_ahead = num_batches_ahead
        self.max_queue_size = max_queue_size or max(num_batches_ahead * 2, 4)
        self.generation_timeout = generation_timeout

        self.pending_batches: set[int] = set()
        self.completed_batches: dict[int, BatchResult] = {}
        self.generation_times: deque[float] = deque(maxlen=100)

        self.started = False

        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)

    def start(self):
        """Mark the requester as ready to accept batch submissions."""
        if self.started:
            return
        self.started = True

    def stop(self):
        """Stop accepting new batch submissions."""
        if not self.started:
            return
        self.started = False

    def request_batch(self, request: BatchRequest) -> bool:
        """Track that a batch will be supplied externally."""
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        with self._condition:
            if request.batch_id in self.completed_batches:
                return True

            if request.batch_id in self.pending_batches:
                return True

            if len(self.pending_batches) >= self.max_queue_size:
                return False

            self.pending_batches.add(request.batch_id)
            self._condition.notify_all()
            return True

    def submit_batch_result(self, result: BatchResult) -> None:
        """Submit a completed batch result that can later be retrieved."""
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        with self._condition:
            self.completed_batches[result.batch_id] = result
            self.pending_batches.discard(result.batch_id)
            self.generation_times.append(result.generation_time)
            self._condition.notify_all()

    def get_batch(self, batch_id: int, timeout: float | None = None) -> BatchResult:
        """Retrieve a completed batch result once it has been submitted."""
        timeout = timeout or self.generation_timeout
        deadline = time.time() + timeout

        with self._condition:
            while batch_id not in self.completed_batches:
                remaining = deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Batch {batch_id} was not submitted within {timeout}s"
                    )
                self._condition.wait(timeout=remaining)

            return self.completed_batches.pop(batch_id)

    def get_pending_count(self) -> int:
        """Get number of batches currently being generated"""
        with self._lock:
            return len(self.pending_batches)

    def wait_for_request(self, batch_id: int, timeout: float | None = None) -> bool:
        """Block until a specific batch_id has been requested."""
        if not self.started:
            raise RuntimeError("AsyncBatchGenerator not started")

        timeout = timeout or self.generation_timeout
        deadline = time.time() + timeout

        with self._condition:
            while batch_id not in self.pending_batches:
                remaining = deadline - time.time()
                if remaining <= 0:
                    return False
                self._condition.wait(timeout=remaining)
            return True

    def get_completed_count(self) -> int:
        """Get number of completed batches waiting to be retrieved"""
        with self._lock:
            return len(self.completed_batches)

    def get_average_generation_time(self) -> float:
        """Get average generation time for recent batches"""
        if not self.generation_times:
            return 0.0
        return sum(self.generation_times) / len(self.generation_times)

    def should_submit_more(self) -> bool:
        """Check if we should request more batches from the producer."""
        with self._lock:
            total_pending = len(self.pending_batches) + len(self.completed_batches)
            return total_pending < self.num_batches_ahead

    def get_pending_batch_ids(self) -> list[int]:
        """Return pending batch IDs in ascending order."""
        with self._lock:
            return sorted(self.pending_batches)
