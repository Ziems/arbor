"""
Simple GPU Manager for tracking GPU allocations.
"""

import threading
from typing import Dict, List, Optional, Set

from arbor.server.core.config import Config
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.utils.logging import get_logger


class GPUAllocationError(Exception):
    """Raised when requested GPUs are not available."""

    pass


class GPUManager(BaseManager):
    """
    Simple GPU manager for tracking which GPUs are allocated to which jobs.

    Prevents multiple jobs from using the same GPUs simultaneously.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = get_logger(__name__)

        # Thread safety
        self._lock = threading.Lock()

        # Available GPUs from config
        self.all_gpus: Set[int] = set(config.gpu_ids)

        # Track which job has which GPUs
        self.gpu_allocations: Dict[str, List[int]] = {}  # job_id -> [gpu_ids]

        self.logger.info(f"GPUManager initialized with GPUs: {sorted(self.all_gpus)}")

    def get_all_allocated_gpus(self) -> Set[int]:
        """Get set of all currently allocated GPUs across all jobs."""
        allocated = set()
        for gpus in self.gpu_allocations.values():
            allocated.update(gpus)
        return allocated

    def allocate_gpus(self, job_id: str, num_gpus: int) -> List[int]:
        """
        Allocate GPUs to a job.

        Args:
            job_id: Unique identifier for the job
            num_gpus: Number of GPUs to allocate

        Returns:
            List of allocated GPU IDs

        Raises:
            GPUAllocationError: If not enough GPUs are available
        """
        with self._lock:
            # Get currently allocated GPUs
            allocated_gpus = set()
            for gpus in self.gpu_allocations.values():
                allocated_gpus.update(gpus)

            # Find free GPUs
            free_gpus = self.all_gpus - allocated_gpus

            if len(free_gpus) < num_gpus:
                raise GPUAllocationError(
                    f"Not enough free GPUs. Requested: {num_gpus}, "
                    f"Available: {len(free_gpus)} {sorted(free_gpus)}, "
                    f"Allocated: {sorted(allocated_gpus)}"
                )

            # Allocate the first N available GPUs
            allocated = sorted(list(free_gpus))[:num_gpus]
            self.gpu_allocations[job_id] = allocated

            self.logger.info(f"Allocated GPUs {allocated} to job {job_id}")
            return allocated

    def get_allocated_gpus(self, job_id: str) -> Optional[List[int]]:
        """Get the GPUs allocated to a specific job."""
        with self._lock:
            return self.gpu_allocations.get(job_id)

    def release_gpus(self, job_id: str) -> bool:
        """
        Release GPUs allocated to a job.

        Args:
            job_id: The job to release GPUs for

        Returns:
            True if GPUs were released, False if job had no allocation
        """
        with self._lock:
            if job_id in self.gpu_allocations:
                released_gpus = self.gpu_allocations[job_id]
                del self.gpu_allocations[job_id]
                self.logger.info(f"Released GPUs {released_gpus} from job {job_id}")
                return True
            return False

    def get_status(self) -> Dict:
        """Get current GPU allocation status."""
        with self._lock:
            allocated_gpus = set()
            for gpus in self.gpu_allocations.values():
                allocated_gpus.update(gpus)

            free_gpus = self.all_gpus - allocated_gpus

            return {
                "total_gpus": sorted(list(self.all_gpus)),
                "free_gpus": sorted(list(free_gpus)),
                "allocated_gpus": sorted(list(allocated_gpus)),
                "allocations": dict(self.gpu_allocations),
            }

    def cleanup(self) -> None:
        """Clean up all GPU allocations."""
        if self._cleanup_called:
            return

        with self._lock:
            allocation_count = len(self.gpu_allocations)
            if allocation_count > 0:
                self.logger.info(f"Cleaning up {allocation_count} GPU allocations...")
                self.gpu_allocations.clear()
                self.logger.info("All GPU allocations released")

        super().cleanup()
