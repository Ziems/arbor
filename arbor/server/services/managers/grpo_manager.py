from typing import TYPE_CHECKING, Optional

from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    GRPOTerminateRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.services.managers.inference_manager import InferenceManager

if TYPE_CHECKING:
    from arbor.server.services.managers.gpu_manager import GPUManager


class GRPOManager(BaseManager):
    def __init__(self, config: Config, gpu_manager: Optional["GPUManager"] = None):
        super().__init__(config)
        self.grpo_jobs: dict[str, GRPOJob] = {}
        self.gpu_manager = gpu_manager

    def initialize(
        self, request: GRPOInitializeRequest, inference_manager: InferenceManager
    ):
        grpo_job = GRPOJob(self.config, request, gpu_manager=self.gpu_manager)
        grpo_job.initialize(request, inference_manager)
        self.grpo_jobs[grpo_job.id] = grpo_job

        return grpo_job.get_status()

    def get_job_status(self, job_id: str) -> GRPOStatus:
        grpo_job = self.grpo_jobs[job_id]
        return grpo_job.get_status()

    def route_grpo_step(self, request: GRPOStepRequest):
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.grpo_step(request)

        return grpo_job.get_status()

    def route_grpo_checkpoint(self, request: GRPOCheckpointRequest):
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.checkpoint(request)

        return grpo_job.get_status()

    def cancel(self, job_id: str) -> GRPOStatus:
        """Cancel a GRPO job"""
        if job_id not in self.grpo_jobs:
            raise ValueError(f"GRPO job {job_id} not found")

        grpo_job = self.grpo_jobs[job_id]
        grpo_job.cancel()

        return grpo_job.get_status()

    def terminate(self, request: GRPOTerminateRequest) -> GRPOStatus:
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.terminate_training()
        grpo_job.terminate_process()
        return grpo_job.get_status()

    def cleanup(self) -> None:
        """Clean up all GRPO jobs and their resources"""
        if self._cleanup_called:
            return

        self.logger.info(f"Cleaning up {len(self.grpo_jobs)} GRPO jobs...")

        for job_id, grpo_job in self.grpo_jobs.items():
            try:
                self.logger.debug(f"Cleaning up GRPO job {job_id}")
                # All GRPO jobs have cancel method
                grpo_job.cancel()
            except Exception as e:
                self.logger.error(f"Error cleaning up GRPO job {job_id}: {e}")

        self.grpo_jobs.clear()
        self._cleanup_called = True
        self.logger.info("GRPOManager cleanup completed")
