from typing import TYPE_CHECKING, Optional

from arbor.server.api.models.schemas import (
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    GRPOCheckpointRequest,
    GRPOTerminateRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.utils.error_handling import (
    TrainingError,
    handle_error,
    record_error,
)

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
        try:
            grpo_job = GRPOJob(self.config, request, gpu_manager=self.gpu_manager)
            grpo_job.initialize(request, inference_manager)
            self.grpo_jobs[grpo_job.id] = grpo_job
            return grpo_job.get_status()
        except Exception as exc:
            payload = request.model_dump()
            handle_error(exc, {"stage": "initialize", "request": payload})
            raise TrainingError(
                "Failed to initialize GRPO job", request=payload
            ) from exc

    def get_job_status(self, job_id: str) -> GRPOStatus:
        grpo_job = self.grpo_jobs.get(job_id)
        if grpo_job is None:
            record_error(
                f"GRPO job {job_id} not found", "GRPOJobMissing", {"job_id": job_id}
            )
            raise ValueError(f"GRPO job {job_id} not found")
        return grpo_job.get_status()

    def route_grpo_step(self, request: GRPOStepRequest):
        grpo_job = self.grpo_jobs.get(request.job_id)
        if grpo_job is None:
            record_error(
                f"GRPO job {request.job_id} not found",
                "GRPOJobMissing",
                {"job_id": request.job_id, "batch_id": request.batch_id},
            )
            raise ValueError(f"GRPO job {request.job_id} not found")
        try:
            grpo_job.grpo_step(request)
            return grpo_job.get_status()
        except Exception as exc:
            context = {"job_id": request.job_id, "batch_id": request.batch_id}
            handle_error(exc, {"stage": "step", **context})
            raise TrainingError("GRPO step failed", **context) from exc

    def route_grpo_checkpoint(self, request: GRPOCheckpointRequest) -> GRPOStatus:
        grpo_job = self.grpo_jobs.get(request.job_id)
        if grpo_job is None:
            record_error(
                f"GRPO job {request.job_id} not found",
                "GRPOJobMissing",
                {"job_id": request.job_id, "checkpoint": request.checkpoint_name},
            )
            raise ValueError(f"GRPO job {request.job_id} not found")
        try:
            return grpo_job.checkpoint(request)
        except Exception as exc:
            context = {"job_id": request.job_id, "checkpoint": request.checkpoint_name}
            handle_error(exc, {"stage": "checkpoint", **context})
            raise TrainingError("GRPO checkpoint failed", **context) from exc

    def cancel(self, job_id: str) -> GRPOStatus:
        """Cancel a GRPO job"""
        grpo_job = self.grpo_jobs.get(job_id)
        if grpo_job is None:
            record_error(
                f"GRPO job {job_id} not found", "GRPOJobMissing", {"job_id": job_id}
            )
            raise ValueError(f"GRPO job {job_id} not found")
        try:
            grpo_job.cancel()
            return grpo_job.get_status()
        except Exception as exc:
            handle_error(exc, {"stage": "cancel", "job_id": job_id})
            raise TrainingError("Failed to cancel GRPO job", job_id=job_id) from exc

    def terminate(self, request: GRPOTerminateRequest) -> GRPOStatus:
        grpo_job = self.grpo_jobs.get(request.job_id)
        if grpo_job is None:
            record_error(
                f"GRPO job {request.job_id} not found",
                "GRPOJobMissing",
                {"job_id": request.job_id},
            )
            raise ValueError(f"GRPO job {request.job_id} not found")
        try:
            grpo_job.save_final_checkpoint()
            grpo_job.terminate_training()
            grpo_job.terminate_process(stop_inference=False, release_gpus=False)
            return grpo_job.get_status()
        except Exception as exc:
            handle_error(exc, {"stage": "terminate", "job_id": request.job_id})
            raise TrainingError(
                "Failed to terminate GRPO job", job_id=request.job_id
            ) from exc

    def cleanup(self) -> None:
        """Clean up all GRPO jobs and their resources"""
        if self._cleanup_called:
            return

        self.logger.info(f"Cleaning up {len(self.grpo_jobs)} GRPO jobs...")

        for job_id, grpo_job in self.grpo_jobs.items():
            try:
                self.logger.debug(f"Cleaning up GRPO job {job_id}")
                grpo_job.cancel()
            except Exception as exc:
                handle_error(exc, {"stage": "cleanup", "job_id": job_id})

        self.grpo_jobs.clear()
        self._cleanup_called = True
        self.logger.info("GRPOManager cleanup completed")
