from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatus,
    GRPOStepRequest,
    GRPOTerminateRequest,
)
from arbor.server.core.config import Config
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.inference_manager import InferenceManager


class GRPOManager:
    def __init__(self, config: Config):
        self.config = config
        self.grpo_jobs: dict[str, GRPOJob] = {}

    def initialize(
        self, request: GRPOInitializeRequest, inference_manager: InferenceManager
    ):
        grpo_job = GRPOJob(self.config, request)
        grpo_job.initialize(request, inference_manager)
        self.grpo_jobs[grpo_job.id] = grpo_job

        return grpo_job.get_status()

    def route_grpo_step(self, request: GRPOStepRequest):
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.grpo_step(request)

        return grpo_job.get_status()

    def route_grpo_checkpoint(self, request: GRPOCheckpointRequest):
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.checkpoint(request)

        return grpo_job.get_status()

    def terminate(self, request: GRPOTerminateRequest):
        grpo_job = self.grpo_jobs[request.job_id]
        grpo_job.terminate()
        # TODO: inference_manager.terminate_job(grpo_job.inference_job)
        # TODO: Maybe also update job_manager or resource manager

        return grpo_job.get_status()
