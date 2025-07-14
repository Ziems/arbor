from arbor.server.api.models.schemas import (
    GRPOCheckpointRequest,
    GRPOInitializeRequest,
    GRPOStatusResponse,
    GRPOStepRequest,
    GRPOTerminateRequest,
)
from arbor.server.core.config import Settings
from arbor.server.services.jobs.grpo_job import GRPOJob
from arbor.server.services.managers.inference_manager import InferenceManager


class GRPOManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.grpo_jobs: dict[str, GRPOJob] = {}

    def initialize(
        self, request: GRPOInitializeRequest, inference_manager: InferenceManager
    ):
        grpo_job = GRPOJob(self.settings)
        grpo_job.initialize(request, inference_manager)
        grpo_job_id = grpo_job.grpo_job_id
        self.grpo_jobs[grpo_job_id] = grpo_job

        grpo_status_dict = grpo_job.get_status_dict()
        return GRPOStatusResponse(**grpo_status_dict)

    def route_grpo_step(
        self, request: GRPOStepRequest, inference_manager: InferenceManager
    ):
        grpo_job = self.grpo_jobs[request.grpo_job_id]
        grpo_job.step(request, inference_manager)

        grpo_status_dict = grpo_job.get_status_dict()
        return GRPOStatusResponse(**grpo_status_dict)

    def route_grpo_checkpoint(
        self, request: GRPOCheckpointRequest, inference_manager: InferenceManager
    ):
        grpo_job = self.grpo_jobs[request.grpo_job_id]
        grpo_job.checkpoint(request, inference_manager)

        grpo_status_dict = grpo_job.get_status_dict()
        return GRPOStatusResponse(**grpo_status_dict)

    def route_grpo_terminate(
        self, request: GRPOTerminateRequest, inference_manager: InferenceManager
    ):
        grpo_job = self.grpo_jobs[request.grpo_job_id]
        grpo_job.terminate(request, inference_manager)

        grpo_status_dict = grpo_job.get_status_dict()
        return GRPOStatusResponse(**grpo_status_dict)
