from typing import Any, Optional

from arbor.core.config import Config
from arbor.server.api.schemas import InferenceLaunchRequest
from arbor.server.services.comms.control_server import TrainerControlServer
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.services.managers.gpu_manager import GPUManager


class InferenceManager(BaseManager):
    def __init__(self, config: Config, gpu_manager: GPUManager):
        super().__init__(config)
        self.inference_jobs: dict[str, InferenceJob] = {}
        self.alias_to_job_id: dict[str, str] = {}
        if gpu_manager is None:
            raise ValueError("InferenceManager requires a GPUManager instance")
        self.gpu_manager = gpu_manager

    # TODO: request_json should be checked for launch_model_config or something
    async def route_inference(self, request_json: dict):
        model = request_json["model"]
        self.logger.debug(f"Running inference for model {model}")

        # If model isnt launched, launch it
        # TODO: Check that there are GPUs available. If not, do hot swap or something.
        inference_job = self._get_job_by_alias(model)
        if inference_job is None:
            try:
                inference_job = self.launch_from_request(
                    InferenceLaunchRequest(model=model)
                )
            except Exception as e:
                self.logger.error(f"Error launching model {model}: {e}")
                raise e

        return await inference_job.run_inference(request_json)

    def _register_job(self, job: InferenceJob, aliases: list[str]) -> None:
        self.inference_jobs[job.id] = job
        for alias in aliases:
            if alias:
                self.alias_to_job_id[alias] = job.id
        self._cleanup_called = False

    def _get_job_by_alias(self, alias: Optional[str]) -> Optional[InferenceJob]:
        if not alias:
            return None
        job_id = self.alias_to_job_id.get(alias)
        if job_id is None:
            return None
        return self.inference_jobs.get(job_id)

    def _unregister_job(self, job: InferenceJob) -> list[str]:
        aliases = [
            alias
            for alias, job_id in list(self.alias_to_job_id.items())
            if job_id == job.id
        ]
        for alias in aliases:
            self.alias_to_job_id.pop(alias, None)
        self.inference_jobs.pop(job.id, None)
        return aliases

    def launch_from_request(
        self,
        launch_request: InferenceLaunchRequest,
        trainer_controller: Optional[TrainerControlServer] = None,
        *,
        reuse_existing: bool = True,
        preallocated_gpu_ids: Optional[list[int]] = None,
    ) -> InferenceJob:
        model_name = launch_request.model

        if reuse_existing and launch_request.owner is None:
            existing_job = self._get_job_by_alias(model_name)
            if existing_job is not None:
                self.logger.debug(
                    "Reusing existing inference job %s for model %s",
                    existing_job.id,
                    model_name,
                )
                return existing_job

        launch_config = InferenceLaunchConfig(
            max_seq_len=launch_request.max_seq_len,
            gpu_ids=list(preallocated_gpu_ids) if preallocated_gpu_ids else None,
            num_gpus=launch_request.num_gpus,
            log_file_path=launch_request.log_file_path,
        )

        if preallocated_gpu_ids:
            launch_config.num_gpus = len(preallocated_gpu_ids)

        if launch_config.num_gpus is not None and launch_config.num_gpus <= 0:
            raise ValueError("num_gpus must be greater than 0")

        owner = launch_request.owner
        if owner is not None:
            if owner.type == "grpo":
                launch_config.grpo_job_id = owner.job_id
            else:
                raise ValueError(
                    f"Unsupported inference launch owner type: {owner.type}"
                )

        if launch_config.grpo_job_id and trainer_controller is None:
            raise ValueError("Trainer controller is required for GRPO inference jobs")

        return self.launch_job(model_name, launch_config, trainer_controller)

    def launch_job(
        self,
        model: str,
        launch_config: InferenceLaunchConfig,
        trainer_controller: Optional[TrainerControlServer] = None,
    ) -> InferenceJob:
        is_grpo_sub_job = bool(launch_config.grpo_job_id)
        inference_job = InferenceJob(self.config, is_grpo_sub_job=is_grpo_sub_job)

        # Use provided GPU IDs or allocate through GPU manager
        allocated_here = False
        if launch_config.gpu_ids is None:
            requested_gpus = launch_config.num_gpus or 1
            allocated_gpus = self.gpu_manager.allocate_gpus(
                inference_job.id, requested_gpus
            )
            launch_config.gpu_ids = allocated_gpus
            allocated_here = True
            self.logger.info(
                f"Allocated GPUs {allocated_gpus} for inference job {inference_job.id}"
            )

        assert launch_config.gpu_ids is not None, "GPU IDs must be set before launching"
        assert len(launch_config.gpu_ids) > 0, (
            "Inference Job must have at least one GPU in gpu_ids. "
            f"Currently set to {launch_config.gpu_ids}"
        )

        try:
            inference_job.launch(model, launch_config, trainer_controller)
        except Exception:
            if allocated_here:
                self.gpu_manager.release_gpus(inference_job.id)
            raise

        aliases = [inference_job.id]
        if launch_config.grpo_job_id:
            aliases.append(launch_config.grpo_job_id)
        else:
            aliases.append(model)

        self._register_job(inference_job, aliases)

        self.logger.debug(
            "Active inference job aliases: %s",
            list(self.alias_to_job_id.keys()),
        )
        return inference_job

    def cleanup(self) -> None:
        """Clean up all inference jobs and their resources"""
        if self._cleanup_called and not self.inference_jobs:
            return

        self.logger.info(f"Cleaning up {len(self.inference_jobs)} inference jobs...")

        for job_id in list(self.inference_jobs.keys()):
            try:
                self.terminate_job(job_id)
            except Exception as e:
                self.logger.error(f"Error cleaning up inference job {job_id}: {e}")
        self._cleanup_called = True
        self.logger.info("InferenceManager cleanup completed")

    def terminate_job(
        self,
        job_id: str,
    ) -> dict[str, Any]:
        job = self.inference_jobs.get(job_id)
        if job is None:
            raise ValueError(f"Inference job '{job_id}' not found")

        aliases: list[str] = []
        release_exception: Optional[Exception] = None
        try:
            try:
                self.gpu_manager.release_gpus(job.id)
            except Exception as exc:
                release_exception = exc
                self.logger.error(
                    "Failed to release GPUs for inference job %s: %s",
                    job.id,
                    exc,
                )

            job.terminate()
        finally:
            aliases = self._unregister_job(job)

        self.logger.info("Terminated inference job %s (aliases=%s)", job.id, aliases)

        if release_exception is not None:
            raise release_exception

        return {"job_id": job.id, "aliases": aliases}
