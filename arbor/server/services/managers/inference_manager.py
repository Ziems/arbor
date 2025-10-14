from arbor.server.core.config import Config
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.comms.control_server import TrainerControlServer
from arbor.server.services.managers.base_manager import BaseManager


class InferenceManager(BaseManager):
    def __init__(self, config: Config, gpu_manager: GPUManager):
        super().__init__(config)
        self.inference_jobs: dict[str, InferenceJob] = {}
        if gpu_manager is None:
            raise ValueError("InferenceManager requires a GPUManager instance")
        self.gpu_manager = gpu_manager

    # TODO: request_json should be checked for launch_model_config or something
    async def route_inference(self, request_json: dict):
        model = request_json["model"]
        self.logger.debug(f"Running inference for model {model}")

        # If model isnt launched, launch it
        # TODO: Check that there are GPUs available. If not, do hot swap or something.
        inference_job = self.inference_jobs.get(model, None)
        if inference_job is None:
            try:
                inference_job = InferenceJob(self.config)

                allocated_gpus = self.gpu_manager.allocate_gpus(inference_job.id, 1)
                self.logger.info(
                    f"Allocated GPUs {allocated_gpus} for inference job {inference_job.id}"
                )

                inference_launch_config = InferenceLaunchConfig(
                    gpu_ids=allocated_gpus,
                )

                inference_job.launch(model, inference_launch_config)
                # This needs to have a unique id or something, not be referenced by model
                self.inference_jobs[model] = inference_job
            except Exception as e:
                self.logger.error(f"Error launching model {model}: {e}")
                # Release GPUs if allocation succeeded but launch failed
                self.gpu_manager.release_gpus(inference_job.id)
                raise e

        return await inference_job.run_inference(request_json)

    def launch_job(
        self,
        model: str,
        launch_config: InferenceLaunchConfig,
        trainer_controller: TrainerControlServer,
    ):
        inference_job = InferenceJob(self.config)

        # Use provided GPU IDs or allocate through GPU manager
        if launch_config.gpu_ids is None:
            allocated_gpus = self.gpu_manager.allocate_gpus(inference_job.id, 1)
            launch_config.gpu_ids = allocated_gpus
            self.logger.info(
                f"Allocated GPUs {allocated_gpus} for inference job {inference_job.id}"
            )

        assert launch_config.gpu_ids is not None, "GPU IDs must be set before launching"
        assert (
            len(launch_config.gpu_ids) > 0
        ), f"Inference Job must have at least one GPU in gpu_ids. Currently set to {launch_config.gpu_ids}"

        inference_job.launch(model, launch_config, trainer_controller)
        if launch_config.is_grpo and launch_config.grpo_job_id:
            self.inference_jobs[launch_config.grpo_job_id] = inference_job
        else:
            self.inference_jobs[model] = inference_job

        self.logger.debug(f"Active inference jobs: {list(self.inference_jobs.keys())}")
        return inference_job

    def cleanup(self) -> None:
        """Clean up all inference jobs and their resources"""
        if self._cleanup_called:
            return

        self.logger.info(f"Cleaning up {len(self.inference_jobs)} inference jobs...")

        for job_id, inference_job in self.inference_jobs.items():
            try:
                self.logger.debug(f"Cleaning up inference job {job_id}")

                # Release GPUs first
                self.gpu_manager.release_gpus(inference_job.id)

                # Terminate the job
                inference_job.terminate()
            except Exception as e:
                self.logger.error(f"Error cleaning up inference job {job_id}: {e}")

        self.inference_jobs.clear()
        self._cleanup_called = True
        self.logger.info("InferenceManager cleanup completed")
