from arbor.server.core.config import Config
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.services.managers.base_manager import BaseManager


class InferenceManager(BaseManager):
    def __init__(self, config: Config):
        super().__init__(config)
        self.inference_jobs: dict[str, InferenceJob] = {}

    # TODO: request_json should be checked for launch_model_config or something
    async def route_inference(self, request_json: dict):
        model = request_json["model"]
        self.logger.info(f"Running inference for model {model}")

        # If model isnt launched, launch it
        # TODO: Check that there are GPUs available. If not, do hot swap or something.
        inference_job = self.inference_jobs.get(model, None)
        if inference_job is None:
            try:
                inference_job = InferenceJob(self.config)
                inference_launch_config = InferenceLaunchConfig(
                    gpu_ids=self.config.arbor_config.inference.gpu_ids
                )
                inference_job.launch(model, inference_launch_config)
                # This needs to have a unique id or something, not be referenced by model
                self.inference_jobs[model] = inference_job
            except Exception as e:
                self.logger.error(f"Error launching model {model}: {e}")
                raise e

        return await inference_job.run_inference(request_json)

    def launch_job(self, model: str, launch_kwargs: InferenceLaunchConfig):
        inference_job = InferenceJob(self.config)
        inference_job.launch(model, launch_kwargs)
        if launch_kwargs.is_grpo and launch_kwargs.grpo_job_id:
            self.inference_jobs[launch_kwargs.grpo_job_id] = inference_job
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
                if hasattr(inference_job, "kill"):
                    inference_job.kill()
                elif hasattr(inference_job, "cleanup"):
                    inference_job.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up inference job {job_id}: {e}")

        self.inference_jobs.clear()
        self._cleanup_called = True
        self.logger.info("InferenceManager cleanup completed")
