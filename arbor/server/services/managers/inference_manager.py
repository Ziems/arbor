import subprocess
from datetime import datetime
from typing import Optional

from arbor.server.core.config import Settings
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.services.jobs.inference_launch_config import InferenceLaunchConfig
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


class InferenceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: Optional[subprocess.Popen] = None
        self.inference_jobs: dict[str, InferenceJob] = {}

    # TODO: request_json should be checked for launch_model_config
    async def route_inference(self, request_json: dict):
        model = request_json["model"]
        logger.info(f"Running inference for model {model}")

        # If model isnt launched, launch it
        inference_job = self.inference_jobs.get(model, None)
        if inference_job is None:
            try:
                inference_job = InferenceJob(self.settings)
                inference_launch_config = InferenceLaunchConfig(gpu_ids=[2])
                inference_job.launch(model, inference_launch_config)
                self.inference_jobs[model] = inference_job
            except Exception as e:
                logger.error(f"Error launching model {model}: {e}")
                raise e

        return await inference_job.run_inference(request_json)

    def launch_model(self, model: str, launch_kwargs: InferenceLaunchConfig):
        """Launch model directly with launch_kwargs"""
        pass
