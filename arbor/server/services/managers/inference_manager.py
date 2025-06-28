import asyncio
import subprocess
from datetime import datetime
from typing import Any, Dict, Optional

from arbor.server.core.config import Settings
from arbor.server.services.jobs.inference_job import InferenceJob
from arbor.server.utils.helpers import strip_prefix
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class InferenceLaunchConfig:
    max_context_length: Optional[int] = None
    gpu_ids: Optional[list[int]] = None


class InferenceManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.process: Optional[subprocess.Popen] = None
        self.inference_jobs: dict[str, InferenceJob] = {}

    # TODO: request_json should be checked for launch_model_config
    async def route_inference(self, request_json: dict):
        logger.info(f"Running inference for model {model}")

        # If model isnt launched, launch it
        inference_job = self.inference_jobs.get(model, None)
        if inference_job is None:
            try:
                inference_job = InferenceJob(self.settings)
                inference_launch_config = InferenceLaunchConfig()
                inference_job.launch(model, inference_launch_config)
                self.inference_jobs[model] = inference_job
            except Exception as e:
                logger.error(f"Error launching model {model}: {e}")
                raise e

        # If model is launched and different, swap the server
        # If more inference GPUs are available, launch a new server

        if model != self.launched_model:
            model = self.launched_model
            request_json["model"] = model

        # Update last_activity timestamp
        self.last_activity = datetime.now()

        if self.process is None:
            raise RuntimeError("Server is not running. Please launch it first.")

        return await self.vllm_client.chat(json_body=request_json)

    def launch_model(self, model: str, launch_kwargs: LaunchModelConfig):
        pass
