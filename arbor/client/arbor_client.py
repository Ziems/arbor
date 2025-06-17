import os
from datetime import datetime

import click
import uvicorn

from arbor.server.core.config import Config
from arbor.server.core.config_manager import ConfigManager
from arbor.server.main import app
from arbor.server.services.file_manager import FileManager
from arbor.server.services.grpo_manager import GRPOManager
from arbor.server.services.health_manager import HealthManager
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager



def create_app(
    config_path: str = None, storage_path: str = None, inference_gpus: str = None, training_gpus: str = None
):
    """Create and configure the Arbor API application

    Args:
        arbor_config_path (str): Path to config file
        storage_path (str): Path to storage directory
        inference_gpus (str): gpu ids to use for inference
        training_gpus (str): gpu ids to use for training

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create new config instance with overrides
    if config_path:
        config = Config.load_config_from_yaml(config_path)
    else:
        config = Config.load_config_directly(storage_path, inference_gpus, training_gpus)

    app.state.log_dir = Config.make_log_dir(config.STORAGE_PATH)

    # Initialize services with config
    health_manager = HealthManager(config=config)
    file_manager = FileManager(config=config)
    job_manager = JobManager(config=config)
    training_manager = TrainingManager(config=config)
    inference_manager = InferenceManager(config=config)
    grpo_manager = GRPOManager(config=config)

    # Inject config into app state
    app.state.config = config
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.training_manager = training_manager
    app.state.inference_manager = inference_manager
    app.state.grpo_manager = grpo_manager
    app.state.health_manager = health_manager

    return app


def client_arbor_serve(
    config_path: str = None,
    storage_path: str = None,
    inference_gpus: str = None,
    training_gpus: str = None,
    host: str = "0.0.0.0",
    port: int = 7453,
):
    create_app(config_path, storage_path, inference_gpus, training_gpus)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    client_arbor_serve(inference_gpus="0, 1", training_gpus="2, 3")
