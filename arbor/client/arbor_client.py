import os
from datetime import datetime

import click
import uvicorn

from arbor.server.core.config import Settings
from arbor.server.core.config_manager import ConfigManager
from arbor.server.main import app
from arbor.server.services.file_manager import FileManager
from arbor.server.services.grpo_manager import GRPOManager
from arbor.server.services.health_manager import HealthManager
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager


def make_log_dir(storage_path: str):
    # Create a timestamped log directory under the storage path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(storage_path, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_app(
    arbor_config_path: str = None, inference_gpus: str = None, training_gpus: str = None
):
    """Create and configure the Arbor API application

    Args:
        arbor_config_path (str): Path to config file

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create new settings instance with overrides
    if arbor_config_path:
        settings = Settings.load_from_yaml(arbor_config_path)
    else:
        settings = Settings.from_gpus(inference_gpus, training_gpus)

    app.state.log_dir = make_log_dir(settings.STORAGE_PATH)

    # Initialize services with settings
    health_manager = HealthManager(settings=settings)
    file_manager = FileManager(settings=settings)
    job_manager = JobManager(settings=settings)
    training_manager = TrainingManager(settings=settings)
    inference_manager = InferenceManager(settings=settings)
    grpo_manager = GRPOManager(settings=settings)

    # Inject settings into app state
    app.state.settings = settings
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.training_manager = training_manager
    app.state.inference_manager = inference_manager
    app.state.grpo_manager = grpo_manager
    app.state.health_manager = health_manager

    return app


def client_arbor_serve(
    arbor_config_path: str = None,
    inference_gpus: str = None,
    training_gpus: str = None,
    host: str = "0.0.0.0",
    port: int = 7453,
):
    create_app(arbor_config_path, inference_gpus, training_gpus)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    client_arbor_serve(inference_gpus="0, 1", training_gpus="2, 3")
