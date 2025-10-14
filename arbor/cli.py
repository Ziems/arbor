import os
from datetime import datetime

import click
import uvicorn

from arbor.server.core.config import Config
from arbor.server.main import app
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.services.managers.file_train_manager import FileTrainManager
from arbor.server.services.managers.gpu_manager import GPUManager
from arbor.server.services.managers.grpo_manager import GRPOManager
from arbor.server.services.managers.inference_manager import InferenceManager
from arbor.server.services.managers.job_manager import JobManager
from arbor.server.utils.logging import (
    get_logger,
    log_configuration,
    setup_logging,
)


def make_log_dir(storage_path: str):
    # Create a timestamped log directory under the storage path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(storage_path, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


@click.group()
def cli():
    pass


def create_app(arbor_config_path: str):
    """Create and configure the Arbor API application

    Args:
        arbor_config_path (str): Path to config file

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create new settings instance with overrides
    config = Config.load(arbor_config_path)
    log_dir = make_log_dir(config.storage_path)
    app.state.log_dir = log_dir

    # Setup logging
    logging_config = setup_logging(
        log_level="INFO",
        log_dir=log_dir,
        enable_file_logging=True,
        enable_console_logging=True,
    )

    # Log configuration and system info
    log_configuration(logging_config)

    # Get logger for this module
    logger = get_logger(__name__)
    logger.info("Initializing Arbor application...")

    # Log system information via health manager
    try:
        versions = config.get_system_versions()
        logger.info("System versions:")
        for category, version_info in versions.items():
            if isinstance(version_info, dict):
                logger.info(f"  {category}:")
                for lib, version in version_info.items():
                    logger.info(f"    {lib}: {version}")
            else:
                logger.info(f"  {category}: {version_info}")
    except Exception as e:
        logger.warning(f"Could not log system versions: {e}")

    # Initialize services with config
    logger.info("Initializing services...")
    gpu_manager = GPUManager(config=config)
    file_manager = FileManager(config=config, gpu_manager=gpu_manager)
    job_manager = JobManager(config=config, gpu_manager=gpu_manager)
    file_train_manager = FileTrainManager(config=config, gpu_manager=gpu_manager)
    inference_manager = InferenceManager(config=config, gpu_manager=gpu_manager)
    grpo_manager = GRPOManager(config=config, gpu_manager=gpu_manager)

    # Inject config into app state
    app.state.config = config
    app.state.gpu_manager = gpu_manager
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.file_train_manager = file_train_manager
    app.state.inference_manager = inference_manager
    app.state.grpo_manager = grpo_manager

    logger.info("Arbor application initialized successfully")
    return app


def start_server(host="0.0.0.0", port=7453, storage_path="./storage", timeout=10):
    """Start the Arbor API server with a single function call"""
    import socket
    import threading
    import time
    from contextlib import closing

    def is_port_in_use(port):
        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
            return sock.connect_ex(("localhost", port)) == 0

    # First ensure the port is free
    if is_port_in_use(port):
        raise RuntimeError(f"Port {port} is already in use")

    app = create_app(storage_path)
    # configure_uvicorn_logging()
    config = uvicorn.Config(
        app, host=host, port=port, log_level="info", access_log=False
    )
    server = uvicorn.Server(config)

    def run_server():
        server.run()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for server to start
    start_time = time.time()
    while not is_port_in_use(port):
        if time.time() - start_time > timeout:
            raise TimeoutError(f"Server failed to start within {timeout} seconds")
        time.sleep(0.1)

    # Give it a little extra time to fully initialize
    time.sleep(0.5)

    return server


def stop_server(server):
    """Stop the Arbor API server"""
    server.should_exit = True


@cli.command()
@click.option("--host", default="0.0.0.0", help="Host to bind to")
@click.option("--port", default=7453, help="Port to bind to")
@click.option("--arbor-config", required=False, help="Path to the Arbor config file")
def serve(host, port, arbor_config):
    """Start the Arbor API server"""

    # Just use the config path or None (will use defaults)
    config_path = arbor_config

    try:
        create_app(config_path)

        # Add signal handling for graceful shutdown
        import signal

        def signal_handler(signum, frame):
            click.echo("\nShutting down server gracefully...")
            # The FastAPI lifespan handler will handle the actual cleanup
            raise KeyboardInterrupt()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        # Temporarily disable custom uvicorn logging configuration
        # configure_uvicorn_logging()
        uvicorn.run(app, host=host, port=port, access_log=False)
    except KeyboardInterrupt:
        click.echo("Server shutdown completed.")
    except Exception as e:
        click.echo(f"Failed to start server: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
