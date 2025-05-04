import click
import uvicorn

from arbor.server.core.config import Settings
from arbor.server.main import app
from arbor.server.services.file_manager import FileManager
from arbor.server.services.grpo_manager import GRPOManager
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager


@click.group()
def cli():
    pass


def create_app(arbor_config_path: str):
    """Create and configure the Arbor API application

    Args:
        storage_path (str): Path to store models and uploaded training files

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create new settings instance with overrides
    settings = Settings.load_from_yaml(arbor_config_path)

    # Initialize services with settings
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
    config = uvicorn.Config(app, host=host, port=port, log_level="info")
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
@click.option("--arbor-config", required=True, help="Path to the Arbor config file")
def serve(host, port, arbor_config):
    """Start the Arbor API server"""
    app = create_app(arbor_config)
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    cli()
