import os
from datetime import datetime

import click
import uvicorn

from arbor.server.core.config import Settings
from arbor.server.core.config_manager import ConfigManager
from arbor.server.main import app
from arbor.server.services.file_manager import FileManager
from arbor.server.services.grpo_manager import GRPOManager
from arbor.server.services.inference_manager import InferenceManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager


def make_log_dir(storage_path: str):
    # Create a timestamped log directory under the storage path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(storage_path, "logs", timestamp)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


@click.group()
def cli():
    pass

@cli.group()
def config():
    """Manage Arbor configuration."""
    pass


def create_app(arbor_config_path: str):
    """Create and configure the Arbor API application

    Args:
        arbor_config_path (str): Path to config file 

    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create new settings instance with overrides
    settings = Settings.load_from_yaml(arbor_config_path)
    app.state.log_dir = make_log_dir(settings.STORAGE_PATH)

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
@click.option("--arbor-config", required=False, help="Path to the Arbor config file")
def serve(host, port, arbor_config):
    """Start the Arbor API server"""
    config_path = arbor_config if arbor_config else Settings.find_config_path()
    create_app(config_path)
        
    uvicorn.run(app, host=host, port=port)


@config.command()
@click.option("--inference", default="0", help="GPU IDs for inference (default: 0)")
@click.option("--training", default="1, 2", help="GPU IDs for training (default: 1, 2)")
def init(inference, training):
    """ Init Arbor configuration. """
    try:
        click.echo("Setting up Arbor configuration...")
        click.echo()
        
        # Get config details
        inference = click.prompt("GPU IDs for inference", default=inference)
        training = click.prompt("GPU IDs for training", default=training)
        click.echo()
        
        # Get config file path
        config_path = click.prompt("Enter path to save config file in", default=ConfigManager.get_default_config_path())
        click.echo()
        
        # Update or create config at path
        config_path = ConfigManager.update_config(inference, training, config_path)
        click.echo(f"Created configuration at: {config_path}")
        
        
        click.echo(f"\nYou can now start Arbor with: arbor serve")
        
        # Confirm config file to use
        config_path = click.prompt("Please enter the config file path you wish to use. Press enter if you wish to use the config file you just created.", default=config_path)
        
        # Check if it is a valid config file
        is_valid, msg = ConfigManager.validate_config_file(config_path)
        if not is_valid: raise click.ClickException(f"Invalid config file: {msg}")

        # Read and display the contents
        _, content = ConfigManager.get_config_contents(config_path)

        click.echo("\nConfiguration file contents:")
        click.echo("---")
        click.echo(content)
        click.echo("---")
        
        # Get host and port
        host = click.prompt("Host to bind to", default="0.0.0.0")
        port = click.prompt("Port to bind to", default=7453)
        
        create_app(config_path)
        uvicorn.run(app, host=host, port=int(port))
     
    except Exception as e:
        click.echo(f"Failed to setup Arbor: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
