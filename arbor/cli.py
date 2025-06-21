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
from arbor.client.arbor_client import create_app


@click.group()
def cli():
    pass

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

    if arbor_config:
        config_path = arbor_config
    else:
        config_path = Config.use_default_config()

        # If no config found, run first-time setup
        if config_path is None:
            config_path = run_first_time_setup()

    # Validate config exists and is readable
    is_valid, msg = ConfigManager.validate_config_file(config_path)

    if not is_valid:
        click.echo(msg)
        raise click.Abort()

    try:
        create_app(config_path)
        # Temporarily disable custom uvicorn logging configuration
        # configure_uvicorn_logging()
        uvicorn.run(app, host=host, port=port)
    except Exception as e:
        click.echo(f"Failed to start server: {e}", err=True)
        raise click.Abort()


def run_first_time_setup() -> str:
    """Run first-time setup and return created config path"""
    click.echo("Welcome to Arbor!")
    click.echo("It looks like this is your first time running Arbor.")
    click.echo("Let's set up your configuration...\n")

    try:
        # Get config details
        inference = click.prompt(
            "Which gpu ids should be used for inference (separated by comma)",
            default="0",
        )
        training = click.prompt(
            "Which gpu ids should be used for training (separated by comma)",
            default="1, 2",
        )
        click.echo()

        # Get config file path
        config_path = click.prompt(
            "Enter path to save config file in. We recommend (~/.arbor/config.yaml)",
            default=ConfigManager.get_default_config_path(),
        )
        logger = get_logger(__name__)
        logger.info(f"Config path selected: {config_path}")
        click.echo()

        # Update or create config at path
        config_path = ConfigManager.update_config(inference, training, config_path)
        click.echo(f"Created configuration at: {config_path}")

        # Check if it is a valid config file
        is_valid, msg = ConfigManager.validate_config_file(config_path)
        if not is_valid:
            raise click.ClickException(f"Invalid config file: {msg}")

        # Read and display the contents
        _, content = ConfigManager.get_config_contents(config_path)

        click.echo("\nConfiguration file contents:")
        click.echo("---")
        click.echo(content)
        click.echo("---")

        click.echo("\nSetup complete! Starting Arbor server...")
        return config_path

    except Exception as e:
        click.echo(f"Failed initial setup of Arbor: {e}", err=True)
        raise click.Abort()


if __name__ == "__main__":
    cli()
