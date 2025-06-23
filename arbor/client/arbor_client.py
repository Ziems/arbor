import asyncio
import os
import socket
import threading
import time
from datetime import datetime

import click
import requests
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

# Global server state
_server = None
_server_thread = None
_server_loop = None
_server_host = None
_server_port = None


def create_app(
    config_path: str = None,
    storage_path: str = None,
    inference_gpus: str = None,
    training_gpus: str = None,
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
    elif inference_gpus and training_gpus:
        config = Config.load_config_directly(
            storage_path, inference_gpus, training_gpus
        )
    else:
        raise ValueError(
            "Either 'config_path' must be provided, or 'inference_gpus', and 'training_gpus' must be provided"
        )

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


def serve(
    config_path: str = None,
    storage_path: str = None,
    inference_gpus: str = None,
    training_gpus: str = None,
    host: str = "0.0.0.0",
    port: int = 7453,
):
    """Start the Arbor API server.

    Starts the server in a background thread and returns once the server is ready to accept requests.
    Use arbor.stop() to shutdown the server.

    Args:
        config_path: Path to YAML config file (optional)
        storage_path: Valid storage directory path (optional)
        inference_gpus: GPU IDs for inference, e.g. "0,1" (optional, default 0)
        training_gpus: GPU IDs for training, e.g. "1,2,3" (optional, default 1,2)
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 7453)

    Example:
        import arbor
        arbor.serve(inference_gpus="0", training_gpus="1,2")
        # Server is now ready to accept requests
        # Later, to stop:
        arbor.stop()
    """
    global _server, _server_thread, _server_loop, _server_host, _server_port

    # Stop existing server if running
    if _server is not None:
        print("🌳 Stopping existing server...")
        stop()

    _server_host = host
    _server_port = port

    create_app(config_path, storage_path, inference_gpus, training_gpus)

    # Start server in background thread
    def run_server():
        global _server, _server_loop

        # Create a new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _server_loop = loop

        # Create uvicorn config and server
        config = uvicorn.Config(app, host=host, port=port, loop=loop)
        server = uvicorn.Server(config)
        _server = server

        # Run the server
        try:
            loop.run_until_complete(server.serve())
        except Exception as e:
            print(f"Server error: {e}")
        finally:
            loop.close()
            _server = None
            _server_loop = None

    # Start server thread
    _server_thread = threading.Thread(target=run_server, daemon=True)
    _server_thread.start()

    print(f"🌳 Starting Arbor server on http://{host}:{port}...")

    # Wait for server to be ready
    try:
        _wait_for_server_ready(host, port, timeout=60)  # Increased timeout
        print(f"🌳 Arbor server is ready and accepting requests!")
    except TimeoutError as e:
        print(f"❌ {e}")
        # Try to stop the server if it failed to start properly
        stop()
        raise


def stop():
    """Stop the Arbor server if it's running."""
    global _server, _server_thread, _server_loop

    if _server is None:
        print("🌳 No server running to stop.")
        return

    print("🌳 Stopping Arbor server...")

    # Schedule server shutdown in the server's event loop
    if _server_loop and _server:
        try:
            asyncio.run_coroutine_threadsafe(_server.shutdown(), _server_loop)
        except Exception as e:
            print(f"Error during shutdown: {e}")

    # Wait for thread to finish
    if _server_thread and _server_thread.is_alive():
        _server_thread.join(timeout=5)

    # Reset global state
    _server = None
    _server_thread = None
    _server_loop = None

    print("🌳 Arbor server stopped.")


def is_running():
    """Check if the Arbor server is currently running."""
    return (
        _server is not None and _server_thread is not None and _server_thread.is_alive()
    )


def _wait_for_server_ready(host, port, timeout=30):
    """Wait for the server to be ready to accept requests."""
    start_time = time.time()
    last_error = None
    port_open = False

    print(f"🌳 Waiting for server to be ready at http://{host}:{port}...")

    while time.time() - start_time < timeout:
        # First check if the port is open
        if not port_open:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1)
                result = sock.connect_ex((host, port))
                sock.close()
                if result == 0:
                    port_open = True
                    print(f"🌳 Port {port} is now open, checking health endpoint...")
                else:
                    last_error = f"Port {port} not yet open"
                    time.sleep(0.5)
                    continue
            except Exception as e:
                last_error = f"Socket error: {e}"
                time.sleep(0.5)
                continue

        # Now try the health check
        try:
            response = requests.get(f"http://{host}:{port}/health/simple", timeout=2)
            if response.status_code == 200:
                print(f"🌳 Server ready! Response: {response.json()}")
                return
            else:
                last_error = f"Health check returned status {response.status_code}"
        except requests.exceptions.ConnectionError as e:
            last_error = f"Connection error: {e}"
            port_open = False  # Port might have closed
        except requests.exceptions.Timeout as e:
            last_error = f"Timeout error: {e}"
        except requests.exceptions.RequestException as e:
            last_error = f"Request error: {e}"
        except Exception as e:
            last_error = f"Unexpected error: {e}"

        # Print progress every 5 seconds
        elapsed = time.time() - start_time
        if int(elapsed) % 5 == 0 and elapsed >= 5:
            print(
                f"🌳 Still waiting... ({elapsed:.1f}s elapsed, port_open={port_open}, last error: {last_error})"
            )

        time.sleep(0.5)

    raise TimeoutError(
        f"Server did not become ready within {timeout} seconds. Last error: {last_error}"
    )


if __name__ == "__main__":
    serve(inference_gpus="0, 1", training_gpus="2, 3")
