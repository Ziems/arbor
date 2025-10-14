"""
Client utilities for running Arbor in interactive environments.

This module provides Ray-like functionality for starting and managing Arbor
servers in notebook environments (Jupyter, Colab, etc.) and other interactive
Python sessions. It includes convenience functions for server management,
job monitoring, and OpenAI client integration.
"""

import atexit
import importlib.util
import os
import socket
import time
from contextlib import closing
from typing import Any, Dict, Optional

from arbor.cli import start_server, stop_server
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)

# Global server instance
_arbor_server = None
_server_thread = None
_server_config = {}


def is_colab_environment() -> bool:
    """Check if running in Google Colab environment."""
    return importlib.util.find_spec("google.colab") is not None


def is_notebook_environment() -> bool:
    """Check if running in any notebook environment (Jupyter, Colab, etc.)."""
    try:
        from IPython import get_ipython

        ipython = get_ipython()
        return ipython is not None and hasattr(ipython, "kernel")
    except ImportError:
        return False


def is_port_available(port: int) -> bool:
    """Check if a port is available for binding."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        try:
            sock.bind(("localhost", port))
            return True
        except socket.error:
            return False


def find_available_port(start_port: int = 7453, max_attempts: int = 100) -> int:
    """Find an available port starting from start_port."""
    for port in range(start_port, start_port + max_attempts):
        if is_port_available(port):
            return port
    raise RuntimeError(
        f"No available ports found in range {start_port}-{start_port + max_attempts}"
    )


def init(
    host: str = "127.0.0.1",
    port: Optional[int] = None,
    storage_path: Optional[str] = None,
    auto_config: bool = True,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Initialize Arbor server in background (Ray-like interface).

    Args:
        host: Host to bind server to (default: "127.0.0.1" for security)
        port: Port to bind to (default: auto-find starting from 7453)
        storage_path: Storage path for Arbor data (default: /content/.arbor in Colab)
        auto_config: Automatically create config if needed (default: True)
        silent: Suppress startup messages (default: False)

    Returns:
        Dict containing server info (host, port, storage_path, etc.)

    Example:
        >>> import arbor
        >>> arbor.init()
        {'host': '127.0.0.1', 'port': 7453, 'storage_path': '/content/.arbor/storage'}

        >>> # Use with OpenAI client
        >>> from openai import OpenAI
        >>> client = OpenAI(base_url="http://127.0.0.1:7453/v1/", api_key="not-needed")
    """
    global _arbor_server, _server_thread, _server_config

    # Check if already initialized
    if _arbor_server is not None:
        if not silent:
            print(
                f"Arbor already running on {_server_config['host']}:{_server_config['port']}"
            )
        return _server_config.copy()

    # Environment detection
    in_colab = is_colab_environment()

    # Auto-configure defaults based on environment
    if port is None:
        port = find_available_port()

    if storage_path is None:
        if in_colab:
            storage_path = "/content/.arbor"
        else:
            storage_path = os.path.expanduser("~/.arbor")

    # Ensure storage directory exists
    os.makedirs(storage_path, exist_ok=True)

    # Create config file if auto_config is enabled
    if auto_config:
        config_path = os.path.join(storage_path, "config.yaml")
        if not os.path.exists(config_path):
            config_content = f"""
storage_path: {os.path.join(storage_path, "storage")}
"""
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, "w") as f:
                f.write(config_content.strip())
            if not silent:
                print(f"Created config at {config_path}")

    try:
        # Start server in background thread
        _arbor_server = start_server(
            host=host,
            port=port,
            storage_path=(
                os.path.join(storage_path, "config.yaml")
                if auto_config
                else storage_path
            ),
            timeout=30,
        )

        # Store config
        _server_config = {
            "host": host,
            "port": port,
            "storage_path": os.path.join(storage_path, "storage"),
            "config_path": (
                os.path.join(storage_path, "config.yaml") if auto_config else None
            ),
            "base_url": f"http://{host}:{port}/v1/",
        }

        if not silent:
            print(f"🌳 Arbor server started on {host}:{port}")
            print(f"📁 Storage: {_server_config['storage_path']}")
            print(f"🔗 Base URL: {_server_config['base_url']}")
            if in_colab:
                print("\n💡 Usage in Colab:")
                print("   from openai import OpenAI")
                print(
                    f"   client = OpenAI(base_url='{_server_config['base_url']}', api_key='not-needed')"
                )

        return _server_config.copy()

    except Exception as e:
        logger.error(f"Failed to start Arbor server: {e}")
        raise RuntimeError(f"Failed to start Arbor server: {e}")


def shutdown():
    """
    Shutdown the Arbor server.

    Example:
        >>> arbor.shutdown()
        Arbor server shutdown complete
    """
    global _arbor_server, _server_thread, _server_config

    if _arbor_server is None:
        print("No Arbor server running")
        return

    try:
        stop_server(_arbor_server)
        _arbor_server = None
        _server_thread = None
        print("🌳 Arbor server shutdown complete")
        _server_config = {}
    except Exception as e:
        logger.error(f"Error shutting down server: {e}")
        print(f"Error shutting down server: {e}")


def status() -> Optional[Dict[str, Any]]:
    """
    Get current Arbor server status.

    Returns:
        Server config dict if running, None if not running

    Example:
        >>> arbor.status()
        {'host': '127.0.0.1', 'port': 7453, 'base_url': 'http://127.0.0.1:7453/v1/'}
    """
    if _arbor_server is None:
        return None
    return _server_config.copy()


def get_client():
    """
    Get a pre-configured OpenAI client for the running Arbor server.

    Returns:
        OpenAI client instance

    Raises:
        RuntimeError: If server is not running

    Example:
        >>> arbor.init()
        >>> client = arbor.get_client()
        >>> client.models.list()
    """
    if _arbor_server is None:
        raise RuntimeError("Arbor server not running. Call arbor.init() first.")

    try:
        from openai import OpenAI

        return OpenAI(base_url=_server_config["base_url"], api_key="not-needed")
    except ImportError:
        raise ImportError("OpenAI package required. Install with: pip install openai")


def shutdown_job(job_identifier: str):
    """
    Shutdown a job by job ID (for training jobs) or model name (for inference jobs).

    Args:
        job_identifier: Either a job ID (for training jobs) or model name (for inference jobs)

    Examples:
        >>> # Shutdown a training job
        >>> arbor.shutdown_job("ftjob:gpt-3.5:training:20241201")

        >>> # Shutdown inference for a specific model
        >>> arbor.shutdown_job("HuggingFaceTB/SmolLM2-135M-Instruct")
    """
    if _arbor_server is None:
        raise RuntimeError("Arbor server not running. Call arbor.init() first.")

    try:
        import requests
        from openai import OpenAI

        base_url = _server_config["base_url"]
        client = OpenAI(base_url=base_url, api_key="not-needed")

        # Try to determine if this is a job ID or model name
        if job_identifier.startswith(("ftjob:", "grpo:")):
            # This looks like a job ID - try to cancel the training job
            try:
                response = client.fine_tuning.jobs.cancel(job_identifier)
                print(f"🛑 Cancelled training job: {job_identifier}")
                print(f"   Status: {response.status}")
                print(
                    "   Associated inference servers and GPU resources have been freed"
                )
                return {
                    "type": "training_job",
                    "status": "cancelled",
                    "job_id": job_identifier,
                }
            except Exception as e:
                print(f"❌ Failed to cancel training job {job_identifier}: {e}")
                raise
        else:
            # This looks like a model name - shutdown inference for this model
            # For now, we'll shutdown all inference servers since there's no per-model endpoint
            try:
                response = requests.post(f"{base_url}chat/kill")
                if response.status_code == 200:
                    print("🛑 Shut down inference servers")
                    print(f"   Model: {job_identifier}")
                    print("   GPU resources have been freed")
                    return {
                        "type": "inference_job",
                        "status": "terminated",
                        "model": job_identifier,
                    }
                else:
                    print(
                        f"❌ Failed to shutdown inference servers: {response.status_code}"
                    )
                    print(f"   Response: {response.text}")
                    raise RuntimeError(
                        f"Failed to shutdown inference: {response.status_code}"
                    )
            except requests.exceptions.RequestException as e:
                print(
                    f"❌ Failed to shutdown inference for model {job_identifier}: {e}"
                )
                raise

    except Exception as e:
        logger.error(f"Failed to shutdown job {job_identifier}: {e}")
        raise RuntimeError(f"Failed to shutdown job {job_identifier}: {e}")


def watch_job(
    job_id: str, max_time: int = 1800, update_interval: int = 10, show_logs: bool = True
):
    """
    Watch a training job with real-time progress updates.

    Args:
        job_id: ID of the job to watch
        max_time: Maximum time to watch in seconds (default: 30 min)
        update_interval: How often to check for updates in seconds
        show_logs: Whether to show training logs in real-time

    Example:
        >>> job = client.fine_tuning.jobs.create(...)
        >>> arbor.watch_job(job.id)
    """
    if _arbor_server is None:
        raise RuntimeError("Arbor server not running. Call arbor.init() first.")

    try:
        from datetime import datetime

        import requests

        base_url = _server_config["base_url"]
        session = requests.Session()

        print(f"👁️  Watching job {job_id}")
        print(f"⏰ Max watch time: {max_time//60} minutes")
        print("=" * 60)

        start_time = time.time()
        last_event_count = 0
        last_status = None

        while (time.time() - start_time) < max_time:
            try:
                # Get job status
                status_resp = session.get(f"{base_url}/fine_tuning/jobs/{job_id}")
                status_resp.raise_for_status()
                status_data = status_resp.json()
                current_status = status_data.get("status", "unknown")

                # Show status changes
                if current_status != last_status:
                    timestamp = datetime.now().strftime("%H:%M:%S")
                    status_icon = {
                        "queued": "⏳",
                        "running": "🏃",
                        "succeeded": "✅",
                        "failed": "❌",
                        "cancelled": "🛑",
                    }.get(current_status, "📊")
                    print(
                        f"[{timestamp}] {status_icon} Status: {current_status.upper()}"
                    )
                    last_status = current_status

                # Show logs if enabled
                if show_logs:
                    events_resp = session.get(
                        f"{base_url}/fine_tuning/jobs/{job_id}/events"
                    )
                    events_resp.raise_for_status()
                    events = events_resp.json()["data"]

                    # Show new log events
                    new_events = events[last_event_count:]
                    for event in new_events:
                        # Only show training logs, skip other events
                        if event.get("data", {}).get("source") == "training_log":
                            timestamp = datetime.now().strftime("%H:%M:%S")
                            print(f"[{timestamp}] 🔧 {event['message']}")

                    last_event_count = len(events)

                # Check if finished
                if current_status in ["succeeded", "failed", "cancelled"]:
                    final_icon = {"succeeded": "🎉", "failed": "💥", "cancelled": "🛑"}[
                        current_status
                    ]
                    print(f"\n{final_icon} Job {current_status.upper()}!")

                    if current_status == "succeeded" and status_data.get(
                        "fine_tuned_model"
                    ):
                        print(f"🎯 Fine-tuned model: {status_data['fine_tuned_model']}")

                    break

                time.sleep(update_interval)

            except Exception as e:
                print(f"⚠️  Error checking job status: {e}")
                time.sleep(update_interval)

        else:
            print(f"\n⏰ Stopped watching after {max_time//60} minutes")

    except ImportError:
        raise ImportError("requests package required for job watching")


# Convenience alias for Ray-like interface
def start(*args, **kwargs):
    """Alias for init() - Ray-like interface."""
    return init(*args, **kwargs)


def stop():
    """Alias for shutdown() - Ray-like interface."""
    return shutdown()


# Auto-cleanup on exit (best effort)
atexit.register(shutdown)
