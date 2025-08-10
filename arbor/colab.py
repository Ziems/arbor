"""
Colab integration utilities for running Arbor in background processes.

This module provides Ray-like functionality for starting and managing Arbor
servers in Google Colab and Jupyter notebook environments.
"""

import os
import socket
import threading
import time
import warnings
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
    try:
        import google.colab

        return True
    except ImportError:
        return False


def is_jupyter_environment() -> bool:
    """Check if running in any Jupyter environment."""
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
    gpu_ids: Optional[list] = None,
    auto_config: bool = True,
    silent: bool = False,
) -> Dict[str, Any]:
    """
    Initialize Arbor server in background (Ray-like interface).

    Args:
        host: Host to bind server to (default: "127.0.0.1" for security)
        port: Port to bind to (default: auto-find starting from 7453)
        storage_path: Storage path for Arbor data (default: /content/.arbor in Colab)
        gpu_ids: List of GPU IDs to use (default: auto-detect available GPUs)
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
        >>> client = OpenAI(base_url="http://127.0.0.1:7453/v1", api_key="not-needed")
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
    in_jupyter = is_jupyter_environment()

    if not in_jupyter and not silent:
        warnings.warn("Arbor.init() is designed for Jupyter/Colab environments")

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

    # Auto-detect GPUs if not specified
    if gpu_ids is None and auto_config:
        try:
            import torch

            if torch.cuda.is_available():
                gpu_ids = list(range(torch.cuda.device_count()))
                if not silent:
                    print(f"Auto-detected {len(gpu_ids)} GPU(s): {gpu_ids}")
            else:
                gpu_ids = []
                if not silent:
                    print("No GPUs detected, using CPU mode")
        except ImportError:
            gpu_ids = []
            if not silent:
                print("PyTorch not available, using CPU mode")

    # Create config file if auto_config is enabled
    if auto_config:
        config_path = os.path.join(storage_path, "config.yaml")
        if not os.path.exists(config_path):
            config_content = f"""
storage_path: {os.path.join(storage_path, "storage")}
inference:
  gpu_ids: {gpu_ids or []}
training:
  gpu_ids: {gpu_ids or []}
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
            "gpu_ids": gpu_ids,
            "base_url": f"http://{host}:{port}/v1",
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
        {'host': '127.0.0.1', 'port': 7453, 'base_url': 'http://127.0.0.1:7453/v1'}
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


# Convenience alias for Ray-like interface
def start(*args, **kwargs):
    """Alias for init() - Ray-like interface."""
    return init(*args, **kwargs)


def stop():
    """Alias for shutdown() - Ray-like interface."""
    return shutdown()


# Auto-cleanup on exit (best effort)
import atexit

atexit.register(shutdown)
