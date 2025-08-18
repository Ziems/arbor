"""
Arbor - A framework for fine-tuning and managing language models
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("arbor-ai")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "dev"
except Exception:
    __version__ = "unknown"

# Client utilities (Ray-like interface for notebooks and interactive environments)
from arbor.client import (
    get_client,
    init,
    shutdown,
    shutdown_job,
    start,
    status,
    stop,
    watch_job,
)

__all__ = [
    "__version__",
    "init",
    "shutdown",
    "shutdown_job",
    "status",
    "get_client",
    "start",
    "stop",
    "watch_job",
]
