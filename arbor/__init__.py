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

# Colab/Jupyter integration (Ray-like interface)
from arbor.colab import get_client, init, shutdown, start, status, stop

__all__ = ["__version__", "init", "shutdown", "status", "get_client", "start", "stop"]
