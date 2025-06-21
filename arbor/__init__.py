"""
Arbor - A framework for fine-tuning and managing language models
"""

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8
    from importlib_metadata import PackageNotFoundError, version

try:
    __version__ = version("arbor-ai")
except PackageNotFoundError:
    # Package is not installed, likely in development mode
    __version__ = "dev"
except Exception:
    __version__ = "unknown"
    
from arbor.client.arbor_client import serve
    
__all__ = ["__version__", "serve"]
