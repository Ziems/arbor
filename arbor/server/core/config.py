from pathlib import Path
from typing import Dict, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Simplified Arbor configuration."""

    # Basic settings
    storage_path: str = str(Path.home() / ".arbor" / "storage")
    max_context_length: Optional[int] = None

    # Training settings
    accelerate_config: Optional[str] = None

    # Server settings
    inactivity_timeout: int = 30

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> "Config":
        """
        Load config from YAML file or use defaults.

        Args:
            config_path: Path to YAML config file. If None, looks for ~/.arbor/config.yaml

        Returns:
            Config instance
        """
        # Try to find config file
        if config_path is None:
            config_path = str(Path.home() / ".arbor" / "config.yaml")

        # If config file exists, load it
        if Path(config_path).exists():
            try:
                with open(config_path, "r") as f:
                    data = yaml.safe_load(f) or {}
                config = cls(**data)
                config._ensure_storage_path()
                return config
            except Exception as e:
                # If config file is invalid, use defaults and warn
                print(f"Warning: Invalid config file {config_path}: {e}")
                print("Using default configuration")

        # Use defaults
        config = cls()

        config._ensure_storage_path()
        return config

    def _ensure_storage_path(self):
        """Ensure storage path directories exist."""
        storage_dir = Path(self.storage_path)
        storage_dir.mkdir(parents=True, exist_ok=True)
        (storage_dir / "uploads").mkdir(exist_ok=True)

    def get_system_versions(self) -> Dict[str, any]:
        """Get system version information."""
        import platform
        import sys

        versions = {
            "python": {
                "version": sys.version.split()[0],
                "implementation": platform.python_implementation(),
            },
            "system": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "processor": platform.processor(),
            },
        }

        # Try to get package versions for key dependencies
        try:
            import torch

            versions["pytorch"] = torch.__version__
        except ImportError:
            pass

        try:
            import transformers

            versions["transformers"] = transformers.__version__
        except ImportError:
            pass

        try:
            import accelerate

            versions["accelerate"] = accelerate.__version__
        except ImportError:
            pass

        try:
            import pydantic

            versions["pydantic"] = pydantic.__version__
        except ImportError:
            pass

        try:
            import fastapi

            versions["fastapi"] = fastapi.__version__
        except ImportError:
            pass

        return versions
