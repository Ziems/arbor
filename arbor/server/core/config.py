from pathlib import Path
from typing import List, Optional

import yaml
from pydantic import BaseModel


class Config(BaseModel):
    """Simplified Arbor configuration."""

    # Basic settings
    storage_path: str = str(Path.home() / ".arbor" / "storage")
    gpu_ids: List[int] = [0, 1, 2]

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
        (storage_dir / "logs").mkdir(exist_ok=True)
        (storage_dir / "models").mkdir(exist_ok=True)
        (storage_dir / "uploads").mkdir(exist_ok=True)
