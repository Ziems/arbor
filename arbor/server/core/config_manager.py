from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml

from arbor.server.core.config import Config


class ConfigManager:
    def __init__(self):
        self._init_arbor_directories()

    def _init_arbor_directories(self):
        arbor_root = Path.home() / ".arbor"
        storage_dir = Path.home() / ".arbor" / "storage"  # Use default storage path

        arbor_root.mkdir(exist_ok=True)
        storage_dir.mkdir(exist_ok=True)
        (storage_dir / "logs").mkdir(exist_ok=True)
        (storage_dir / "models").mkdir(exist_ok=True)
        (storage_dir / "uploads").mkdir(exist_ok=True)

    @staticmethod
    def get_default_config_path() -> Path:
        return str(Path.home() / ".arbor" / "config.yaml")

    @staticmethod
    def get_config_template() -> Dict:
        return {"storage_path": str(Path.home() / ".arbor" / "storage")}

    @classmethod
    def update_config(
        cls,
        config_path: Optional[str] = None,
    ) -> str:
        """Update existing config or create new one."""

        if config_path is None:
            config_path = str(cls.get_default_config_path())

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        # Load existing config or use template
        if config_file.exists():
            with open(config_file, "r") as f:
                config = yaml.safe_load(f) or {}
        else:
            config = cls.get_config_template()

        temp_path = config_file.with_suffix(".tmp")
        try:
            with open(temp_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False, default_style="'")
            temp_path.rename(config_file)
        except Exception:
            if temp_path.exists():
                temp_path.unlink()
            raise

        return str(config_file)

    @classmethod
    def validate_config_file(cls, config_path: str) -> Tuple[bool, str]:
        """Validate a config file"""
        try:
            if not Path(config_path).exists():
                return False, f"Config file does not exist: {config_path}"

            # If we do have a config file, try to see if it will load
            Config.load(config_path)
            return True, "Config is valid"

        except Exception as e:
            return False, f"Invalid config: {e}"

    @classmethod
    def get_config_contents(cls, config_path: str) -> Tuple[bool, str]:
        try:
            with open(config_path, "r") as f:
                config_content = f.read()
            return True, config_content
        except Exception as e:
            return False, str(e)
