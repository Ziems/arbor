from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, ConfigDict


class InferenceConfig(BaseModel):
    gpu_ids: str = "0"


class TrainingConfig(BaseModel):
    gpu_ids: str = "0"
    accelerate_config: Optional[str] = None


class ArborConfig(BaseModel):
    inference: InferenceConfig
    training: TrainingConfig


class Settings(BaseModel):

    STORAGE_PATH: str = "./storage"
    INACTIVITY_TIMEOUT: int = 30  # 5 seconds
    arbor_config: ArborConfig

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Settings":
        if not yaml_path:
            raise ValueError("Config file path is required")
        if not Path(yaml_path).exists():
            raise ValueError(f"Config file {yaml_path} does not exist")

        try:
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)

            settings = cls(
                arbor_config=ArborConfig(
                    inference=InferenceConfig(**config["inference"]),
                    training=TrainingConfig(**config["training"]),
                )
            )
            return settings
        except Exception as e:
            raise ValueError(f"Error loading config file {yaml_path}: {e}")
