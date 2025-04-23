from pydantic import BaseModel, ConfigDict
import yaml
from pathlib import Path

class InferenceConfig(BaseModel):
    gpu_ids: str = "0"

class TrainingConfig(BaseModel):
    gpu_ids: str = "0"
    num_processes: int = 1

class ArborConfig(BaseModel):
    inference: InferenceConfig
    training: TrainingConfig

class Settings(BaseModel):

    STORAGE_PATH: str = "./storage"
    INACTIVITY_TIMEOUT: int = 30 # 5 seconds
    arbor_config: ArborConfig = ArborConfig()

    @classmethod
    def load_from_yaml(cls, yaml_path: str = "arbor.yaml") -> "Settings":

        settings = cls()
        if Path(yaml_path).exists():
            with open(yaml_path, "r") as f:
                config = yaml.safe_load(f)
            settings.inference.gpu_ids = config["inference"]["gpu_ids"]
            settings.training.gpu_ids = config["training"]["gpu_ids"]
            settings.training.num_processes = config["training"]["num_processes"]

        return settings