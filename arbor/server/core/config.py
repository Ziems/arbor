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

    STORAGE_PATH: str = str(Path.home() / ".arbor" / "storage")
    INACTIVITY_TIMEOUT: int = 30  # 5 seconds
    arbor_config: ArborConfig
    
    def __init__(self, **data):
        super().__init__(**data)
        
        # Create ~/.arbor directories if it DNE
        self._init_arbor_directories()


    def _init_arbor_directories(self):
        arbor_root = Path.home() / ".arbor"
        storage_dir = Path(self.STORAGE_PATH)
        
        arbor_root.mkdir(exist_ok=True)
        storage_dir.mkdir(exist_ok=True)
        (storage_dir / "logs").mkdir(exist_ok=True)
        (storage_dir / "models").mkdir(exist_ok=True)
        (storage_dir / "uploads").mkdir(exist_ok=True)

    @classmethod
    def use_default_config(cls) -> Optional[str]:
        """ Search for: ~/.arbor/config.yaml, else return None"""

        # Check ~/.arbor/config.yaml
        arbor_config = Path.home() / ".arbor" / "config.yaml"
        if arbor_config.exists(): 
            return str(arbor_config)
    
        return None

    @classmethod
    def load_from_yaml(cls, yaml_path: str) -> "Settings":
        # If yaml file is not provided, try to use ~/.arbor/config.yaml 
        if not yaml_path:
            yaml_path = cls.use_default_config()
        
        if not yaml_path:
            raise ValueError(
                "No config file found. Please create ~/.arbor/config.yaml or "
                "provide a config file path with --arbor-config"
            )
            
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



        