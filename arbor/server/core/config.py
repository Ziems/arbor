import datetime
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, ClassVar, Dict, Optional

import yaml
from pydantic import BaseModel

try:
    from importlib.metadata import PackageNotFoundError, version
except ImportError:
    # For Python < 3.8
    from importlib_metadata import PackageNotFoundError, version


class InferenceConfig(BaseModel):
    gpu_ids: str = "0"


class TrainingConfig(BaseModel):
    gpu_ids: str = "0"
    accelerate_config: Optional[str] = None


class ArborConfig(BaseModel):
    inference: InferenceConfig
    training: TrainingConfig


class Config(BaseModel):
    STORAGE_PATH: ClassVar[str] = str(Path.home() / ".arbor" / "storage")
    INACTIVITY_TIMEOUT: int = 30  # 5 seconds
    arbor_config: ArborConfig

    @staticmethod
    def validate_storage_path(storage_path: str):
        """Validates a storage path, return True for success, False if failed."""
        try:
            if not Path(storage_path).exists():
                return False
            return True

        except Exception as e:
            return False

    @classmethod
    def set_storage_path(cls, storage_path: str):
        """Set a valid storage path to use, return True for success, False if failed."""
        if not cls.validate_storage_path(storage_path):
            return False

        cls.STORAGE_PATH = storage_path

        return True

    @staticmethod
    def validate_storage_path(storage_path: str) -> None:
        """Validates a storage path, raises exception if invalid."""
        if not storage_path:
            raise ValueError("Storage path cannot be empty")

        path = Path(storage_path)

        if not path.exists():
            raise FileNotFoundError(f"Storage path does not exist: {storage_path}")

        if not path.is_dir():
            raise NotADirectoryError(f"Storage path is not a directory: {storage_path}")

        # Check if we can write to the directory
        if not os.access(path, os.W_OK):
            raise PermissionError(
                f"No write permission for storage path: {storage_path}"
            )

    @classmethod
    def set_storage_path(cls, storage_path: str) -> None:
        """Set a valid storage path to use, raises exception if invalid."""
        cls.validate_storage_path(storage_path)  # raises if invalid
        cls.STORAGE_PATH = storage_path

    @classmethod
    def make_log_dir(cls, storage_path: str = None):
        """Create a timestamped log directory under the storage path."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        log_dir = Path(
            storage_path if storage_path else cls.STORAGE_PATH / "logs" / timestamp
        )
        log_dir.mkdir(exist_ok=True)

        return log_dir

    @staticmethod
    def get_arbor_version() -> str:
        """Get the installed version of arbor package."""
        try:
            return version("arbor-ai")
        except PackageNotFoundError:
            # Fallback to a default version if package not found
            # This might happen in development mode
            return "dev"
        except Exception:
            return "unknown"

    @staticmethod
    def get_cuda_version() -> str:
        """Get CUDA runtime version."""
        try:
            import torch

            if torch.cuda.is_available():
                return torch.version.cuda
            else:
                return "not_available"
        except ImportError:
            try:
                # Try getting CUDA version from nvcc
                result = subprocess.run(
                    ["nvcc", "--version"], capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse nvcc output for version
                    for line in result.stdout.split("\n"):
                        if "release" in line.lower():
                            # Extract version from line like "Cuda compilation tools, release 11.8, V11.8.89"
                            parts = line.split("release")
                            if len(parts) > 1:
                                version_part = parts[1].split(",")[0].strip()
                                return version_part
                return "unknown"
            except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
                return "not_installed"
        except Exception:
            return "unknown"

    @staticmethod
    def get_nvidia_driver_version() -> str:
        """Get NVIDIA driver version."""
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=driver_version",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip().split("\n")[0]
            return "unknown"
        except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
            return "not_installed"

    @staticmethod
    def get_python_package_version(package_name: str) -> str:
        """Get version of a Python package."""
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "not_installed"
        except Exception:
            return "unknown"

    @classmethod
    def get_ml_library_versions(cls) -> Dict[str, str]:
        """Get versions of common ML libraries."""
        libraries = {
            "torch": "torch",
            "transformers": "transformers",
            "vllm": "vllm",
            "trl": "trl",
            "peft": "peft",
            "accelerate": "accelerate",
            "ray": "ray",
            "wandb": "wandb",
            "numpy": "numpy",
            "pandas": "pandas",
            "scikit-learn": "scikit-learn",
        }

        versions = {}
        for lib_name, package_name in libraries.items():
            versions[lib_name] = cls.get_python_package_version(package_name)

        return versions

    @classmethod
    def get_cuda_library_versions(cls) -> Dict[str, str]:
        """Get versions of CUDA-related libraries."""
        cuda_info = {}

        # CUDA runtime version
        cuda_info["cuda_runtime"] = cls.get_cuda_version()

        # NVIDIA driver version
        cuda_info["nvidia_driver"] = cls.get_nvidia_driver_version()

        # cuDNN version (if available through PyTorch)
        try:
            import torch

            if torch.cuda.is_available() and hasattr(torch.backends.cudnn, "version"):
                cuda_info["cudnn"] = str(torch.backends.cudnn.version())
            else:
                cuda_info["cudnn"] = "not_available"
        except ImportError:
            cuda_info["cudnn"] = "torch_not_installed"
        except Exception:
            cuda_info["cudnn"] = "unknown"

        # NCCL version (if available through PyTorch)
        try:
            import torch

            if torch.cuda.is_available() and hasattr(torch, "__version__"):
                # NCCL version is often embedded in PyTorch build info
                try:
                    import torch.distributed as dist

                    if hasattr(dist, "is_nccl_available") and dist.is_nccl_available():
                        # Try to get NCCL version from PyTorch
                        if hasattr(torch.cuda.nccl, "version"):
                            nccl_version = torch.cuda.nccl.version()
                            cuda_info["nccl"] = (
                                f"{nccl_version[0]}.{nccl_version[1]}.{nccl_version[2]}"
                            )
                        else:
                            cuda_info["nccl"] = "available"
                    else:
                        cuda_info["nccl"] = "not_available"
                except Exception:
                    cuda_info["nccl"] = "unknown"
            else:
                cuda_info["nccl"] = "cuda_not_available"
        except ImportError:
            cuda_info["nccl"] = "torch_not_installed"
        except Exception:
            cuda_info["nccl"] = "unknown"

        return cuda_info

    @classmethod
    def get_system_versions(cls) -> Dict[str, Any]:
        """Get comprehensive version information for the system."""
        return {
            "arbor": cls.get_arbor_version(),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ml_libraries": cls.get_ml_library_versions(),
            "cuda_stack": cls.get_cuda_library_versions(),
        }

    @classmethod
    def _init_arbor_directories(cls):
        arbor_root = Path.home() / ".arbor"
        storage_dir = Path(cls.STORAGE_PATH)

        arbor_root.mkdir(exist_ok=True)
        storage_dir.mkdir(exist_ok=True)
        (storage_dir / "logs").mkdir(exist_ok=True)
        (storage_dir / "models").mkdir(exist_ok=True)
        (storage_dir / "uploads").mkdir(exist_ok=True)

    @classmethod
    def use_default_config(cls) -> Optional[str]:
        """Search for: ~/.arbor/config.yaml, else return None"""

        # Check ~/.arbor/config.yaml
        arbor_config = Path.home() / ".arbor" / "config.yaml"
        if arbor_config.exists():
            return str(arbor_config)

        return None

    @classmethod
    def load_config_from_yaml(cls, yaml_path: str) -> "Config":
        # If yaml file is not provided, try to use ~/.arbor/config.yaml
        cls._init_arbor_directories()

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

            storage_path = config.get("storage_path")
            if storage_path:
                cls.set_storage_path(storage_path)

            return settings
        except Exception as e:
            raise ValueError(f"Error loading config file {yaml_path}: {e}")

    @classmethod
    def load_config_directly(
        cls,
        storage_path: str = None,
        inference_gpus: str = "0",
        training_gpus: str = "1,2",
    ):
        cls._init_arbor_directories()

        # create settings without yaml file
        config = ArborConfig(
            inference=InferenceConfig(gpu_ids=inference_gpus),
            training=TrainingConfig(gpu_ids=training_gpus),
        )

        if storage_path:
            cls.set_storage_path(storage_path)

        return cls(arbor_config=config)
