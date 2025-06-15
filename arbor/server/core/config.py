import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

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


class Settings(BaseModel):

    STORAGE_PATH: str = str(Path.home() / ".arbor" / "storage")
    INACTIVITY_TIMEOUT: int = 30  # 5 seconds
    arbor_config: ArborConfig

    def __init__(self, **data):
        super().__init__(**data)

        # Create ~/.arbor directories if it DNE
        self._init_arbor_directories()

    def get_arbor_version(self) -> str:
        """Get the installed version of arbor package."""
        try:
            return version("arbor-ai")
        except PackageNotFoundError:
            # Fallback to a default version if package not found
            # This might happen in development mode
            return "dev"
        except Exception:
            return "unknown"

    def get_cuda_version(self) -> str:
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

    def get_nvidia_driver_version(self) -> str:
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

    def get_python_package_version(self, package_name: str) -> str:
        """Get version of a Python package."""
        try:
            return version(package_name)
        except PackageNotFoundError:
            return "not_installed"
        except Exception:
            return "unknown"

    def get_ml_library_versions(self) -> Dict[str, str]:
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
            versions[lib_name] = self.get_python_package_version(package_name)

        return versions

    def get_cuda_library_versions(self) -> Dict[str, str]:
        """Get versions of CUDA-related libraries."""
        cuda_info = {}

        # CUDA runtime version
        cuda_info["cuda_runtime"] = self.get_cuda_version()

        # NVIDIA driver version
        cuda_info["nvidia_driver"] = self.get_nvidia_driver_version()

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

    def get_system_versions(self) -> Dict[str, Any]:
        """Get comprehensive version information for the system."""
        return {
            "arbor": self.get_arbor_version(),
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "ml_libraries": self.get_ml_library_versions(),
            "cuda_stack": self.get_cuda_library_versions(),
        }

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
        """Search for: ~/.arbor/config.yaml, else return None"""

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

    @classmethod
    def from_gpus(cls, inference_gpus: str = "0", training_gpus: str = "1,2"):
        # create settings without yaml file
        config = ArborConfig(
            inference=InferenceConfig(gpu_ids=inference_gpus),
            training=TrainingConfig(gpu_ids=training_gpus),
        )
        return cls(arbor_config=config)
