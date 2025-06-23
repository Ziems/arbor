import platform
from datetime import datetime
from typing import Any, Dict

import psutil

from arbor.server.core.config import Config

try:
    import GPUtil

    GPU_AVAILABLE = True
except ImportError:
    GPU_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class HealthManager:
    """Manages system health checks including GPU monitoring."""

    def __init__(self, config: Config = None):
        self.config = config

    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information including available and used GPUs."""
        gpu_info = {
            "gpus_available": 0,
            "gpus_used": 0,
            "gpu_details": [],
            "cuda_available": False,
            "gpu_library": "none",
        }

        # Try GPUtil first
        if GPU_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                gpu_info["gpus_available"] = len(gpus)
                gpu_info["gpu_library"] = "GPUtil"

                for i, gpu in enumerate(gpus):
                    gpu_detail = {
                        "id": gpu.id,
                        "name": gpu.name,
                        "memory_total": f"{gpu.memoryTotal}MB",
                        "memory_used": f"{gpu.memoryUsed}MB",
                        "memory_free": f"{gpu.memoryFree}MB",
                        "utilization": f"{gpu.load * 100:.1f}%",
                        "temperature": f"{gpu.temperature}°C",
                    }
                    gpu_info["gpu_details"].append(gpu_detail)

                    # Consider GPU "used" if utilization > 10% or memory usage > 10%
                    if gpu.load > 0.1 or (gpu.memoryUsed / gpu.memoryTotal) > 0.1:
                        gpu_info["gpus_used"] += 1

            except Exception as e:
                gpu_info["error"] = f"GPUtil error: {str(e)}"

        # Try PyTorch as fallback/additional info
        if TORCH_AVAILABLE:
            try:
                gpu_info["cuda_available"] = torch.cuda.is_available()
                if torch.cuda.is_available():
                    cuda_count = torch.cuda.device_count()
                    if not GPU_AVAILABLE:  # Only use torch info if GPUtil not available
                        gpu_info["gpus_available"] = cuda_count
                        gpu_info["gpu_library"] = "PyTorch"

                        for i in range(cuda_count):
                            props = torch.cuda.get_device_properties(i)
                            memory_allocated = (
                                torch.cuda.memory_allocated(i) / 1024**2
                            )  # MB
                            memory_cached = (
                                torch.cuda.memory_reserved(i) / 1024**2
                            )  # MB
                            memory_total = props.total_memory / 1024**2  # MB

                            gpu_detail = {
                                "id": i,
                                "name": props.name,
                                "memory_total": f"{memory_total:.0f}MB",
                                "memory_allocated": f"{memory_allocated:.0f}MB",
                                "memory_cached": f"{memory_cached:.0f}MB",
                                "compute_capability": f"{props.major}.{props.minor}",
                            }
                            gpu_info["gpu_details"].append(gpu_detail)

                            # Consider GPU "used" if memory allocated > 100MB
                            if memory_allocated > 100:
                                gpu_info["gpus_used"] += 1

            except Exception as e:
                gpu_info["torch_error"] = f"PyTorch error: {str(e)}"

        return gpu_info

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information including CPU, memory, and disk usage."""
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage("/")
        cpu_percent = psutil.cpu_percent(interval=1)

        return {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "cpu_usage": f"{cpu_percent}%",
            "memory": {
                "total": f"{memory.total / 1024**3:.2f}GB",
                "available": f"{memory.available / 1024**3:.2f}GB",
                "used": f"{memory.used / 1024**3:.2f}GB",
                "percentage": f"{memory.percent}%",
            },
            "disk": {
                "total": f"{disk.total / 1024**3:.2f}GB",
                "free": f"{disk.free / 1024**3:.2f}GB",
                "used": f"{disk.used / 1024**3:.2f}GB",
                "percentage": f"{(disk.used / disk.total) * 100:.1f}%",
            },
            "gpu": self.get_gpu_info(),
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status including system and GPU information."""
        version = self.config.get_arbor_version() if self.config else "unknown"
        versions = (
            self.config.get_system_versions() if self.config else {"arbor": version}
        )

        return {
            "status": "healthy",
            "version": version,  # Keep for backward compatibility
            "versions": versions,  # Comprehensive version info
            "timestamp": datetime.now().isoformat(),
            "system": self.get_system_info(),
        }

    def is_healthy(self) -> bool:
        """Check if the system is healthy based on various metrics."""
        try:
            # Check memory usage (unhealthy if > 90%)
            memory = psutil.virtual_memory()
            if memory.percent > 90:
                print(f"Memory usage is {memory.percent}%")
                return False

            # Check disk usage (unhealthy if > 95%)
            disk = psutil.disk_usage("/")
            if (disk.used / disk.total) * 100 > 95:
                print(f"Disk usage is {disk.used / disk.total * 100}%")
                return False

            # Check CPU usage (unhealthy if > 95% sustained)
            cpu_percent = psutil.cpu_percent(interval=2)
            if cpu_percent > 95:
                print(f"CPU usage is {cpu_percent}%")
                return False

            return True
        except Exception:
            return False
