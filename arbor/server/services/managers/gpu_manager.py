"""Simple GPU Manager for tracking GPU allocations and availability."""

import threading
from typing import Dict, List, Optional, Set

from arbor.server.core.config import Config
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.utils.logging import get_logger


class NoGPUsDetectedError(RuntimeError):
    """Raised when no GPUs are detected on the host system."""


class NoGPUAvailableError(RuntimeError):
    """Raised when no free GPUs are available for allocation."""


LOGGER = get_logger(__name__)


def _detect_gpus_with_nvml() -> Dict[str, Set[int]]:
    """Return GPU availability information using NVML."""

    try:
        import pynvml
    except ImportError as exc:
        raise NoGPUsDetectedError(
            "GPU detection requires NVML support (install the pynvml package)."
        ) from exc

    try:
        pynvml.nvmlInit()
    except pynvml.NVMLError as exc:  # type: ignore[attr-defined]
        raise NoGPUsDetectedError(
            "Failed to initialize NVML for GPU detection."
        ) from exc

    try:
        total: Set[int] = set()
        free: Set[int] = set()
        busy: Set[int] = set()

        device_count = pynvml.nvmlDeviceGetCount()

        try:
            compute_proc_getter = pynvml.nvmlDeviceGetComputeRunningProcesses_v3
        except AttributeError:
            try:
                compute_proc_getter = pynvml.nvmlDeviceGetComputeRunningProcesses
            except AttributeError:
                compute_proc_getter = None

        try:
            graphics_proc_getter = pynvml.nvmlDeviceGetGraphicsRunningProcesses_v3
        except AttributeError:
            try:
                graphics_proc_getter = pynvml.nvmlDeviceGetGraphicsRunningProcesses
            except AttributeError:
                graphics_proc_getter = None

        getters = [
            getter for getter in (compute_proc_getter, graphics_proc_getter) if getter
        ]

        for idx in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            total.add(idx)

            running = []
            for getter in getters:
                try:
                    running.extend(getter(handle))
                except pynvml.NVMLError:
                    # Some devices/drivers might not support one of the calls.
                    continue

            if not running or all(
                getattr(proc, "usedGpuMemory", 0) in (None, 0) for proc in running
            ):
                free.add(idx)
            else:
                busy.add(idx)

        return {"total": total, "free": free, "busy": busy}
    except pynvml.NVMLError as exc:
        raise NoGPUsDetectedError("Failed to query GPU state via NVML.") from exc
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def get_gpu_state() -> Dict[str, List[int]]:
    """Return a snapshot of total, free, and busy GPU IDs."""

    info = _detect_gpus_with_nvml()

    total = sorted(info["total"])
    if not total:
        raise NoGPUsDetectedError("No GPUs detected on this host.")

    return {
        "total": total,
        "free": sorted(info["free"]),
        "busy": sorted(info["busy"]),
    }


def detect_all_gpu_ids() -> List[int]:
    """Detect every GPU ID visible on the system."""

    state = get_gpu_state()
    gpu_ids = state["total"]
    LOGGER.info("Detected %d GPU(s): %s", len(gpu_ids), gpu_ids)
    return gpu_ids


def detect_available_gpus() -> List[int]:
    """Detect GPUs that are not currently being used by other processes."""

    state = get_gpu_state()
    free_gpu_ids = state["free"]
    busy_gpu_ids = state["busy"]

    LOGGER.info(
        "Auto-detected %d available GPU(s): %s", len(free_gpu_ids), free_gpu_ids
    )
    if busy_gpu_ids:
        LOGGER.info("GPUs currently in use: %s", busy_gpu_ids)

    return free_gpu_ids


class GPUAllocationError(Exception):
    """Raised when requested GPUs are not available."""

    pass


class GPUManager(BaseManager):
    """
    Simple GPU manager for tracking which GPUs are allocated to which jobs.

    Prevents multiple jobs from using the same GPUs simultaneously.
    """

    def __init__(self, config: Config):
        super().__init__(config)
        self.logger = LOGGER

        # Thread safety
        self._lock = threading.Lock()

        # Track GPU allocations even if initialization fails mid-way
        self.gpu_allocations: Dict[str, List[int]] = {}  # job_id -> [gpu_ids]
        self.all_gpus: Set[int] = set()

        try:
            state = get_gpu_state()
        except NoGPUsDetectedError as exc:
            raise GPUAllocationError(str(exc)) from exc

        if not state["free"]:
            busy_gpus = state["busy"]
            raise GPUAllocationError(
                "No GPUs available on this host."
                if not busy_gpus
                else f"All GPUs are currently in use: {busy_gpus}"
            )

        self.all_gpus = set(state["total"])

        detected_gpus = sorted(self.all_gpus)
        self.logger.info(
            f"GPUManager initialized with GPUs: {detected_gpus}",
            context={"gpus": detected_gpus},
        )
        if state["busy"]:
            busy_gpus = state["busy"]
            self.logger.info(
                f"GPUs currently in use at startup: {busy_gpus}",
                context={"busy_gpus": busy_gpus},
            )

    def get_all_allocated_gpus(self) -> Set[int]:
        """Get set of all currently allocated GPUs across all jobs."""
        allocated = set()
        for gpus in self.gpu_allocations.values():
            allocated.update(gpus)
        return allocated

    def allocate_gpus(self, job_id: str, num_gpus: int) -> List[int]:
        """
        Allocate GPUs to a job.

        Args:
            job_id: Unique identifier for the job
            num_gpus: Number of GPUs to allocate

        Returns:
            List of allocated GPU IDs

        Raises:
            GPUAllocationError: If not enough GPUs are available
        """
        with self._lock:
            # GPUs currently allocated by this manager
            allocated_gpus = set()
            for gpus in self.gpu_allocations.values():
                allocated_gpus.update(gpus)

            state = get_gpu_state()
            system_free_gpus = set(state["free"])
            system_total_gpus = set(state["total"]) if state["total"] else set()

            if not self.all_gpus:
                # If we didn't detect anything at startup, fall back to system view now.
                self.all_gpus = system_total_gpus or system_free_gpus

            candidate_pool = self.all_gpus if self.all_gpus else system_total_gpus
            available_pool = (candidate_pool & system_free_gpus) - allocated_gpus

            if len(available_pool) < num_gpus:
                busy_gpus = sorted(
                    (candidate_pool - system_free_gpus) | set(state["busy"])
                )
                raise GPUAllocationError(
                    "Not enough free GPUs. "
                    f"Requested: {num_gpus}, "
                    f"Free: {sorted(list(available_pool))}, "
                    f"Busy: {busy_gpus}, "
                    f"Allocated: {sorted(allocated_gpus)}"
                )

            # Allocate the first N available GPUs
            allocated = sorted(list(available_pool))[:num_gpus]
            self.gpu_allocations[job_id] = allocated

            self.logger.info(f"Allocated GPUs {allocated} to job {job_id}")
            return allocated

    def get_allocated_gpus(self, job_id: str) -> Optional[List[int]]:
        """Get the GPUs allocated to a specific job."""
        with self._lock:
            return self.gpu_allocations.get(job_id)

    def release_gpus(self, job_id: str) -> bool:
        """
        Release GPUs allocated to a job.

        Args:
            job_id: The job to release GPUs for

        Returns:
            True if GPUs were released, False if job had no allocation
        """
        with self._lock:
            if job_id in self.gpu_allocations:
                released_gpus = self.gpu_allocations[job_id]
                del self.gpu_allocations[job_id]
                self.logger.info(f"Released GPUs {released_gpus} from job {job_id}")
                return True
            return False

    def get_status(self) -> Dict:
        """Get current GPU allocation status."""
        with self._lock:
            allocated_gpus = self.get_all_allocated_gpus()
            state = get_gpu_state()
            system_free_gpus = set(state["free"])
            if not self.all_gpus:
                self.all_gpus = set(state["total"]) or system_free_gpus

            free_gpus = (self.all_gpus & system_free_gpus) - allocated_gpus

            return {
                "total_gpus": sorted(list(self.all_gpus)),
                "free_gpus": sorted(list(free_gpus)),
                "allocated_gpus": sorted(list(allocated_gpus)),
                "allocations": dict(self.gpu_allocations),
            }

    def cleanup(self) -> None:
        """Clean up all GPU allocations."""
        if self._cleanup_called:
            return

        with self._lock:
            allocation_count = len(self.gpu_allocations)
            if allocation_count > 0:
                self.logger.info(f"Cleaning up {allocation_count} GPU allocations...")
                self.gpu_allocations.clear()
                self.logger.info("All GPU allocations released")

        super().cleanup()
