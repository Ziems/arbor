import os
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Callable, List, Literal, Optional

import coolname

from arbor.server.api.models.schemas import JobStatus, JobStatusModel
from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


class JobArtifact(Enum):
    """Enum defining types of artifacts that jobs can produce."""

    LOGS = "logs"
    MODEL = "model"
    CHECKPOINTS = "checkpoints"
    METRICS = "metrics"


class JobEvent:
    def __init__(
        self, level: Literal["info", "warning", "error"], message: str, data: dict = {}
    ):
        self.level = level
        self.message = message
        self.data = data

        self.id = str(f"ftevent-{uuid.uuid4()}")
        self.created_at = datetime.now()


class JobCheckpoint:
    def __init__(
        self,
        fine_tuned_model_checkpoint: str,
        fine_tuning_job_id: str,
        metrics: dict,
        step_number: int,
    ):
        self.id = str(f"ftckpt-{uuid.uuid4()}")
        self.fine_tuned_model_checkpoint = fine_tuned_model_checkpoint
        self.fine_tuning_job_id = fine_tuning_job_id
        self.metrics = metrics
        self.step_number = step_number
        self.created_at = datetime.now()


class Job:
    def __init__(
        self, config, id=None, prefix="ftjob", artifacts: List[JobArtifact] = None
    ):
        self.config = config

        if id is None:
            readable_slug = coolname.generate_slug(2)
            timestamp = datetime.now().strftime("%Y%m%d")
            self.id = str(f"{prefix}:{readable_slug}:{timestamp}")
        else:
            self.id = id
        self.status = JobStatus.CREATED
        self.fine_tuned_model = None
        self.events: list[JobEvent] = []
        self.checkpoints: list[JobCheckpoint] = []
        self.log_file_path: Optional[str] = None

        self.created_at = datetime.now()

        # Default artifacts if none specified - just logs for most jobs
        if artifacts is None:
            artifacts = [JobArtifact.LOGS]

        # Set up directories based on artifacts
        self._setup_directories(artifacts)

    def add_event(self, event: JobEvent):
        self.events.append(event)

    def get_events(self) -> list[JobEvent]:
        return self.events

    def add_checkpoint(self, checkpoint: JobCheckpoint):
        self.checkpoints.append(checkpoint)

    def get_checkpoints(self) -> list[JobCheckpoint]:
        return self.checkpoints

    def create_log_callback(self, job_type: str = "JOB") -> Callable:
        """
        Create a universal log callback that handles:
        1. Standard Python logging (human-readable format)
        2. Log file persistence (structured JSONL format)
        3. Optional job events (only when explicitly requested)

        Args:
            job_type: Type name for log prefixes (e.g., "SFT", "GRPO", "INFERENCE")

        Returns:
            A callback function that accepts (message: str, extra_data: dict = None, create_event: bool = False)
        """
        import json

        def enhanced_log_callback(
            line: str, extra_data: dict = None, create_event: bool = False
        ):
            timestamp = datetime.now()
            timestamp_iso = timestamp.isoformat()

            # 1. Log to standard logger (human-readable format)
            logger.info(f"[{job_type} LOG] {line}")

            # 2. Save to log file as structured JSONL if path is set
            if self.log_file_path:
                try:
                    os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

                    # Create structured log entry with base fields
                    log_entry = {
                        "timestamp": timestamp_iso,
                        "level": "info",
                        "job_id": self.id,
                        "job_type": job_type,
                        "message": line,
                        "source": "training_log",
                    }

                    # Add any extra data provided
                    if extra_data:
                        log_entry.update(extra_data)

                    with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                        log_file.write(json.dumps(log_entry) + "\n")
                        log_file.flush()  # Ensure immediate write
                except Exception as e:
                    logger.error(
                        f"Failed to write to log file {self.log_file_path}: {e}"
                    )

            # 3. Optionally add as job event for API access (only for important milestones)
            if create_event:
                event_data = {"source": "training_log"}
                if extra_data:
                    event_data.update(extra_data)
                event = JobEvent(level="info", message=line, data=event_data)
                self.add_event(event)

        return enhanced_log_callback

    def log(
        self,
        message: str,
        level: str = "info",
        extra_data: dict = None,
        job_type: str = "JOB",
        create_event: bool = False,
    ):
        """
        Unified logging method that handles both detailed logs and important events.

        Args:
            message: The log message
            level: Log level (info, warning, error)
            extra_data: Additional structured data to include
            job_type: Type name for log prefixes
            create_event: Whether to create a JobEvent (for important milestones only)
        """
        import json

        timestamp = datetime.now()
        timestamp_iso = timestamp.isoformat()

        # 1. Log to standard logger
        logger_fn = getattr(logger, level, logger.info)
        logger_fn(f"[{job_type} LOG] {message}")

        # 2. Save to log file as structured JSONL if path is set
        if self.log_file_path:
            try:
                os.makedirs(os.path.dirname(self.log_file_path), exist_ok=True)

                # Create structured log entry
                log_entry = {
                    "timestamp": timestamp_iso,
                    "level": level,
                    "job_id": self.id,
                    "job_type": job_type,
                    "message": message,
                    "source": "direct_log",
                }

                # Add any extra data provided
                if extra_data:
                    log_entry.update(extra_data)

                with open(self.log_file_path, "a", encoding="utf-8") as log_file:
                    log_file.write(json.dumps(log_entry) + "\n")
                    log_file.flush()
            except Exception as e:
                logger.error(f"Failed to write to log file {self.log_file_path}: {e}")

        # 3. Optionally add as job event for API access (only for important milestones)
        if create_event:
            event_data = {"source": "direct_log"}
            if extra_data:
                event_data.update(extra_data)
            event = JobEvent(level=level, message=message, data=event_data)
            self.add_event(event)

    # Keep log_structured as an alias for backward compatibility
    def log_structured(
        self,
        message: str,
        level: str = "info",
        extra_data: dict = None,
        job_type: str = "JOB",
    ):
        """Alias for log() method. Use log() instead."""
        self.log(message, level, extra_data, job_type, create_event=True)

    def _make_log_dir(self) -> str:
        """Create log directory for this job. Can be overridden by subclasses."""
        log_dir = Path(self.config.storage_path).resolve() / self.id / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        return str(log_dir)

    def _make_model_dir(self) -> str:
        """Create model output directory for this job. Can be overridden by subclasses."""
        model_dir = Path(self.config.storage_path).resolve() / self.id / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def _make_checkpoints_dir(self) -> str:
        """Create checkpoints directory for this job. Can be overridden by subclasses."""
        checkpoints_dir = (
            Path(self.config.storage_path).resolve() / self.id / "checkpoints"
        )
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        return str(checkpoints_dir)

    def _make_metrics_dir(self) -> str:
        """Create metrics directory for this job. Can be overridden by subclasses."""
        metrics_dir = Path(self.config.storage_path).resolve() / self.id / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        return str(metrics_dir)

    def _setup_directories(self, artifacts: List[JobArtifact]):
        """Set up directories based on requested artifacts."""
        for artifact in artifacts:
            if artifact == JobArtifact.LOGS:
                log_dir = self._make_log_dir()
                # Set default log file path - subclasses can override the filename
                self.log_file_path = os.path.join(
                    log_dir, f"{self.__class__.__name__.lower()}.log"
                )

            elif artifact == JobArtifact.MODEL:
                # Create model directory - subclasses can use self._make_model_dir() to get the path
                self._make_model_dir()

            elif artifact == JobArtifact.CHECKPOINTS:
                # Create checkpoints directory
                self._make_checkpoints_dir()

            elif artifact == JobArtifact.METRICS:
                # Create metrics directory for storing training metrics, plots, etc.
                self._make_metrics_dir()

    def cancel(self):
        """Cancel the job. Override in subclasses for specific cancellation logic."""
        if self.status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel job with status {self.status.value}")

        self.status = JobStatus.CANCELLED

    def _ensure_gpu_cleanup(self):
        """
        Base GPU cleanup method. Override in subclasses that allocate GPUs.
        This is called automatically when jobs fail or crash.
        """
        pass

    def to_status_model(self) -> JobStatusModel:
        """Convert this job to a JobStatusModel for API responses."""
        return JobStatusModel(
            id=self.id,
            status=self.status,
            fine_tuned_model=self.fine_tuned_model,
        )
