import uuid
from datetime import datetime
from typing import Literal

import coolname

from arbor.server.api.models.schemas import JobStatus


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
    def __init__(self, id=None, prefix="ftjob"):
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

        self.created_at = datetime.now()

    def add_event(self, event: JobEvent):
        self.events.append(event)

    def get_events(self) -> list[JobEvent]:
        return self.events

    def add_checkpoint(self, checkpoint: JobCheckpoint):
        self.checkpoints.append(checkpoint)

    def get_checkpoints(self) -> list[JobCheckpoint]:
        return self.checkpoints
