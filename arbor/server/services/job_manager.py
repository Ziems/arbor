import uuid
from datetime import datetime
from typing import Literal

from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Settings


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
    def __init__(self, status: JobStatus):
        self.id = str(f"ftjob-{uuid.uuid4()}")
        self.status = status
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


class JobManager:
    def __init__(self, settings: Settings):
        self.jobs = {}

    def get_job(self, job_id: str):
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def create_job(self):
        job = Job(status=JobStatus.PENDING)
        self.jobs[job.id] = job
        return job

    def get_jobs(self):
        return list(self.jobs.values())

    def get_active_job(self):
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                return job
        return None
