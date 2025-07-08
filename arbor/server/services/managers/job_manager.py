from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Settings
from arbor.server.services.jobs.job import Job


class JobManager:
    def __init__(self, settings: Settings):
        self.jobs = {}

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def create_job(self) -> Job:
        job = Job()
        self.jobs[job.id] = job
        return job

    def get_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    def get_active_job(self) -> Job:
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                return job
        return None
