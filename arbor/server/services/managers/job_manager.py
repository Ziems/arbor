from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.jobs.job import Job
from arbor.server.services.managers.base_manager import BaseManager


class JobManager(BaseManager):
    def __init__(self, config: Config):
        super().__init__(config)
        self.jobs = {}

    def cleanup(self) -> None:
        """Clean up all jobs and their resources"""
        if self._cleanup_called:
            return

        self.logger.info(f"Cleaning up {len(self.jobs)} jobs...")

        for job_id, job in self.jobs.items():
            try:
                self.logger.debug(f"Cleaning up job {job_id}")
                # Call cleanup methods based on job type
                if hasattr(job, "terminate"):
                    job.terminate()
                elif hasattr(job, "kill"):
                    job.kill()
                elif hasattr(job, "cleanup"):
                    job.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up job {job_id}: {e}")

        self.jobs.clear()
        self._cleanup_called = True
        self.logger.info("JobManager cleanup completed")

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def create_job(self) -> Job:
        job = Job()
        self.jobs[job.id] = job
        return job

    def create_file_train_job(self) -> FileTrainJob:
        job = FileTrainJob(self.config)
        self.jobs[job.id] = job
        return job

    def get_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    def get_active_job(self) -> Job:
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                return job
        return None
