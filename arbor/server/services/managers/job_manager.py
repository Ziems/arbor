from arbor.server.api.models.schemas import JobStatus
from arbor.server.core.config import Config
from arbor.server.services.jobs.file_train_job import FileTrainJob
from arbor.server.services.jobs.job import Job
from arbor.server.services.managers.base_manager import BaseManager


class JobManager(BaseManager):
    def __init__(self, config: Config, gpu_manager=None):
        super().__init__(config)
        self.jobs = {}
        self.gpu_manager = gpu_manager

    def cleanup(self) -> None:
        """Clean up all jobs and their resources"""
        if self._cleanup_called:
            return

        self.logger.info(f"Cleaning up {len(self.jobs)} jobs...")

        for job_id, job in self.jobs.items():
            try:
                self.logger.debug(f"Cleaning up job {job_id}")
                # All jobs have terminate method
                job.terminate()
            except Exception as e:
                self.logger.error(f"Error cleaning up job {job_id}: {e}")
                # Ensure GPU cleanup even if terminate fails
                try:
                    job._ensure_gpu_cleanup()
                except Exception as cleanup_error:
                    self.logger.error(
                        f"Error during GPU cleanup for job {job_id}: {cleanup_error}"
                    )

        self.jobs.clear()
        self._cleanup_called = True
        self.logger.info("JobManager cleanup completed")

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise ValueError(f"Job {job_id} not found")
        return self.jobs[job_id]

    def create_job(self) -> Job:
        job = Job(self.config)
        self.jobs[job.id] = job
        return job

    def create_file_train_job(self) -> FileTrainJob:
        job = FileTrainJob(self.config, gpu_manager=self.gpu_manager)
        self.jobs[job.id] = job
        return job

    def get_jobs(self) -> list[Job]:
        return list(self.jobs.values())

    def get_active_job(self) -> Job:
        for job in self.jobs.values():
            if job.status == JobStatus.RUNNING:
                return job
        return None

    def cancel_job(self, job_id: str) -> Job:
        """Cancel a job by its ID"""
        job = self.get_job(job_id)  # This will raise ValueError if not found

        # Check if job can be cancelled
        if job.status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
            raise ValueError(f"Cannot cancel job with status {job.status.value}")

        # Set to pending cancel first
        job.status = JobStatus.PENDING_CANCEL

        try:
            # Call the job's cancel method
            job.cancel()
            self.logger.info(f"Successfully cancelled job {job_id}")
        except Exception as e:
            # If cancellation fails, set to failed status
            job.status = JobStatus.FAILED
            self.logger.error(f"Failed to cancel job {job_id}: {e}")
            raise

        return job
