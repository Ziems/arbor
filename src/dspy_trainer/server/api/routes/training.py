from fastapi import APIRouter, BackgroundTasks
from dspy_trainer.server.services.training_manager import TrainingManager
from dspy_trainer.server.api.models.schemas import FineTuneRequest, JobStatusResponse
from dspy_trainer.server.api.routes.jobs import job_manager # this should probably be in the jobs manager, not in the routes
from dspy_trainer.server.services.job_manager import JobStatus

router = APIRouter()
training_manager = TrainingManager()


@router.post("", response_model=JobStatusResponse)
def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks):
    job = job_manager.create_job()
    background_tasks.add_task(training_manager.fine_tune, request, job)
    job.status = JobStatus.QUEUED
    return JobStatusResponse(job_id=job.id, status=job.status.value)
