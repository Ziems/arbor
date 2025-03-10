from fastapi import APIRouter, BackgroundTasks, Request

from arbor.server.api.models.schemas import FineTuneRequest, JobStatusResponse
from arbor.server.services.job_manager import JobStatus

router = APIRouter()

@router.post("", response_model=JobStatusResponse)
def fine_tune(request: Request, fine_tune_request: FineTuneRequest, background_tasks: BackgroundTasks):
    job_manager = request.app.state.job_manager
    file_manager = request.app.state.file_manager
    training_manager = request.app.state.training_manager

    job = job_manager.create_job()
    background_tasks.add_task(training_manager.fine_tune, fine_tune_request, job, file_manager)
    job.status = JobStatus.QUEUED
    return JobStatusResponse(id=job.id, status=job.status.value)
