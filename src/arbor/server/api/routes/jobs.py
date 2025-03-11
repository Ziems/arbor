from fastapi import APIRouter, Request, BackgroundTasks
from typing import List

from arbor.server.api.models.schemas import JobStatusResponse, FineTuneRequest, JobStatus
from arbor.server.services.job_manager import JobStatus

router = APIRouter()

@router.post("", response_model=JobStatusResponse)
def create_fine_tune_job(request: Request, fine_tune_request: FineTuneRequest, background_tasks: BackgroundTasks):
    job_manager = request.app.state.job_manager
    file_manager = request.app.state.file_manager
    training_manager = request.app.state.training_manager

    job = job_manager.create_job()
    background_tasks.add_task(training_manager.fine_tune, fine_tune_request, job, file_manager)
    job.status = JobStatus.QUEUED
    return JobStatusResponse(id=job.id, status=job.status.value)

@router.get("", response_model=List[JobStatusResponse])
def get_jobs(request: Request):
    job_manager = request.app.state.job_manager
    return [JobStatusResponse(id=job.id, status=job.status.value) for job in job_manager.get_jobs()]


@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    request: Request,
    job_id: str,
):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    return JobStatusResponse(id=job_id, status=job.status.value, fine_tuned_model=job.fine_tuned_model)