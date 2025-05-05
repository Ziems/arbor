from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from arbor.server.api.models.schemas import (
    FineTuneRequest,
    JobCheckpointModel,
    JobEventModel,
    JobStatus,
    JobStatusModel,
    PaginatedResponse,
)
from arbor.server.services.job_manager import JobStatus

router = APIRouter()


# Create a fine-tune job
@router.post("", response_model=JobStatusModel)
def create_fine_tune_job(
    request: Request,
    fine_tune_request: FineTuneRequest,
    background_tasks: BackgroundTasks,
):
    job_manager = request.app.state.job_manager
    file_manager = request.app.state.file_manager
    training_manager = request.app.state.training_manager

    job = job_manager.create_job()
    background_tasks.add_task(
        training_manager.fine_tune, fine_tune_request, job, file_manager
    )
    job.status = JobStatus.QUEUED
    return JobStatusModel(id=job.id, status=job.status.value)


# List fine-tune jobs (paginated)
@router.get("", response_model=PaginatedResponse[JobStatusModel])
def get_jobs(request: Request):
    job_manager = request.app.state.job_manager
    return PaginatedResponse(
        data=[
            JobStatusModel(id=job.id, status=job.status.value)
            for job in job_manager.get_jobs()
        ],
        has_more=False,
    )


# List fine-tuning events
@router.get("/{job_id}/events", response_model=PaginatedResponse[JobEventModel])
def get_job_events(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    return PaginatedResponse(
        data=[
            JobEventModel(
                id=event.id,
                level=event.level,
                message=event.message,
                data=event.data,
                created_at=int(event.created_at.timestamp()),
                type="message",
            )
            for event in job.get_events()
        ],
        has_more=False,
    )


# List fine-tuning checkpoints
@router.get(
    "/{job_id}/checkpoints", response_model=PaginatedResponse[JobCheckpointModel]
)
def get_job_checkpoints(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    return PaginatedResponse(
        data=[
            JobCheckpointModel(
                id=checkpoint.id,
                fine_tuned_model_checkpoint=checkpoint.fine_tuned_model_checkpoint,
                fine_tuning_job_id=checkpoint.fine_tuning_job_id,
                metrics=checkpoint.metrics,
                step_number=checkpoint.step_number,
            )
            for checkpoint in job.get_checkpoints()
        ],
        has_more=False,
    )


# Retrieve a fine-tune job by id
@router.get("/{job_id}", response_model=JobStatusModel)
def get_job_status(
    request: Request,
    job_id: str,
):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    return JobStatusModel(
        id=job_id, status=job.status.value, fine_tuned_model=job.fine_tuned_model
    )


# Cancel a fine-tune job
@router.post("/{job_id}/cancel", response_model=JobStatusModel)
def cancel_job(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)

    # Only allow cancellation of jobs that aren't finished
    if job.status in [JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED]:
        raise HTTPException(
            status_code=400, detail=f"Cannot cancel job with status {job.status.value}"
        )

    job.status = JobStatus.PENDING_CANCEL
    return JobStatusModel(id=job.id, status=job.status.value)
