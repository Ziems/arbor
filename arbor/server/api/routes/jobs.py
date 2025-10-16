from fastapi import APIRouter, HTTPException, Request

from arbor.server.api.models.schemas import (
    FineTuneRequest,
    JobCheckpointModel,
    JobEventModel,
    JobStatus,
    JobStatusModel,
    PaginatedResponse,
)
from arbor.server.services.managers.file_manager import FileManager
from arbor.server.services.managers.file_train_manager import FileTrainManager
from arbor.server.services.managers.job_manager import JobManager

router = APIRouter()


# Create a fine-tune job
@router.post("", response_model=JobStatusModel)
def create_fine_tune_job(
    request: Request,
    fine_tune_request: FineTuneRequest,
):
    job_manager: JobManager = request.app.state.job_manager
    file_manager: FileManager = request.app.state.file_manager
    file_train_manager: FileTrainManager = request.app.state.file_train_manager

    job = job_manager.create_file_train_job()
    try:
        file_train_manager.fine_tune(fine_tune_request, job, file_manager)
        job.status = JobStatus.QUEUED
        return job.to_status_model()
    except ValueError as e:
        # Handle cases where training file is not found or other validation errors
        raise HTTPException(status_code=400, detail=str(e))


# List fine-tune jobs (paginated)
@router.get("", response_model=PaginatedResponse[JobStatusModel])
def get_jobs(request: Request):
    job_manager = request.app.state.job_manager
    return PaginatedResponse(
        data=[job.to_status_model() for job in job_manager.get_jobs()],
        has_more=False,
    )


# List fine-tuning events
@router.get("/{job_id}/events", response_model=PaginatedResponse[JobEventModel])
def get_job_events(request: Request, job_id: str):
    job_manager = request.app.state.job_manager
    try:
        job = job_manager.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
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
    try:
        job = job_manager.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
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
    print("getting job status")
    job_manager = request.app.state.job_manager
    try:
        job = job_manager.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_status_model()


# Cancel a fine-tune job
@router.post("/{job_id}/cancel", response_model=JobStatusModel)
def cancel_job(request: Request, job_id: str):
    job_manager = request.app.state.job_manager

    try:
        job = job_manager.cancel_job(job_id)
        return job.to_status_model()
    except ValueError as e:
        # Check if it's a "not found" error or "cannot cancel" error
        if "not found" in str(e):
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to cancel job: {str(e)}")