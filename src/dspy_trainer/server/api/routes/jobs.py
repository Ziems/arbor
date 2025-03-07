from fastapi import APIRouter, UploadFile, File
from dspy_trainer.server.services.job_manager import JobManager
from dspy_trainer.server.api.models.schemas import JobStatusResponse

router = APIRouter()
job_manager = JobManager()

@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str):
    job = job_manager.get_job(job_id)
    return JobStatusResponse(status=job.status)