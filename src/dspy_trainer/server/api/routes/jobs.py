from fastapi import APIRouter, UploadFile, File
from dspy_trainer.server.services.job_manager import JobManager

router = APIRouter()
job_manager = JobManager()

@router.get("/{job_id}")
def get_job_status(job_id: str):
    pass