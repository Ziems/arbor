from fastapi import APIRouter, UploadFile, File
from services.job_manager import JobManager

router = APIRouter()
job_manager = JobManager()

@router.post("/job/{job_id}")
def get_job_status(job_id: str):
    pass