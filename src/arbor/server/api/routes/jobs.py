from fastapi import APIRouter, Depends
from arbor.server.services.job_manager import JobManager
from arbor.server.services.dependencies import get_job_manager
from arbor.server.api.models.schemas import JobStatusResponse

router = APIRouter()

@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    job_id: str,
    job_manager: JobManager = Depends(get_job_manager)
):
    status = job_manager.get_job_status(job_id)
    return JobStatusResponse(job_id=job_id, status=status.value)