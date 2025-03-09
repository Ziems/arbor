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
    job = job_manager.get_job(job_id)
    return JobStatusResponse(id=job_id, status=job.status.value, fine_tuned_model=job.fine_tuned_model)