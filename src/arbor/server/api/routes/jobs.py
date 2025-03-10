from fastapi import APIRouter, Request
from arbor.server.api.models.schemas import JobStatusResponse

router = APIRouter()

@router.get("/{job_id}", response_model=JobStatusResponse)
def get_job_status(
    request: Request,
    job_id: str,
):
    job_manager = request.app.state.job_manager
    job = job_manager.get_job(job_id)
    return JobStatusResponse(id=job_id, status=job.status.value, fine_tuned_model=job.fine_tuned_model)