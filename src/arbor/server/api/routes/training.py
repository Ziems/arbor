from fastapi import APIRouter, BackgroundTasks, Depends

from arbor.server.api.models.schemas import FineTuneRequest, JobStatusResponse
from arbor.server.services.job_manager import JobManager, JobStatus
from arbor.server.services.file_manager import FileManager
from arbor.server.services.training_manager import TrainingManager
from arbor.server.services.dependencies import get_training_manager, get_job_manager, get_file_manager

router = APIRouter()

@router.post("", response_model=JobStatusResponse)
def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks, training_manager: TrainingManager = Depends(get_training_manager), job_manager: JobManager = Depends(get_job_manager), file_manager: FileManager = Depends(get_file_manager)):
    job = job_manager.create_job()
    background_tasks.add_task(training_manager.fine_tune, request, job, file_manager)
    job.status = JobStatus.QUEUED
    return JobStatusResponse(job_id=job.id, status=job.status.value)
