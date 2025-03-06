from fastapi import APIRouter, BackgroundTasks
from dspy_trainer.server.services.training_manager import TrainingManager
from dspy_trainer.server.api.models.schemas import FineTuneRequest, JobStatus

router = APIRouter()
training_manager = TrainingManager()


@router.post("")
def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks):
    pass