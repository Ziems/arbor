from fastapi import APIRouter, BackgroundTasks
from services.training_manager import TrainingManager
from pydantic import BaseModel

router = APIRouter()
training_manager = TrainingManager()


class FineTuneRequest(BaseModel):
    model_name: str
    training_file: str  # id of uploaded jsonl file

@router.post("/fine-tune")
def fine_tune(request: FineTuneRequest, background_tasks: BackgroundTasks):
    pass