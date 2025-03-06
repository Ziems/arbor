from fastapi import APIRouter, UploadFile, File
from dspy_trainer.server.services.file_manager import FileManager
from dspy_trainer.server.api.models.schemas import FileResponse

router = APIRouter()
file_manager = FileManager()


@router.post("", response_model=FileResponse)
def upload_file(file: UploadFile = File(...)):
    return file_manager.save_uploaded_file(file)