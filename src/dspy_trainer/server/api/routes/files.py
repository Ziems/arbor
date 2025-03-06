from fastapi import APIRouter, UploadFile, File
from services.file_manager import FileManager

router = APIRouter()
file_manager = FileManager()

@router.post("/files")
def upload_file(file: UploadFile = File(...)):
    pass