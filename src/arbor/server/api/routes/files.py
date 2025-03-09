from fastapi import APIRouter, UploadFile, File, Depends
from arbor.server.services.file_manager import FileManager
from arbor.server.api.models.schemas import FileResponse
from arbor.server.services.dependencies import get_file_manager

router = APIRouter()

@router.post("", response_model=FileResponse)
def upload_file(
    file: UploadFile = File(...),
    file_manager: FileManager = Depends(get_file_manager)
):
    return file_manager.save_uploaded_file(file)