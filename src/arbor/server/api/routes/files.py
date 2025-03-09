from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from arbor.server.services.file_manager import FileManager
from arbor.server.api.models.schemas import FileResponse
from arbor.server.services.dependencies import get_file_manager
from arbor.server.services.file_manager import FileValidationError

router = APIRouter()

@router.post("", response_model=FileResponse)
async def upload_file(
    file: UploadFile = File(...),
    file_manager: FileManager = Depends(get_file_manager)
):
    if not file.filename.endswith('.jsonl'):
        raise HTTPException(status_code=400, detail="Only .jsonl files are allowed")

    try:
        content = await file.read()
        file_manager.validate_file_format(content)
        await file.seek(0)  # Reset file pointer to beginning
        return file_manager.save_uploaded_file(file)
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")