from typing import Literal

from fastapi import APIRouter, Body, File, HTTPException, Request, UploadFile

from arbor.server.api.models.schemas import FileModel, PaginatedResponse
from arbor.server.services.managers.file_manager import FileManager, FileValidationError

# https://platform.openai.com/docs/api-reference/files/list
router = APIRouter()


@router.post("", response_model=FileModel)
async def upload_file(
    request: Request,
    file: UploadFile = File(...),
    purpose: Literal["assistants", "vision", "fine-tune", "batch"] = Body("fine-tune"),
):
    file_manager: FileManager = request.app.state.file_manager
    if not file.filename.endswith(".jsonl"):
        raise HTTPException(status_code=400, detail="Only .jsonl files are allowed")

    try:
        # Read and validate content before saving
        content = await file.read()
        file_manager.validate_content_format(content)

        # Reset file pointer to beginning for saving
        await file.seek(0)

        return FileModel(**file_manager.save_uploaded_file(file))
    except FileValidationError as e:
        raise HTTPException(status_code=400, detail=f"Invalid file format: {str(e)}")


@router.get("", response_model=PaginatedResponse[FileModel])
def list_files(request: Request):
    file_manager: FileManager = request.app.state.file_manager
    return PaginatedResponse(
        items=file_manager.get_files(),
        total=len(file_manager.get_files()),
        page=1,
        page_size=10,
    )


@router.get("/{file_id}", response_model=FileModel)
def get_file(request: Request, file_id: str):
    file_manager: FileManager = request.app.state.file_manager
    return file_manager.get_file(file_id)


@router.delete("/{file_id}")
def delete_file(request: Request, file_id: str):
    file_manager: FileManager = request.app.state.file_manager
    file_manager.delete_file(file_id)
    return {"message": "File deleted"}
