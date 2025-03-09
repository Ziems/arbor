from pydantic import BaseModel

class FileResponse(BaseModel):
    id: str
    object: str = "file"
    bytes: int
    created_at: int
    filename: str
    purpose: str

class FineTuneRequest(BaseModel):
    model: str
    training_file: str  # id of uploaded jsonl file

class JobStatusResponse(BaseModel):
    id: str
    status: str
    details: str = ""
    fine_tuned_model: str | None = None