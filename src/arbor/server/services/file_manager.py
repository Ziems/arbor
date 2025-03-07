from pathlib import Path
import json
import os
import shutil
import time
import uuid
from fastapi import UploadFile
from arbor.server.api.models.schemas import FileResponse

class FileManager:
  def __init__(self):
    self.uploads_dir = Path("uploads")
    self.uploads_dir.mkdir(exist_ok=True)
    self.files = self.load_files_from_uploads()

  def load_files_from_uploads(self):
    files = {}

    # Scan through all directories in uploads directory
    for dir_path in self.uploads_dir.glob("*"):
      if not dir_path.is_dir():
        continue

      # Check for metadata.json
      metadata_path = dir_path / "metadata.json"
      if not metadata_path.exists():
        continue

      # Load metadata
      with open(metadata_path) as f:
        metadata = json.load(f)

      # Find the .jsonl file
      jsonl_files = list(dir_path.glob("*.jsonl"))
      if not jsonl_files:
        continue

      file_path = jsonl_files[0]
      files[dir_path.name] = {
        "path": str(file_path),
        "purpose": metadata.get("purpose", "training"),
        "bytes": file_path.stat().st_size,
        "created_at": metadata.get("created_at", int(file_path.stat().st_mtime)),
        "filename": metadata.get("filename", file_path.name)
      }

    return files

  def save_uploaded_file(self, file: UploadFile) -> FileResponse:
    file_id = str(uuid.uuid4())
    dir_path = self.uploads_dir / file_id
    dir_path.mkdir(exist_ok=True)

    # Save the actual file
    file_path = dir_path / f"data.jsonl"
    with open(file_path, "wb") as f:
      shutil.copyfileobj(file.file, f)

    # Create metadata
    metadata = {
      "purpose": "training",
      "created_at": int(time.time()),
      "filename": file.filename
    }

    # Save metadata
    with open(dir_path / "metadata.json", "w") as f:
      json.dump(metadata, f)

    file_data = {
      "id": file_id,
      "path": str(file_path),
      "purpose": metadata["purpose"],
      "bytes": file.size,
      "created_at": metadata["created_at"],
      "filename": metadata["filename"]
    }

    self.files[file_id] = file_data
    return FileResponse(**file_data)

  def get_file(self, file_id: str):
    return self.files[file_id]