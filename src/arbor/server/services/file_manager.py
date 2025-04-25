from pathlib import Path
import json
import os
import shutil
import time
import uuid
from fastapi import UploadFile
from arbor.server.core.config import Settings

class FileValidationError(Exception):
    """Custom exception for file validation errors"""
    pass

class FileManager:
  def __init__(self, settings: Settings):
    self.uploads_dir = Path(settings.STORAGE_PATH) / "uploads"
    self.uploads_dir.mkdir(parents=True, exist_ok=True)
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

  def save_uploaded_file(self, file: UploadFile):
    file_id = f'file-{str(uuid.uuid4())}'
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
    return file_data

  def get_file(self, file_id: str):
    return self.files[file_id]

  def delete_file(self, file_id: str):
    if file_id not in self.files:
      return

    dir_path = self.uploads_dir / file_id
    if dir_path.exists():
      shutil.rmtree(dir_path)

    del self.files[file_id]


  def validate_file_format_sft(self, file_path: str) -> None:
    """
    Validates that the file at file_path is properly formatted JSONL with expected structure.
    Raises FileValidationError if validation fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    data = json.loads(line)

                    if not isinstance(data, dict):
                        raise FileValidationError(f"Line {line_num}: Each line must be a JSON object")

                    if "messages" not in data:
                        raise FileValidationError(f"Line {line_num}: Missing 'messages' field")

                    if not isinstance(data["messages"], list):
                        raise FileValidationError(f"Line {line_num}: 'messages' must be an array")

                    for msg in data["messages"]:
                        if not isinstance(msg, dict):
                            raise FileValidationError(f"Line {line_num}: Each message must be an object")
                        if "role" not in msg or "content" not in msg:
                            raise FileValidationError(f"Line {line_num}: Messages must have 'role' and 'content' fields")
                        if not isinstance(msg["role"], str) or not isinstance(msg["content"], str):
                            raise FileValidationError(f"Line {line_num}: Message 'role' and 'content' must be strings")

                except json.JSONDecodeError:
                    raise FileValidationError(f"Invalid JSON on line {line_num}")

    except Exception as e:
        raise FileValidationError(f"Failed to read or validate file: {e}")


  def validate_file_format_dpo(self, file_path: str) -> None:
    """
    Validates that the file at file_path is properly formatted JSONL with expected structure.
    Raises FileValidationError if validation fails.
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.read().strip().split('\n')
    except Exception as e:
        raise FileValidationError(f"Failed to read file: {e}")

    if not lines:
        raise FileValidationError("File is empty or contains no valid data")

    for line_num, line in enumerate(lines, 1):
        try:
            data = json.loads(line)

            if not isinstance(data, dict):
                raise FileValidationError(f"Line {line_num}: Each line must be a JSON object")

            if "chosen" not in data:
                raise FileValidationError(f"Line {line_num}: Missing 'chosen' field")
            if "rejected" not in data:
                raise FileValidationError(f"Line {line_num}: Missing 'rejected' field")

            # https://github.com/huggingface/trl/blob/29c5e05e3a5f1f8a369aaef78fc9f36878db9194/trl/data_utils.py#L349

        except json.JSONDecodeError:
            raise FileValidationError(f"Invalid JSON on line {line_num}")