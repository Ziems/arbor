import json
import shutil
import time
import uuid
from pathlib import Path
from typing import Literal

from fastapi import UploadFile

from arbor.server.core.config import Config
from arbor.server.services.managers.base_manager import BaseManager
from arbor.server.utils.format_detection import detect_file_format


class FileValidationError(Exception):
    """Custom exception for file validation errors"""

    pass


class FileManager(BaseManager):
    def __init__(self, config: Config, gpu_manager=None):
        super().__init__(config)
        self.gpu_manager = gpu_manager
        self.uploads_dir = Path(config.storage_path) / "uploads"
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
                "created_at": metadata.get(
                    "created_at", int(file_path.stat().st_mtime)
                ),
                "filename": metadata.get("filename", file_path.name),
                "format": metadata.get("format", "unknown"),
            }

        return files

    def check_file_format(self, file_id: str) -> Literal["sft", "dpo", "unknown"]:
        """
        Gets the detected format of an uploaded file.

        Args:
            file_id: ID of the uploaded file

        Returns:
            The detected format from metadata, or attempts detection if not stored
        """
        file = self.get_file(file_id)
        if file is None:
            raise FileValidationError(f"File {file_id} not found")

        # Try to get format from metadata first
        if "format" in file:
            return file["format"]

        # If not in metadata, try to detect it
        detected_format = detect_file_format(file["path"])

        # Update the file metadata with detected format
        file["format"] = detected_format
        self._update_file_metadata(file_id, {"format": detected_format})

        return detected_format

    def _update_file_metadata(self, file_id: str, updates: dict):
        """Helper method to update file metadata on disk"""
        try:
            dir_path = self.uploads_dir / file_id
            metadata_path = dir_path / "metadata.json"

            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                metadata.update(updates)

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f)
        except Exception as e:
            self.logger.warning(f"Failed to update metadata for {file_id}: {e}")

    def save_uploaded_file(self, file: UploadFile):
        file_id = f"file-{str(uuid.uuid4())}"
        dir_path = self.uploads_dir / file_id
        dir_path.mkdir(exist_ok=True)

        # Save the actual file
        file_path = dir_path / "data.jsonl"
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Detect file format
        detected_format = detect_file_format(str(file_path))

        # Create metadata with format information
        metadata = {
            "purpose": "training",
            "created_at": int(time.time()),
            "filename": file.filename,
            "format": detected_format,
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
            "filename": metadata["filename"],
            "format": metadata["format"],
        }

        self.files[file_id] = file_data
        return file_data

    def get_file(self, file_id: str):
        return self.files.get(file_id)

    def delete_file(self, file_id: str):
        if file_id not in self.files:
            return

        dir_path = self.uploads_dir / file_id
        if dir_path.exists():
            shutil.rmtree(dir_path)

        del self.files[file_id]

    def validate_content_format(self, content: bytes) -> None:
        """
        Validates the format of the content.
        """
        detected_format = self._detect_content_format(content)
        if detected_format == "sft":
            self._validate_sft_content(content)
        elif detected_format == "dpo":
            self._validate_dpo_content(content)
        else:
            raise FileValidationError(
                "File format could not be determined. Please ensure the file is valid SFT or DPO format."
            )

    def _detect_content_format(
        self, content: bytes
    ) -> Literal["sft", "dpo", "unknown"]:
        """
        Detect the format of content by examining its structure.
        """
        try:
            lines = content.decode("utf-8").split("\n")
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    if not isinstance(data, dict):
                        continue

                    # Check for SFT format indicators
                    if "messages" in data and isinstance(data["messages"], list):
                        return "sft"

                    # Check for DPO format indicators
                    if all(
                        key in data
                        for key in ["input", "preferred_output", "non_preferred_output"]
                    ):
                        return "dpo"

                except json.JSONDecodeError:
                    continue

            return "unknown"

        except Exception as e:
            self.logger.warning(f"Error detecting content format: {e}")
            return "unknown"

    def validate_file_format(
        self, file_path: str, format_type: Literal["sft", "dpo"]
    ) -> None:
        """
        Validates that the file at file_path is properly formatted JSONL with expected structure.

        Args:
            file_path: Path to the file to validate
            format_type: Type of format to validate ('sft' or 'dpo')

        Raises:
            FileValidationError: If validation fails
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()

            if format_type == "sft":
                self._validate_sft_content(content)
            elif format_type == "dpo":
                self._validate_dpo_content(content)
            else:
                raise FileValidationError(f"Unknown format type: {format_type}")
        except Exception as e:
            raise FileValidationError(f"Failed to read or validate file: {e}")

    def _validate_sft_content(self, content: bytes) -> None:
        """
        Validates SFT format content: JSONL with messages array structure.
        """
        try:
            lines = content.decode("utf-8").split("\n")
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue  # skip empty lines
                try:
                    data = json.loads(line)

                    if not isinstance(data, dict):
                        raise FileValidationError(
                            f"Line {line_num}: Each line must be a JSON object"
                        )

                    if "messages" not in data:
                        raise FileValidationError(
                            f"Line {line_num}: Missing 'messages' field"
                        )

                    if not isinstance(data["messages"], list):
                        raise FileValidationError(
                            f"Line {line_num}: 'messages' must be an array"
                        )

                    for msg in data["messages"]:
                        if not isinstance(msg, dict):
                            raise FileValidationError(
                                f"Line {line_num}: Each message must be an object"
                            )
                        if "role" not in msg or "content" not in msg:
                            raise FileValidationError(
                                f"Line {line_num}: Messages must have 'role' and 'content' fields"
                            )
                        if not isinstance(msg["role"], str) or not isinstance(
                            msg["content"], str
                        ):
                            raise FileValidationError(
                                f"Line {line_num}: Message 'role' and 'content' must be strings"
                            )

                except json.JSONDecodeError:
                    raise FileValidationError(f"Invalid JSON on line {line_num}")

        except Exception as e:
            raise FileValidationError(f"Failed to validate content: {e}")

    def _validate_dpo_content(self, content: bytes) -> None:
        """
        Validates DPO format content: JSONL with input, preferred_output, and non_preferred_output structure.
        """
        try:
            lines = content.decode("utf-8").split("\n")
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)

                    if not isinstance(data, dict):
                        raise FileValidationError(
                            f"Line {line_num}: Each line must be a JSON object"
                        )

                    input_data = data.get("input")
                    if not isinstance(input_data, dict):
                        raise FileValidationError(
                            f"Line {line_num}: Missing or invalid 'input' field"
                        )

                    if "messages" not in input_data or not isinstance(
                        input_data["messages"], list
                    ):
                        raise FileValidationError(
                            f"Line {line_num}: 'input.messages' must be a list"
                        )
                    for msg in input_data["messages"]:
                        if not isinstance(msg, dict):
                            raise FileValidationError(
                                f"Line {line_num}: Each 'message' must be an object"
                            )
                        if "role" not in msg or "content" not in msg:
                            raise FileValidationError(
                                f"Line {line_num}: Each message must have 'role' and 'content'"
                            )
                        if not isinstance(msg["role"], str) or not isinstance(
                            msg["content"], str
                        ):
                            raise FileValidationError(
                                f"Line {line_num}: 'role' and 'content' must be strings"
                            )

                    if "tools" not in input_data or not isinstance(
                        input_data["tools"], list
                    ):
                        raise FileValidationError(
                            f"Line {line_num}: 'input.tools' must be a list"
                        )

                    if "parallel_tool_calls" not in input_data or not isinstance(
                        input_data["parallel_tool_calls"], bool
                    ):
                        raise FileValidationError(
                            f"Line {line_num}: 'input.parallel_tool_calls' must be a boolean"
                        )

                    preferred = data.get("preferred_output")
                    if not isinstance(preferred, list):
                        raise FileValidationError(
                            f"Line {line_num}: 'preferred_output' must be a list"
                        )
                    for msg in preferred:
                        if not isinstance(msg, dict):
                            raise FileValidationError(
                                f"Line {line_num}: Each 'preferred_output' message must be an object"
                            )
                        if "role" not in msg or "content" not in msg:
                            raise FileValidationError(
                                f"Line {line_num}: Each preferred_output message must have 'role' and 'content'"
                            )
                        if not isinstance(msg["role"], str) or not isinstance(
                            msg["content"], str
                        ):
                            raise FileValidationError(
                                f"Line {line_num}: 'role' and 'content' in preferred_output must be strings"
                            )

                    non_preferred = data.get("non_preferred_output")
                    if not isinstance(non_preferred, list):
                        raise FileValidationError(
                            f"Line {line_num}: 'non_preferred_output' must be a list"
                        )
                    for msg in non_preferred:
                        if not isinstance(msg, dict):
                            raise FileValidationError(
                                f"Line {line_num}: Each 'non_preferred_output' message must be an object"
                            )
                        if "role" not in msg or "content" not in msg:
                            raise FileValidationError(
                                f"Line {line_num}: Each non_preferred_output message must have 'role' and 'content'"
                            )
                        if not isinstance(msg["role"], str) or not isinstance(
                            msg["content"], str
                        ):
                            raise FileValidationError(
                                f"Line {line_num}: 'role' and 'content' in non_preferred_output must be strings"
                            )

                except json.JSONDecodeError:
                    raise FileValidationError(f"Invalid JSON on line {line_num}")

        except Exception as e:
            raise FileValidationError(f"Failed to validate content: {e}")

    def _validate_sft_format(self, file_path: str) -> None:
        """
        Validates SFT format: JSONL with messages array structure.
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._validate_sft_content(content)
        except Exception as e:
            raise FileValidationError(f"Failed to read or validate file: {e}")

    def _validate_dpo_format(self, file_path: str) -> None:
        """
        Validates DPO format: JSONL with input, preferred_output, and non_preferred_output structure.
        """
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self._validate_dpo_content(content)
        except Exception as e:
            raise FileValidationError(f"Failed to validate file: {e}")

    def cleanup(self) -> None:
        """Clean up FileManager resources"""
        if self._cleanup_called:
            return

        self.logger.info("FileManager cleanup completed (no active resources to clean)")
        self._cleanup_called = True
