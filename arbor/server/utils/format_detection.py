"""
Utilities for detecting training data formats (SFT vs DPO) from JSONL files.
"""

import json
from pathlib import Path
from typing import Any, Dict, Literal

from arbor.server.utils.logging import get_logger

logger = get_logger(__name__)


def detect_record_format(record: Dict[Any, Any]) -> Literal["sft", "dpo", "unknown"]:
    """
    Detect the format of a single training record.

    Args:
        record: A dictionary representing a single training example

    Returns:
        "sft" if the record appears to be SFT format
        "dpo" if the record appears to be DPO format
        "unknown" if the format cannot be determined
    """
    if not isinstance(record, dict):
        return "unknown"

    # Check for SFT format indicators
    if "messages" in record and isinstance(record["messages"], list):
        return "sft"

    # Check for DPO format indicators
    if all(
        key in record for key in ["input", "preferred_output", "non_preferred_output"]
    ):
        return "dpo"

    return "unknown"


def detect_file_format(file_path: str) -> Literal["sft", "dpo", "unknown"]:
    """
    Detect the format of a JSONL training file by examining its structure.

    Args:
        file_path: Path to the JSONL file to analyze

    Returns:
        "sft" if the file appears to be SFT format
        "dpo" if the file appears to be DPO format
        "unknown" if the format cannot be determined
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return "unknown"

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    detected_format = detect_record_format(data)

                    # Return as soon as we detect a format
                    if detected_format != "unknown":
                        return detected_format

                except json.JSONDecodeError:
                    continue

            return "unknown"

    except Exception as e:
        logger.warning(f"Error detecting file format for {file_path}: {e}")
        return "unknown"


def validate_format_consistency(
    file_path: str, expected_format: Literal["sft", "dpo"]
) -> bool:
    """
    Validate that all records in a file match the expected format.

    Args:
        file_path: Path to the JSONL file to validate
        expected_format: The expected format ("sft" or "dpo")

    Returns:
        True if all records match the expected format, False otherwise
    """
    try:
        file_path = Path(file_path)
        if not file_path.exists():
            return False

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue

                try:
                    data = json.loads(line)
                    detected_format = detect_record_format(data)

                    if detected_format != expected_format:
                        logger.warning(
                            f"Format mismatch at line {line_num}: "
                            f"expected {expected_format}, got {detected_format}"
                        )
                        return False

                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON at line {line_num}")
                    return False

        return True

    except Exception as e:
        logger.error(f"Error validating format consistency for {file_path}: {e}")
        return False
