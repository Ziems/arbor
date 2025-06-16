"""
Logging configuration for Arbor.
Provides centralized logging setup with file and console handlers.
"""

import logging
import logging.config
import sys
from pathlib import Path
from typing import Any, Dict, Optional


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
    }

    def format(self, record):
        # Use the name mapping to get short names, fallback to original name
        name_mappings = {
            "arbor.server.services.inference_manager": "infer",
            "arbor.server.services.grpo_manager": "grpo",
            "arbor.server.services.file_manager": "files",
            "arbor.server.services.health_manager": "health",
            "arbor.server.services.job_manager": "jobs",
            "arbor.server.services.training_manager": "train",
            "arbor.server.services.comms.comms": "comms",
            "arbor.server.services.scripts.sft_training": "sft",
            "arbor.server.services.scripts.grpo_training": "grpo",
            "arbor.config": "config",
            "arbor.cli": "cli",
            "__main__": "main",
            # Add uvicorn loggers
            "uvicorn": "api",
            "uvicorn.access": "api",
            "uvicorn.error": "api",
        }

        # Get short name if available, otherwise use the provided name
        short_name = name_mappings.get(record.name, record.name)
        name = short_name.upper()
        record.name = f"[{name}]".rjust(
            8
        )  # Right-align the whole bracketed name (8 chars max)

        # Store original level name for color lookup
        original_level = record.levelname

        # Convert level names to 4-character abbreviations
        level_abbreviations = {
            "DEBUG": "DEBG",
            "INFO": "INFO",
            "WARNING": "WARN",
            "ERROR": "ERRO",
            "CRITICAL": "CRIT",
        }

        # Get abbreviated level name and add brackets with padding
        abbreviated_level = level_abbreviations.get(original_level, original_level[:4])
        record.levelname = f"[{abbreviated_level}]"

        # Add color to the bracketed level name
        if original_level in self.COLORS:
            record.levelname = (
                f"{self.COLORS[original_level]}{record.levelname}{self.COLORS['RESET']}"
            )

        return super().format(record)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    log_format: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Setup logging configuration for Arbor.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        log_format: Custom log format string

    Returns:
        Dictionary with logging configuration details
    """

    # Default log format
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Console format with colors and tight alignment
    console_format = "%(name)s - %(levelname)s - %(message)s"

    # Create formatters
    file_formatter = logging.Formatter(log_format)
    console_formatter = ColoredFormatter(console_format)

    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    handlers = []

    # Console handler
    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, log_level.upper()))
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        handlers.append("console")

    # File handlers
    if enable_file_logging and log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        # Main log file (all levels)
        main_log_file = log_dir / "arbor.log"
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(logging.DEBUG)  # Capture all levels in file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)
        handlers.append("file")

        # Error log file (errors and critical only)
        error_log_file = log_dir / "arbor_error.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_handler)
        handlers.append("error_file")

    # Configure specific loggers for different components
    configure_component_loggers(log_level)

    return {
        "level": log_level,
        "handlers": handlers,
        "log_dir": str(log_dir) if log_dir else None,
        "main_log_file": str(log_dir / "arbor.log") if log_dir else None,
        "error_log_file": str(log_dir / "arbor_error.log") if log_dir else None,
    }


def configure_component_loggers(log_level: str):
    """Configure loggers for specific Arbor components."""

    # Arbor component loggers with shorter names (8 chars max for proper centering)
    component_loggers = {
        "arbor.server.services.inference_manager": "infer",
        "arbor.server.services.grpo_manager": "grpo",
        "arbor.server.services.file_manager": "files",
        "arbor.server.services.health_manager": "health",
        "arbor.server.services.job_manager": "jobs",
        "arbor.server.services.training_manager": "train",
        "arbor.server.services.comms": "comms",
        "arbor.server.services.scripts.sft_training": "sft",
        "arbor.server.services.scripts.grpo_training": "grpo",
        "arbor.config": "config",
        "arbor.cli": "cli",
        "__main__": "main",
    }

    # Create shorter logger aliases
    for full_name, short_name in component_loggers.items():
        # Set up the short name logger to use the same handlers as root
        short_logger = logging.getLogger(short_name)
        short_logger.setLevel(getattr(logging, log_level.upper()))

        # Also configure the full name for backward compatibility
        full_logger = logging.getLogger(full_name)
        full_logger.setLevel(getattr(logging, log_level.upper()))

    # Third-party library loggers (usually more verbose)
    third_party_loggers = {
        "uvicorn": "WARNING",
        "uvicorn.access": "WARNING",
        "uvicorn.error": "WARNING",
        "fastapi": "WARNING",
        "httpx": "WARNING",
        "urllib3": "WARNING",
        "vllm": "INFO",
        "torch": "WARNING",
        "transformers": "WARNING",
    }

    for logger_name, level in third_party_loggers.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level))


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name, automatically mapping long module names to short ones.

    Args:
        name: Logger name, typically __name__

    Returns:
        Configured logger instance with short name if available
    """
    # Mapping of full module names to short names (8 chars max for proper centering)
    name_mappings = {
        "arbor.server.services.inference_manager": "infer",
        "arbor.server.services.grpo_manager": "grpo",
        "arbor.server.services.file_manager": "files",
        "arbor.server.services.health_manager": "health",
        "arbor.server.services.job_manager": "jobs",
        "arbor.server.services.training_manager": "train",
        "arbor.server.services.comms.comms": "comms",
        "arbor.server.services.scripts.sft_training": "sft",
        "arbor.server.services.scripts.grpo_training": "grpo",
        "arbor.config": "config",
        "arbor.cli": "cli",
        "__main__": "main",
        # Add uvicorn loggers
        "uvicorn": "api",
        "uvicorn.access": "api",
        "uvicorn.error": "api",
    }

    # Use short name if available, otherwise use the provided name
    short_name = name_mappings.get(name, name)
    return logging.getLogger(short_name)


def log_system_info():
    """Log system information at startup."""
    logger = get_logger("arbor.startup")

    logger.info("=" * 60)
    logger.info("ARBOR SYSTEM STARTUP")
    logger.info("=" * 60)

    # This will be populated by the health manager
    logger.info("System information logged via health manager")


def log_configuration(config: Dict[str, Any]):
    """Log configuration information."""
    logger = get_logger("arbor.config")

    logger.info("Logging configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


# Context manager for temporary log level changes
class LogLevel:
    """Context manager to temporarily change log level."""

    def __init__(self, logger_name: str, level: str):
        self.logger = logging.getLogger(logger_name)
        self.level = getattr(logging, level.upper())
        self.original_level = None

    def __enter__(self):
        self.original_level = self.logger.level
        self.logger.setLevel(self.level)
        return self.logger

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.logger.setLevel(self.original_level)


def apply_uvicorn_formatting():
    """
    Apply our custom formatting to uvicorn loggers after they're set up.
    This runs after the FastAPI app is created to avoid interfering with startup.
    """
    console_formatter = ColoredFormatter("%(name)s - %(levelname)s - %(message)s")

    # List of uvicorn loggers to modify
    uvicorn_loggers = ["uvicorn", "uvicorn.error", "uvicorn.access"]

    for logger_name in uvicorn_loggers:
        logger = logging.getLogger(logger_name)

        # Update formatters for existing handlers
        for handler in logger.handlers:
            if hasattr(handler, "setFormatter"):
                handler.setFormatter(console_formatter)
