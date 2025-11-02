"""
Enhanced logging system for Arbor with debugging and observability focus.

This module provides structured logging with context, request tracing, and debugging utilities
designed to make development and troubleshooting as easy as possible.
"""

import json
import logging
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Optional


class ArborFormatter(logging.Formatter):
    """Enhanced formatter with colors, context, and structured data."""

    # ANSI color codes
    COLORS = {
        "DEBUG": "\033[36m",  # Cyan
        "INFO": "\033[32m",  # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",  # Red
        "CRITICAL": "\033[35m",  # Magenta
        "RESET": "\033[0m",  # Reset
        "BOLD": "\033[1m",  # Bold
        "DIM": "\033[2m",  # Dim
    }

    # Component name mappings for cleaner output
    NAME_MAPPINGS = {
        "arbor.server.services.managers.inference_manager": "infer",
        "arbor.server.services.managers.grpo_manager": "grpo",
        "arbor.server.services.managers.file_manager": "files",
        "arbor.server.services.managers.health_manager": "health",
        "arbor.server.services.managers.job_manager": "jobs",
        "arbor.server.services.managers.file_train_manager": "train",
        "arbor.training.sft.trainer": "sft",
        "arbor.training.grpo.trainer": "grpo",
        "arbor.server.services.inference.vllm_client": "vllm",
        "arbor.server.services.inference.vllm_serve": "vllm-srv",
        "arbor.server.services.jobs.inference_job": "inf-job",
        "arbor.server.services.jobs.grpo_job": "grpo-job",
        "arbor.server.services.jobs.file_train_job": "train-job",
        "arbor.server.api.routes.inference": "api-inf",
        "arbor.server.api.routes.grpo": "api-grpo",
        "arbor.server.api.routes.files": "api-files",
        "arbor.server.api.routes.jobs": "api-jobs",
        "arbor.config": "config",
        "arbor.cli": "cli",
        "__main__": "main",
        "uvicorn": "api",
        "uvicorn.access": "api-acc",
        "uvicorn.error": "api-err",
        "fastapi": "api",
    }

    def __init__(self, show_context: bool = True, show_colors: bool = True):
        super().__init__()
        self.show_context = show_context
        self.show_colors = show_colors and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        # Parse structured message if present
        message = record.getMessage()
        context_data = {}

        if " | {" in message and message.endswith("}"):
            try:
                msg_part, json_part = message.rsplit(" | ", 1)
                context_data = json.loads(json_part)
                message = msg_part
            except (ValueError, json.JSONDecodeError):
                pass  # Not structured, use as-is

        # Get short component name
        component = self.NAME_MAPPINGS.get(record.name, record.name)
        if len(component) > 10:
            component = component[:10]

        # Format timestamp
        timestamp = self.formatTime(record, "%H:%M:%S")

        # Apply colors if enabled
        if self.show_colors:
            level_color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            bold = self.COLORS["BOLD"]
            dim = self.COLORS["DIM"]  # noqa: F841
            level = f"{level_color}{record.levelname[:4]}{reset}"
            component = f"{bold}{component}{reset}"
        else:
            level = record.levelname[:4]
        parts = [timestamp, f"[{level}]", f"[{component}]", message]

        # Add structured data as separate lines in debug mode
        if record.levelno == logging.DEBUG and context_data:
            formatted_context = json.dumps(context_data, indent=2)
            parts.append(f"\n  Context: {formatted_context}")

        return " ".join(parts)


class ArborLogger:
    """Enhanced logger with structured logging and context support."""

    def __init__(self, name: str):
        self.name = name
        # Get short name for cleaner output
        self.short_name = ArborFormatter.NAME_MAPPINGS.get(name, name)
        self._logger = logging.getLogger(self.short_name)

    def _log_with_context(
        self,
        level: int,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        exc_info: Optional[bool] = None,
        **kwargs,
    ):
        """Log with context and structured data."""
        if args:
            try:
                message = message % args
            except TypeError:
                message = message.format(*args)
        # Merge context and kwargs
        full_context: dict[str, Any] = {}
        if context is not None:
            if isinstance(context, Mapping):
                full_context.update(context)
            else:
                full_context["context"] = context
        if kwargs:
            full_context.update(kwargs)

        # Build structured message
        if full_context:
            structured_message = (
                f"{message} | {json.dumps(full_context, separators=(',', ':'))}"
            )
        else:
            structured_message = message

        self._logger.log(level, structured_message, exc_info=exc_info)

    def debug(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, *args, context=context, **kwargs)

    def info(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, *args, context=context, **kwargs)

    def warning(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Log warning message with context."""
        self._log_with_context(
            logging.WARNING, message, *args, context=context, **kwargs
        )

    def error(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs,
    ):
        """Log error message with context."""
        self._log_with_context(
            logging.ERROR,
            message,
            *args,
            context=context,
            exc_info=exc_info,
            **kwargs,
        )

    def critical(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs,
    ):
        """Log critical message with context."""
        self._log_with_context(
            logging.CRITICAL,
            message,
            *args,
            context=context,
            exc_info=exc_info,
            **kwargs,
        )

    def exception(
        self,
        message: str,
        *args,
        context: Optional[dict[str, Any]] = None,
        **kwargs,
    ):
        """Log exception with context and stack trace."""
        self._log_with_context(
            logging.ERROR,
            message,
            *args,
            context=context,
            exc_info=True,
            **kwargs,
        )


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    show_context: bool = True,
    show_colors: bool = True,
    debug_mode: bool = False,
) -> dict[str, Any]:
    """
    Setup enhanced logging for Arbor.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files
        enable_file_logging: Whether to enable file logging
        enable_console_logging: Whether to enable console logging
        show_context: Whether to show context in console output
        show_colors: Whether to use colors in console output
        debug_mode: Enable debug mode with extra verbose logging

    Returns:
        Dictionary with logging configuration details
    """
    import os
    import tempfile

    # Use a persistent log directory during tests to avoid cleanup issues
    if "PYTEST_CURRENT_TEST" in os.environ or "pytest" in sys.modules:
        if log_dir is None or (log_dir and "/tmp" in str(log_dir)):
            # Use a stable temp directory that won't get cleaned up during tests
            stable_temp_dir = Path(tempfile.gettempdir()) / "arbor_test_logs"
            stable_temp_dir.mkdir(exist_ok=True)
            log_dir = stable_temp_dir

    # Set debug level if debug mode is enabled
    if debug_mode:
        log_level = "DEBUG"

    # Create formatters
    console_formatter = ArborFormatter(
        show_context=show_context, show_colors=show_colors
    )
    file_formatter = ArborFormatter(show_context=True, show_colors=False)

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
        file_handler.setLevel(logging.DEBUG)
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

        # Debug log file (debug mode only)
        if debug_mode:
            debug_log_file = log_dir / "arbor_debug.log"
            debug_handler = logging.FileHandler(debug_log_file)
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(file_formatter)
            root_logger.addHandler(debug_handler)
            handlers.append("debug_file")

    # Configure third-party loggers
    _configure_third_party_loggers(debug_mode)

    return {
        "level": log_level,
        "handlers": handlers,
        "log_dir": str(log_dir) if log_dir else None,
        "debug_mode": debug_mode,
        "show_context": show_context,
        "show_colors": show_colors,
    }


def _configure_third_party_loggers(debug_mode: bool = False):
    """Configure third-party library loggers."""

    # Set levels for third-party libraries
    third_party_levels = {
        "uvicorn": "INFO" if debug_mode else "WARNING",
        "uvicorn.access": "INFO" if debug_mode else "WARNING",
        "uvicorn.error": "INFO" if debug_mode else "WARNING",
        "fastapi": "INFO" if debug_mode else "WARNING",
        "httpx": "INFO" if debug_mode else "WARNING",
        "urllib3": "INFO" if debug_mode else "WARNING",
        "vllm": "INFO",
        "torch": "WARNING",
        "transformers": "WARNING",
        "accelerate": "WARNING",
        "datasets": "WARNING",
        "trl": "INFO" if debug_mode else "WARNING",
    }

    for logger_name, level in third_party_levels.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, level))


def get_logger(name: str) -> ArborLogger:
    """
    Get an enhanced Arbor logger.

    Args:
        name: Logger name, typically __name__

    Returns:
        Enhanced ArborLogger instance
    """
    return ArborLogger(name)


def log_configuration(config: dict[str, Any]):
    """Log configuration information."""
    logger = get_logger("arbor.config")
    logger.info("Configuration loaded", config=config)
