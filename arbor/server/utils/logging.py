"""
Enhanced logging system for Arbor with debugging and observability focus.

This module provides structured logging with context, request tracing, and debugging utilities
designed to make development and troubleshooting as easy as possible.
"""

import json
import logging
import logging.config
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Optional

# Context variables for tracking across async operations
request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
job_id_context: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
user_context: ContextVar[Optional[str]] = ContextVar("user", default=None)
operation_context: ContextVar[Optional[str]] = ContextVar("operation", default=None)


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
        "arbor.server.services.comms.comms": "comms",
        "arbor.server.services.scripts.sft_training": "sft",
        "arbor.server.services.scripts.grpo_training": "grpo",
        "arbor.server.services.scripts.mmgrpo_training": "mmgrpo",
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

    def format(self, record):
        # Get context information
        request_id = request_id_context.get()
        job_id = job_id_context.get()
        user = user_context.get()
        operation = operation_context.get()

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

        # Build context string
        context_parts = []
        if request_id:
            context_parts.append(f"req:{request_id}")
        if job_id:
            context_parts.append(f"job:{job_id}")
        if user:
            context_parts.append(f"user:{user}")
        if operation:
            context_parts.append(f"op:{operation}")

        context_str = f"[{','.join(context_parts)}]" if context_parts else ""

        # Add important context data to context string
        if context_data:
            important_keys = [
                "step",
                "progress_pct",
                "duration_ms",
                "error_type",
                "function",
            ]
            for key in important_keys:
                if key in context_data:
                    value = context_data[key]
                    if key == "duration_ms" and isinstance(value, (int, float)):
                        context_parts.append(f"{value}ms")
                    elif key == "step":
                        context_parts.append(f"step:{value}")
                    elif key == "progress_pct":
                        context_parts.append(f"{value}%")
                    elif key == "error_type":
                        context_parts.append(f"err:{value}")
                    elif key == "function":
                        func_name = value.split(".")[-1] if "." in value else value
                        context_parts.append(f"fn:{func_name}")

        if context_parts and not context_str:
            context_str = f"[{','.join(context_parts)}]"
        elif context_parts:
            additional = ",".join(
                context_parts[len([p for p in context_str[1:-1].split(",") if p]) :]
            )
            if additional:
                context_str = context_str[:-1] + f",{additional}]"

        # Apply colors if enabled
        if self.show_colors:
            level_color = self.COLORS.get(record.levelname, "")
            reset = self.COLORS["RESET"]
            bold = self.COLORS["BOLD"]
            dim = self.COLORS["DIM"]

            # Color the level
            level = f"{level_color}{record.levelname[:4]}{reset}"

            # Color the component
            component = f"{bold}{component}{reset}"

            # Dim the context
            if context_str:
                context_str = f"{dim}{context_str}{reset}"
        else:
            level = record.levelname[:4]

        # Build final message
        parts = [timestamp, f"[{level}]", f"[{component}]"]
        if context_str and self.show_context:
            parts.append(context_str)
        parts.append(message)

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
        context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[bool] = None,
        **kwargs,
    ):
        """Log with context and structured data."""
        # Merge context and kwargs
        full_context = {}
        if context:
            full_context.update(context)
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

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, context, **kwargs)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, context, **kwargs)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, context, **kwargs)

    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs,
    ):
        """Log error message with context."""
        self._log_with_context(
            logging.ERROR, message, context, exc_info=exc_info, **kwargs
        )

    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: bool = False,
        **kwargs,
    ):
        """Log critical message with context."""
        self._log_with_context(
            logging.CRITICAL, message, context, exc_info=exc_info, **kwargs
        )

    def exception(
        self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs
    ):
        """Log exception with context and stack trace."""
        self._log_with_context(logging.ERROR, message, context, exc_info=True, **kwargs)


class RequestContext:
    """Context manager for request-level debugging."""

    def __init__(
        self,
        request_id: Optional[str] = None,
        user: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        self.request_id = request_id or f"req-{str(uuid.uuid4())[:8]}"
        self.user = user
        self.operation = operation
        self.start_time = time.time()

        # Store previous values
        self._prev_request_id = None
        self._prev_user = None
        self._prev_operation = None

    def __enter__(self):
        # Store previous context
        self._prev_request_id = request_id_context.get()
        self._prev_user = user_context.get()
        self._prev_operation = operation_context.get()

        # Set new context
        request_id_context.set(self.request_id)
        if self.user:
            user_context.set(self.user)
        if self.operation:
            operation_context.set(self.operation)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = time.time() - self.start_time

        # Log request completion
        logger = get_logger("request")
        success = exc_type is None

        if success:
            logger.info(
                f"Request completed: {self.operation or 'unknown'}",
                duration_ms=round(duration * 1000, 2),
                success=True,
            )
        else:
            logger.error(
                f"Request failed: {self.operation or 'unknown'}",
                duration_ms=round(duration * 1000, 2),
                success=False,
                error_type=type(exc_val).__name__ if exc_val else None,
                error_message=str(exc_val) if exc_val else None,
            )

        # Restore previous context
        request_id_context.set(self._prev_request_id)
        user_context.set(self._prev_user)
        operation_context.set(self._prev_operation)


class JobContext:
    """Context manager for job-level debugging."""

    def __init__(
        self, job_id: str, job_type: Optional[str] = None, model: Optional[str] = None
    ):
        self.job_id = job_id
        self.job_type = job_type
        self.model = model
        self.start_time = time.time()

        # Store previous value
        self._prev_job_id = None

    def __enter__(self):
        # Store previous context
        self._prev_job_id = job_id_context.get()

        # Set new context
        job_id_context.set(self.job_id)

        # Log job start
        logger = get_logger("job")
        logger.info(
            f"Job started: {self.job_type or 'unknown'}",
            job_type=self.job_type,
            model=self.model,
        )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Calculate duration
        duration = time.time() - self.start_time

        # Log job completion
        logger = get_logger("job")
        success = exc_type is None

        if success:
            logger.info(
                f"Job completed: {self.job_type or 'unknown'}",
                duration_ms=round(duration * 1000, 2),
                success=True,
            )
        else:
            logger.error(
                f"Job failed: {self.job_type or 'unknown'}",
                duration_ms=round(duration * 1000, 2),
                success=False,
                error_type=type(exc_val).__name__ if exc_val else None,
                error_message=str(exc_val) if exc_val else None,
            )

        # Restore previous context
        job_id_context.set(self._prev_job_id)


def log_function_call(include_args: bool = False, include_result: bool = False):
    """Decorator to log function calls with timing."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            func_name = func.__name__
            start_time = time.time()

            # Build context
            context = {"function": f"{func.__module__}.{func_name}"}
            if include_args:
                context["args_count"] = len(args)
                context["kwargs_keys"] = list(kwargs.keys())

            logger.debug(f"Calling {func_name}", context)

            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                result_context = {
                    "function": f"{func.__module__}.{func_name}",
                    "duration_ms": round(duration * 1000, 2),
                    "success": True,
                }

                if include_result and result is not None:
                    result_context["result_type"] = type(result).__name__

                logger.debug(f"Completed {func_name}", result_context)
                return result

            except Exception as e:
                duration = time.time() - start_time

                logger.error(
                    f"Failed {func_name}: {str(e)}",
                    function=f"{func.__module__}.{func_name}",
                    duration_ms=round(duration * 1000, 2),
                    error_type=type(e).__name__,
                    success=False,
                    exc_info=True,
                )
                raise

        return wrapper

    return decorator


def log_slow_operations(threshold_ms: float = 1000.0):
    """Decorator to log operations that exceed a time threshold."""

    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            duration_ms = duration * 1000

            if duration_ms > threshold_ms:
                logger = get_logger(func.__module__)
                logger.warning(
                    f"Slow operation: {func.__name__}",
                    function=f"{func.__module__}.{func.__name__}",
                    duration_ms=round(duration_ms, 2),
                    threshold_ms=threshold_ms,
                    performance_warning=True,
                )

            return result

        return wrapper

    return decorator


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    show_context: bool = True,
    show_colors: bool = True,
    debug_mode: bool = False,
) -> Dict[str, Any]:
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


# Convenience functions for common debugging patterns
def debug_checkpoint(message: str, **context):
    """Log a debug checkpoint with context."""
    logger = get_logger("debug.checkpoint")
    logger.debug(f"Checkpoint: {message}", context)


def debug_timing(operation: str):
    """Context manager to time operations and log results."""

    class TimingContext:
        def __init__(self, operation: str):
            self.operation = operation
            self.start_time = None
            self.logger = get_logger("debug.timing")

        def __enter__(self):
            self.start_time = time.time()
            self.logger.debug(f"Started: {self.operation}")
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            duration = time.time() - self.start_time
            success = exc_type is None

            context = {
                "operation": self.operation,
                "duration_ms": round(duration * 1000, 2),
                "success": success,
            }

            if not success:
                context["error_type"] = type(exc_val).__name__ if exc_val else None

            if success:
                self.logger.debug(f"Completed: {self.operation}", context)
            else:
                self.logger.error(f"Failed: {self.operation}", context)

    return TimingContext(operation)


# Legacy compatibility
def apply_uvicorn_formatting():
    """Compatibility function - no longer needed with new formatter."""
    pass


def log_system_info():
    """Log system startup information."""
    logger = get_logger("arbor.startup")
    logger.info("=" * 60)
    logger.info("ARBOR SYSTEM STARTUP")
    logger.info("=" * 60)


def log_configuration(config: Dict[str, Any]):
    """Log configuration information."""
    logger = get_logger("arbor.config")
    logger.info("Configuration loaded", config=config)
