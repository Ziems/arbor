"""Minimal logging utilities for Arbor.

This module keeps a tiny abstraction layer around the standard :mod:`logging`
package.  It offers a small formatter that can display context stored in
``ContextVar`` instances, a lightweight wrapper class that accepts keyword
arguments as contextual data, and a couple of helpers that existing code relies
on (request/job contexts and a few decorators).

Usage
-----
Create module-level loggers by importing :func:`get_logger` and calling it
with the module's ``__name__``.  The helper returns :class:`ArborLogger`, which
wraps the standard :class:`logging.Logger` while adding optional
keyword-context support::

    from arbor.server.utils.logging import get_logger

    logger = get_logger(__name__)

    def handler(event: dict[str, object]) -> None:
        logger.info(
            "Processing event",
            event_type=event.get("type"),
        )

"""

from __future__ import annotations

import json
import logging
import sys
import time
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Callable, Dict, Optional

request_id_context: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
job_id_context: ContextVar[Optional[str]] = ContextVar("job_id", default=None)
user_context: ContextVar[Optional[str]] = ContextVar("user", default=None)
operation_context: ContextVar[Optional[str]] = ContextVar("operation", default=None)


# ---------------------------------------------------------------------------
# Formatting and logger wrapper
# ---------------------------------------------------------------------------

class ArborFormatter(logging.Formatter):
    """Very small formatter that optionally appends context information."""

    def __init__(self, *, show_context: bool = True) -> None:
        super().__init__(fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
        self.show_context = show_context

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401
        """Format the record and append context stored in :mod:`contextvars`."""

        message = super().format(record)
        if not self.show_context:
            return message

        context_parts = []
        for label, ctx in (
            ("request", request_id_context),
            ("job", job_id_context),
            ("user", user_context),
            ("operation", operation_context),
        ):
            value = ctx.get()
            if value:
                context_parts.append(f"{label}={value}")

        if context_parts:
            return f"{message} ({', '.join(context_parts)})"
        return message


def _encode_context(data: Dict[str, Any]) -> str:
    """Serialize contextual data in a predictable manner."""

    if not data:
        return ""

    try:
        return json.dumps(data, separators=(",", ":"), sort_keys=True, default=str)
    except TypeError:
        # Fall back to a very small manual conversion – every value becomes a
        # string representation.  This keeps logging robust even if the caller
        # provides complex objects.
        safe_data = {key: str(value) for key, value in data.items()}
        return json.dumps(safe_data, separators=(",", ":"), sort_keys=True)


class ArborLogger:
    """Thin wrapper around :class:`logging.Logger` accepting context kwargs."""

    def __init__(self, name: str) -> None:
        self._logger = logging.getLogger(name)

    def setLevel(self, level: Union[int, str]) -> None:
        """Proxy ``setLevel`` to the wrapped :class:`logging.Logger`."""

        self._logger.setLevel(level)

    def _log(
        self,
        level: int,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        exc_info: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        payload: Dict[str, Any] = {}
        if context:
            payload.update(context)
        if kwargs:
            payload.update(kwargs)

        if payload:
            message = f"{message} | {_encode_context(payload)}"

        self._logger.log(level, message, exc_info=exc_info)

    def debug(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._log(logging.DEBUG, message, context, **kwargs)

    def info(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._log(logging.INFO, message, context, **kwargs)

    def warning(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._log(logging.WARNING, message, context, **kwargs)

    def error(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        self._log(logging.ERROR, message, context, exc_info=exc_info, **kwargs)

    def critical(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        *,
        exc_info: bool = False,
        **kwargs: Any,
    ) -> None:
        self._log(logging.CRITICAL, message, context, exc_info=exc_info, **kwargs)

    def exception(self, message: str, context: Optional[Dict[str, Any]] = None, **kwargs: Any) -> None:
        self._log(logging.ERROR, message, context, exc_info=True, **kwargs)


def get_logger(name: str) -> ArborLogger:
    """Return an :class:`ArborLogger` for ``name``."""

    return ArborLogger(name)


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


class _BaseContext:
    """Utility base class handling ContextVar tokens."""

    def __init__(self) -> None:
        self._tokens: Dict[ContextVar[Any], Any] = {}

    def _set(self, var: ContextVar[Any], value: Any) -> None:
        self._tokens[var] = var.set(value)

    def _reset(self) -> None:
        for var, token in self._tokens.items():
            var.reset(token)
        self._tokens.clear()


class RequestContext(_BaseContext):
    """Track request details for the duration of a ``with`` block.

    Examples
    --------
    >>> from arbor.server.utils.logging import RequestContext, get_logger
    >>> with RequestContext("req-123", user="alice", operation="predict"):
    ...     get_logger("api").info("Handled request")
    """

    def __init__(
        self,
        request_id: Optional[str] = None,
        *,
        user: Optional[str] = None,
        operation: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.request_id = request_id or f"req-{int(time.time() * 1000):x}"
        self.user = user
        self.operation = operation
        self._start = 0.0

    def __enter__(self) -> "RequestContext":
        self._start = time.time()
        self._set(request_id_context, self.request_id)
        if self.user is not None:
            self._set(user_context, self.user)
        if self.operation is not None:
            self._set(operation_context, self.operation)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        duration_ms = round((time.time() - self._start) * 1000, 2)
        logger = get_logger("request")

        if exc_type is None:
            logger.info(
                "Request completed",
                duration_ms=duration_ms,
                success=True,
                operation=self.operation,
            )
        else:
            logger.error(
                "Request failed",
                duration_ms=duration_ms,
                success=False,
                operation=self.operation,
                error_type=exc_type.__name__,
                error_message=str(exc) if exc else None,
            )

        self._reset()


class JobContext(_BaseContext):
    """Context helper for job execution blocks.

    Examples
    --------
    >>> from arbor.server.utils.logging import JobContext
    >>> with JobContext("job-42", job_type="batch", model="gpt-x"):
    ...     ...  # execute the job body
    """

    def __init__(self, job_id: str, *, job_type: Optional[str] = None, model: Optional[str] = None) -> None:
        super().__init__()
        self.job_id = job_id
        self.job_type = job_type
        self.model = model
        self._start = 0.0

    def __enter__(self) -> "JobContext":
        self._start = time.time()
        self._set(job_id_context, self.job_id)
        logger = get_logger("job")
        logger.info("Job started", job_type=self.job_type, model=self.model)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401
        duration_ms = round((time.time() - self._start) * 1000, 2)
        logger = get_logger("job")

        if exc_type is None:
            logger.info("Job completed", duration_ms=duration_ms, success=True)
        else:
            logger.error(
                "Job failed",
                duration_ms=duration_ms,
                success=False,
                error_type=exc_type.__name__,
                error_message=str(exc) if exc else None,
            )

        self._reset()


# ---------------------------------------------------------------------------
# Decorators and helpers
# ---------------------------------------------------------------------------


def log_function_call(include_args: bool = False, include_result: bool = False) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that logs start/finish information for a function call.

    Examples
    --------
    >>> from arbor.server.utils.logging import log_function_call
    >>> @log_function_call(include_args=True)
    ... def add(x, y):
    ...     return x + y
    >>> add(1, 2)
    3
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = get_logger(func.__module__)
            context: Dict[str, Any] = {"function": f"{func.__module__}.{func.__name__}"}

            if include_args:
                context["args_count"] = len(args)
                context["kwargs_keys"] = list(kwargs.keys())

            logger.debug(f"Calling {func.__name__}", context)
            start = time.time()

            try:
                result = func(*args, **kwargs)
            except Exception as exc:  # pragma: no cover - defensive path
                duration_ms = round((time.time() - start) * 1000, 2)
                logger.error(
                    f"Failed {func.__name__}",
                    function=context["function"],
                    duration_ms=duration_ms,
                    error_type=type(exc).__name__,
                    exc_info=True,
                )
                raise

            duration_ms = round((time.time() - start) * 1000, 2)
            result_context: Dict[str, Any] = {"function": context["function"], "duration_ms": duration_ms, "success": True}
            if include_result and result is not None:
                result_context["result_type"] = type(result).__name__

            logger.debug(f"Completed {func.__name__}", result_context)
            return result

        return wrapper

    return decorator


def log_slow_operations(threshold_ms: float = 1000.0) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator logging a warning when an operation exceeds ``threshold_ms``.

    Examples
    --------
    >>> from arbor.server.utils.logging import log_slow_operations
    >>> @log_slow_operations(threshold_ms=5)
    ... def expensive_call():
    ...     return "done"
    >>> expensive_call()
    'done'
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.time()
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start) * 1000

            if duration_ms > threshold_ms:
                logger = get_logger(func.__module__)
                logger.warning(
                    f"Slow operation: {func.__name__}",
                    function=f"{func.__module__}.{func.__name__}",
                    duration_ms=round(duration_ms, 2),
                    threshold_ms=threshold_ms,
                )

            return result

        return wrapper

    return decorator


def debug_checkpoint(message: str, **context: Any) -> None:
    """Log a debug checkpoint with optional context data.

    Examples
    --------
    >>> from arbor.server.utils.logging import debug_checkpoint
    >>> debug_checkpoint("finished parsing", records=120)
    """

    logger = get_logger("debug.checkpoint")
    logger.debug(f"Checkpoint: {message}", context)


def debug_timing(operation: str):
    """Context manager recording the duration of ``operation``.

    Examples
    --------
    >>> from arbor.server.utils.logging import debug_timing
    >>> with debug_timing("load-model"):
    ...     ...  # code being timed
    """

    class TimingContext:
        def __init__(self, operation_name: str) -> None:
            self.operation = operation_name
            self.start = 0.0
            self.logger = get_logger("debug.timing")

        def __enter__(self) -> "TimingContext":
            self.start = time.time()
            self.logger.debug(f"Started: {self.operation}")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            duration_ms = round((time.time() - self.start) * 1000, 2)
            if exc_type is None:
                self.logger.debug("Completed operation", operation=self.operation, duration_ms=duration_ms, success=True)
            else:
                self.logger.error(
                    "Failed operation",
                    operation=self.operation,
                    duration_ms=duration_ms,
                    success=False,
                    error_type=exc_type.__name__,
                )

    return TimingContext(operation)


# ---------------------------------------------------------------------------
# Setup helpers
# ---------------------------------------------------------------------------


def _configure_third_party_loggers(debug_mode: bool) -> None:
    """Keep third-party loggers quiet unless debug mode is active."""

    level = logging.DEBUG if debug_mode else logging.INFO
    for name in ("uvicorn", "fastapi", "httpx", "urllib3"):
        logging.getLogger(name).setLevel(level)


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[Path] = None,
    *,
    enable_file_logging: bool = True,
    enable_console_logging: bool = True,
    show_context: bool = True,
    show_colors: bool = True,  # kept for signature compatibility, unused now
    debug_mode: bool = False,
) -> Dict[str, Any]:
    """Configure the root logger with a tiny amount of ceremony.

    Examples
    --------
    >>> from pathlib import Path
    >>> from arbor.server.utils.logging import setup_logging, get_logger
    >>> setup_logging(log_level="DEBUG", log_dir=Path("./logs"))
    {'level': 'DEBUG', 'handlers': ['console', 'file'], ...}
    >>> get_logger("example").info("Ready")
    """

    if debug_mode:
        log_level = "DEBUG"

    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric_level)
    root.handlers.clear()

    formatter = ArborFormatter(show_context=show_context)
    handlers: list[str] = []

    if enable_console_logging:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root.addHandler(console_handler)
        handlers.append("console")

    if enable_file_logging and log_dir:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path / "arbor.log")
        file_handler.setLevel(logging.DEBUG if debug_mode else numeric_level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        handlers.append("file")

    _configure_third_party_loggers(debug_mode)

    return {
        "level": log_level.upper(),
        "handlers": handlers,
        "log_dir": str(log_dir) if log_dir else None,
        "debug_mode": debug_mode,
        "show_context": show_context,
        "show_colors": show_colors,
    }


def apply_uvicorn_formatting() -> None:  # pragma: no cover - retained for API compatibility
    """Legacy helper kept for compatibility with previous imports."""


def log_system_info() -> None:
    """Log a short startup banner.

    Examples
    --------
    >>> from arbor.server.utils.logging import log_system_info
    >>> log_system_info()
    """

    logger = get_logger("arbor.startup")
    logger.info("=" * 60)
    logger.info("ARBOR SYSTEM STARTUP")
    logger.info("=" * 60)


def log_configuration(config: Dict[str, Any]) -> None:
    """Log the provided configuration mapping.

    Examples
    --------
    >>> from arbor.server.utils.logging import log_configuration
    >>> log_configuration({"model": "gpt-x", "temperature": 0.1})
    """

    logger = get_logger("arbor.config")
    logger.info("Configuration loaded", config=config)

