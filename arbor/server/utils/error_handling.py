"""Minimal error logging helpers for Arbor.

Example:
    >>> try:
    ...     raise ValueError("boom")
    ... except Exception as exc:
    ...     logged = handle_error(exc, {"job_id": "123"})
    >>> logged.error_type
    'ValueError'
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .logging import get_logger


@dataclass
class LoggedError:
    message: str
    error_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    context: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-friendly representation of the error."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ArborError(Exception):
    """Base Arbor exception carrying context."""

    def __init__(self, message: str, status_code: int = 500, **context: Any):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.context = context


class ValidationError(ArborError):
    pass


class ResourceError(ArborError):
    pass


class ModelError(ArborError):
    pass


class TrainingError(ArborError):
    pass


class ConfigError(ArborError):
    pass


class ErrorHandler:
    """Tiny in-memory error recorder."""

    def __init__(self, max_history: int = 200):
        """Create an error handler that keeps up to ``max_history`` entries."""
        self._logger = get_logger("errors")
        self._max_history = max_history
        self._history: List[LoggedError] = []

    def log_exception(
        self, exc: Exception, context: Optional[Dict[str, Any]] = None
    ) -> LoggedError:
        """Capture an exception, log it, and stash it in memory."""
        ctx = dict(context or {})
        if isinstance(exc, ArborError):
            ctx.update(exc.context)
        message = str(exc) or type(exc).__name__
        error = LoggedError(message, type(exc).__name__, context=ctx)
        log_context = dict(ctx)
        log_context["error_type"] = error.error_type
        self._logger.error(message, context=log_context)
        self._history.append(error)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return error

    def add_error(
        self,
        message: str,
        error_type: str = "Error",
        context: Optional[Dict[str, Any]] = None,
    ) -> LoggedError:
        """Record a manual error message without an exception object."""
        ctx = dict(context or {})
        ctx["error_type"] = error_type
        error = LoggedError(message, error_type, context=ctx)
        self._logger.error(message, context=ctx)
        self._history.append(error)
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history :]
        return error

    def get_recent_errors(self, count: int = 10) -> List[LoggedError]:
        """Return the ``count`` most recent logged errors."""
        return self._history[-count:]

    def get_error_summary(self) -> Dict[str, Any]:
        """Summarize the last 100 errors by type along with the newest entry."""
        recent = self._history[-100:]
        by_type: Dict[str, int] = {}
        for error in recent:
            by_type[error.error_type] = by_type.get(error.error_type, 0) + 1
        most_recent = recent[-1].to_dict() if recent else None
        return {"total": len(recent), "by_type": by_type, "most_recent": most_recent}


error_handler = ErrorHandler()


def handle_error(
    exc: Exception, context: Optional[Dict[str, Any]] = None
) -> LoggedError:
    """Convenience wrapper around :meth:`ErrorHandler.log_exception`."""
    return error_handler.log_exception(exc, context)


def record_error(
    message: str, error_type: str = "Error", context: Optional[Dict[str, Any]] = None
) -> LoggedError:
    """Convenience wrapper to log a manual error entry."""
    return error_handler.add_error(message, error_type, context)
