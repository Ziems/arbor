"""
Centralized error handling system for Arbor.

Provides structured error handling with context, automatic logging, and debugging support.
"""

import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from .logging import get_logger, job_id_context, request_id_context, user_context


class ErrorCategory(Enum):
    """Categories of errors for better organization."""

    VALIDATION = "validation"
    RESOURCE = "resource"
    PERMISSION = "permission"
    NETWORK = "network"
    GPU = "gpu"
    MODEL = "model"
    TRAINING = "training"
    INFERENCE = "inference"
    FILE_SYSTEM = "file_system"
    CONFIG = "config"
    INTERNAL = "internal"
    EXTERNAL = "external"


class ErrorSeverity(Enum):
    """Error severity levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Structured error context information."""

    request_id: Optional[str] = None
    job_id: Optional[str] = None
    user: Optional[str] = None
    operation: Optional[str] = None
    component: Optional[str] = None
    model: Optional[str] = None
    file_path: Optional[str] = None
    function_name: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary, excluding None values."""
        data = asdict(self)
        return {k: v for k, v in data.items() if v is not None}


@dataclass
class ArborError:
    """Structured error information."""

    error_id: str
    message: str
    category: ErrorCategory
    severity: ErrorSeverity
    error_type: str
    context: ErrorContext
    timestamp: datetime
    traceback_str: Optional[str] = None
    suggestions: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["category"] = self.category.value
        data["severity"] = self.severity.value
        data["timestamp"] = self.timestamp.isoformat()
        return data


class ArborException(Exception):
    """Base exception class with enhanced error handling."""

    def __init__(
        self,
        message: str,
        category: ErrorCategory = ErrorCategory.INTERNAL,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Optional[ErrorContext] = None,
        suggestions: Optional[List[str]] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.message = message
        self.category = category
        self.severity = severity
        self.context = context or ErrorContext()
        self.suggestions = suggestions or []
        self.cause = cause

        # Auto-populate context from current state
        self._populate_context()

        # Generate error ID
        self.error_id = self._generate_error_id()

        # Log the error
        self._log_error()

    def _populate_context(self):
        """Auto-populate context from current execution state."""
        if not self.context.request_id:
            self.context.request_id = request_id_context.get()
        if not self.context.job_id:
            self.context.job_id = job_id_context.get()
        if not self.context.user:
            self.context.user = user_context.get()

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        import uuid

        return f"err-{str(uuid.uuid4())[:8]}"

    def _log_error(self):
        """Log the error with full context."""
        logger = get_logger("error")

        # Build log context
        log_context = self.context.to_dict()
        log_context.update(
            {
                "error_id": self.error_id,
                "category": self.category.value,
                "severity": self.severity.value,
                "suggestions": self.suggestions,
            }
        )

        if self.cause:
            log_context["caused_by"] = {
                "type": type(self.cause).__name__,
                "message": str(self.cause),
            }

        # Log at appropriate level based on severity
        if self.severity == ErrorSeverity.CRITICAL:
            logger.critical(
                f"Critical error: {self.message}", log_context, exc_info=True
            )
        elif self.severity == ErrorSeverity.HIGH:
            logger.error(
                f"High severity error: {self.message}", log_context, exc_info=True
            )
        elif self.severity == ErrorSeverity.MEDIUM:
            logger.error(f"Error: {self.message}", log_context)
        else:
            logger.warning(f"Low severity error: {self.message}", log_context)

    def to_arbor_error(self) -> ArborError:
        """Convert to ArborError structure."""
        return ArborError(
            error_id=self.error_id,
            message=self.message,
            category=self.category,
            severity=self.severity,
            error_type=type(self).__name__,
            context=self.context,
            timestamp=datetime.now(),
            traceback_str=traceback.format_exc(),
            suggestions=self.suggestions,
        )


# Specific exception types for different error categories
class ValidationError(ArborException):
    """Validation errors with helpful suggestions."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        **kwargs,
    ):
        context = kwargs.get("context", ErrorContext())
        if field:
            context.additional_data = context.additional_data or {}
            context.additional_data["field"] = field
            context.additional_data["value"] = str(value) if value is not None else None

        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            severity=ErrorSeverity.MEDIUM,
            context=context,
            **kwargs,
        )


class ResourceError(ArborException):
    """Resource-related errors (GPU, memory, disk, etc.)."""

    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        context = kwargs.get("context", ErrorContext())
        if resource_type:
            context.additional_data = context.additional_data or {}
            context.additional_data["resource_type"] = resource_type

        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class ModelError(ArborException):
    """Model-related errors."""

    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        context = kwargs.get("context", ErrorContext())
        if model_name:
            context.model = model_name

        super().__init__(
            message,
            category=ErrorCategory.MODEL,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class TrainingError(ArborException):
    """Training-related errors."""

    def __init__(self, message: str, step: Optional[int] = None, **kwargs):
        context = kwargs.get("context", ErrorContext())
        if step is not None:
            context.additional_data = context.additional_data or {}
            context.additional_data["training_step"] = step

        super().__init__(
            message,
            category=ErrorCategory.TRAINING,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class InferenceError(ArborException):
    """Inference-related errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INFERENCE,
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class ConfigError(ArborException):
    """Configuration-related errors."""

    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        context = kwargs.get("context", ErrorContext())
        if config_key:
            context.additional_data = context.additional_data or {}
            context.additional_data["config_key"] = config_key

        super().__init__(
            message,
            category=ErrorCategory.CONFIG,
            severity=ErrorSeverity.HIGH,
            context=context,
            **kwargs,
        )


class ErrorHandler:
    """Centralized error handler with context and recovery suggestions."""

    def __init__(self):
        self.logger = get_logger("error_handler")
        self.error_history: List[ArborError] = []
        self.max_history = 1000

    def handle_exception(
        self,
        exc: Exception,
        context: Optional[ErrorContext] = None,
        operation: Optional[str] = None,
        suggestions: Optional[List[str]] = None,
    ) -> ArborError:
        """Handle any exception and convert to ArborError."""

        # If it's already an ArborException, just get the ArborError
        if isinstance(exc, ArborException):
            arbor_error = exc.to_arbor_error()
        else:
            # Convert generic exception to ArborError
            error_context = context or ErrorContext()
            if operation:
                error_context.operation = operation

            # Auto-populate context
            if not error_context.request_id:
                error_context.request_id = request_id_context.get()
            if not error_context.job_id:
                error_context.job_id = job_id_context.get()
            if not error_context.user:
                error_context.user = user_context.get()

            # Determine category and severity based on exception type
            category, severity = self._categorize_exception(exc)

            arbor_error = ArborError(
                error_id=f"err-{str(__import__('uuid').uuid4())[:8]}",
                message=str(exc),
                category=category,
                severity=severity,
                error_type=type(exc).__name__,
                context=error_context,
                timestamp=datetime.now(),
                traceback_str=traceback.format_exc(),
                suggestions=suggestions or self._generate_suggestions(exc),
            )

            # Log the error
            self._log_arbor_error(arbor_error)

        # Add to history
        self._add_to_history(arbor_error)

        return arbor_error

    def _categorize_exception(
        self, exc: Exception
    ) -> tuple[ErrorCategory, ErrorSeverity]:
        """Automatically categorize exceptions."""
        exc_type = type(exc).__name__.lower()

        # Categorization rules
        if any(keyword in exc_type for keyword in ["validation", "value", "type"]):
            return ErrorCategory.VALIDATION, ErrorSeverity.MEDIUM
        elif any(
            keyword in exc_type for keyword in ["memory", "resource", "gpu", "cuda"]
        ):
            return ErrorCategory.RESOURCE, ErrorSeverity.HIGH
        elif any(keyword in exc_type for keyword in ["permission", "access", "auth"]):
            return ErrorCategory.PERMISSION, ErrorSeverity.MEDIUM
        elif any(
            keyword in exc_type for keyword in ["network", "connection", "timeout"]
        ):
            return ErrorCategory.NETWORK, ErrorSeverity.MEDIUM
        elif any(keyword in exc_type for keyword in ["file", "io", "path"]):
            return ErrorCategory.FILE_SYSTEM, ErrorSeverity.MEDIUM
        elif any(keyword in exc_type for keyword in ["config", "setting"]):
            return ErrorCategory.CONFIG, ErrorSeverity.HIGH
        else:
            return ErrorCategory.INTERNAL, ErrorSeverity.MEDIUM

    def _generate_suggestions(self, exc: Exception) -> List[str]:
        """Generate helpful suggestions based on exception type."""
        exc_type = type(exc).__name__.lower()
        message = str(exc).lower()

        suggestions = []

        # Common suggestions based on exception type
        if "filenotfound" in exc_type or "no such file" in message:
            suggestions.extend(
                [
                    "Check if the file path is correct",
                    "Verify the file exists and is readable",
                    "Check file permissions",
                ]
            )
        elif "permission" in exc_type or "access denied" in message:
            suggestions.extend(
                [
                    "Check file/directory permissions",
                    "Run with appropriate user privileges",
                    "Verify the path is accessible",
                ]
            )
        elif "memory" in message or "out of memory" in message:
            suggestions.extend(
                [
                    "Reduce batch size or model size",
                    "Check available system memory",
                    "Consider using GPU memory optimization",
                ]
            )
        elif "cuda" in message or "gpu" in message:
            suggestions.extend(
                [
                    "Check if CUDA is available and properly installed",
                    "Verify GPU driver version",
                    "Try running without GPU (CPU mode)",
                ]
            )
        elif "connection" in message or "timeout" in message:
            suggestions.extend(
                [
                    "Check network connectivity",
                    "Verify service is running and accessible",
                    "Increase timeout if appropriate",
                ]
            )
        elif "model" in message:
            suggestions.extend(
                [
                    "Verify the model name is correct",
                    "Check if model is downloaded/available",
                    "Try a different model or check model compatibility",
                ]
            )

        return suggestions

    def _log_arbor_error(self, error: ArborError):
        """Log an ArborError with appropriate level."""
        context = error.context.to_dict()
        context.update(
            {
                "error_id": error.error_id,
                "category": error.category.value,
                "suggestions": error.suggestions,
            }
        )

        if error.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"Critical: {error.message}", context)
        elif error.severity == ErrorSeverity.HIGH:
            self.logger.error(f"High severity: {error.message}", context)
        elif error.severity == ErrorSeverity.MEDIUM:
            self.logger.error(f"Error: {error.message}", context)
        else:
            self.logger.warning(f"Warning: {error.message}", context)

    def _add_to_history(self, error: ArborError):
        """Add error to history, maintaining max size."""
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history = self.error_history[-self.max_history :]

    def get_recent_errors(self, count: int = 10) -> List[ArborError]:
        """Get recent errors."""
        return self.error_history[-count:]

    def get_errors_by_category(self, category: ErrorCategory) -> List[ArborError]:
        """Get errors by category."""
        return [e for e in self.error_history if e.category == category]

    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of recent errors."""
        if not self.error_history:
            return {"total": 0, "by_category": {}, "by_severity": {}}

        recent_errors = self.error_history[-100:]  # Last 100 errors

        category_counts = {}
        severity_counts = {}

        for error in recent_errors:
            # Count by category
            cat = error.category.value
            category_counts[cat] = category_counts.get(cat, 0) + 1

            # Count by severity
            sev = error.severity.value
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        return {
            "total": len(recent_errors),
            "by_category": category_counts,
            "by_severity": severity_counts,
            "most_recent": recent_errors[-1].to_dict() if recent_errors else None,
        }


# Global error handler instance
error_handler = ErrorHandler()


def handle_error(
    exc: Exception,
    context: Optional[ErrorContext] = None,
    operation: Optional[str] = None,
    suggestions: Optional[List[str]] = None,
) -> ArborError:
    """Convenience function to handle errors."""
    return error_handler.handle_exception(exc, context, operation, suggestions)


def safe_execute(
    func: callable,
    *args,
    context: Optional[ErrorContext] = None,
    operation: Optional[str] = None,
    **kwargs,
):
    """Execute a function safely with error handling."""
    try:
        return func(*args, **kwargs)
    except Exception as e:
        arbor_error = handle_error(e, context, operation or func.__name__)
        raise ArborException(
            arbor_error.message,
            category=arbor_error.category,
            severity=arbor_error.severity,
            context=arbor_error.context,
            suggestions=arbor_error.suggestions,
            cause=e,
        )


def error_recovery_decorator(
    operation: Optional[str] = None, suggestions: Optional[List[str]] = None
):
    """Decorator for automatic error handling and recovery."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ArborException:
                # Re-raise ArborExceptions as-is
                raise
            except Exception as e:
                # Handle other exceptions
                context = ErrorContext(
                    operation=operation or func.__name__,
                    function_name=f"{func.__module__}.{func.__name__}",
                )

                arbor_error = handle_error(e, context, operation, suggestions)

                # Raise as ArborException for consistent handling
                raise ArborException(
                    arbor_error.message,
                    category=arbor_error.category,
                    severity=arbor_error.severity,
                    context=arbor_error.context,
                    suggestions=arbor_error.suggestions or suggestions,
                    cause=e,
                )

        return wrapper

    return decorator
