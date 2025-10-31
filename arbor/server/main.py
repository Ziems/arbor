from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from arbor.core.logging import get_logger
from arbor.server.api.routes import grpo, inference

logger = get_logger(__name__)


# Small helper to serialize exceptions for responses/logging
def make_error_payload(exc: Exception) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "timestamp": datetime.now().isoformat(),
    }


def cleanup_managers(app: FastAPI):
    """Clean up all managers and their resources"""
    logger.info("Starting application cleanup...")

    # List of managers that should have cleanup methods
    manager_names = [
        "gpu_manager",
        "job_manager",
        "inference_manager",
        "grpo_manager",
        "file_train_manager",
        "file_manager",
    ]

    for manager_name in manager_names:
        if hasattr(app.state, manager_name):
            manager = getattr(app.state, manager_name)
            if hasattr(manager, "cleanup"):
                try:
                    logger.info(f"Cleaning up {manager_name}...")
                    manager.cleanup()
                except Exception as e:
                    logger.error(f"Error cleaning up {manager_name}: {e}")
        else:
            logger.debug(f"No {manager_name} found in app state")

    logger.info("Application cleanup completed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app."""
    # Startup
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Application shutdown initiated")
    cleanup_managers(app)


app = FastAPI(title="Arbor API", lifespan=lifespan)


# Include routers
app.include_router(grpo.router, prefix="/v1/fine_tuning/grpo")
app.include_router(inference.router, prefix="/v1/chat")


# ---- Centralized exception handling ----


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception):
    # Map common exception types to HTTP statuses; default to 500
    status_map = {
        ValueError: 400,
        KeyError: 400,
        FileNotFoundError: 404,
        TimeoutError: 504,
        ConnectionError: 502,
    }
    status = 500
    for etype, code in status_map.items():
        if isinstance(exc, etype):
            status = code
            break
    return JSONResponse(status_code=status, content=make_error_payload(exc))
