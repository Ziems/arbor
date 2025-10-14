from contextlib import asynccontextmanager

from fastapi import FastAPI

from arbor.server.api.routes import files, grpo, inference, jobs, monitor
from arbor.server.utils.logging import apply_uvicorn_formatting, get_logger

logger = get_logger(__name__)


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
    apply_uvicorn_formatting()
    logger.info("Application startup completed")

    yield

    # Shutdown
    logger.info("Application shutdown initiated")
    cleanup_managers(app)


app = FastAPI(title="Arbor API", lifespan=lifespan)


# Include routers
app.include_router(files.router, prefix="/v1/files")
app.include_router(jobs.router, prefix="/v1/fine_tuning/jobs")
app.include_router(grpo.router, prefix="/v1/fine_tuning/grpo")
app.include_router(inference.router, prefix="/v1/chat")
# Monitoring and observability
app.include_router(monitor.router)
