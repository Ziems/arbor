from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from arbor.server.api.routes import files, grpo, inference, jobs
from arbor.server.utils.logging import apply_uvicorn_formatting


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan event handler for FastAPI app."""
    # Startup
    apply_uvicorn_formatting()
    yield
    # Shutdown (if needed)


app = FastAPI(title="Arbor API", lifespan=lifespan)


# Include routers
app.include_router(files.router, prefix="/v1/files")
app.include_router(jobs.router, prefix="/v1/fine_tuning/jobs")
app.include_router(grpo.router, prefix="/v1/fine_tuning/grpo")
app.include_router(inference.router, prefix="/v1/chat")


@app.get("/health")
def health_check(request: Request):
    """Enhanced health check with system and GPU information."""
    health_manager = request.app.state.health_manager
    return health_manager.get_health_status()


@app.get("/health/simple")
def simple_health_check(request: Request):
    """Simple health check that returns just the status."""
    health_manager = request.app.state.health_manager
    return {
        "status": "healthy" if health_manager.is_healthy() else "unhealthy",
        "timestamp": health_manager.get_health_status()["timestamp"],
    }
