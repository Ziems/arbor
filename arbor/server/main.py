from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from arbor.server.api.routes import files, grpo, inference, jobs, monitor
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
# Monitoring and observability
app.include_router(monitor.router)
