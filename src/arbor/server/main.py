from fastapi import FastAPI
from arbor.server.api.routes import training, files, jobs
from arbor.server.core.config import settings

app = FastAPI(title="Arbor API")

# Include routers
app.include_router(training.router, prefix="/api/fine-tune")
app.include_router(files.router, prefix="/api/files")
app.include_router(jobs.router, prefix="/api/job")