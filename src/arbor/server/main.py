from fastapi import FastAPI
from arbor.server.api.routes import files, jobs

app = FastAPI(title="Arbor API")

# Include routers
app.include_router(files.router, prefix="/v1/files")
app.include_router(jobs.router, prefix="/v1/fine_tuning/jobs")