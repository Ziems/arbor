from fastapi import FastAPI
from dspy_trainer.server.api.routes import training, files, jobs
from dspy_trainer.server.core.config import settings

app = FastAPI(title="DSPy Trainer API")

# Include routers
app.include_router(training.router, prefix="/api/fine-tune")
app.include_router(files.router, prefix="/api/files")
app.include_router(jobs.router, prefix="/api/job")