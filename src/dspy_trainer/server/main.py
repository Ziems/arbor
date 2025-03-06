from fastapi import FastAPI
from api.routes import training, files
from core.config import settings

app = FastAPI(title="DSPy Trainer API")

# Include routers
app.include_router(training.router, prefix="/api/training")
app.include_router(files.router, prefix="/api/files")