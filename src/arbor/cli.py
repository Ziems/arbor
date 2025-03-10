import click
import uvicorn
from arbor.server.main import app
from arbor.server.core.config import Settings
from arbor.server.services.file_manager import FileManager
from arbor.server.services.job_manager import JobManager
from arbor.server.services.training_manager import TrainingManager

@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
@click.option('--storage-path', default='./storage', help='Path to store models and uploaded training files')
def serve(host, port, storage_path):
    """Start the Arbor API server"""

    # Create new settings instance with overrides
    settings = Settings(
        STORAGE_PATH=storage_path,
    )

    # Initialize services with settings
    file_manager = FileManager(settings=settings)
    job_manager = JobManager(settings=settings)
    training_manager = TrainingManager(settings=settings)

    # Inject settings into app state
    app.state.settings = settings
    app.state.file_manager = file_manager
    app.state.job_manager = job_manager
    app.state.training_manager = training_manager

    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    cli()