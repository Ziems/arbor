import click
import uvicorn
from arbor.server.main import app

@click.group()
def cli():
    pass

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind to')
@click.option('--port', default=8000, help='Port to bind to')
def serve(host, port):
    """Start the Arbor API server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == '__main__':
    cli()