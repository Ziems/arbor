import click

from .app import ArborMonitorApp


@click.command()
def monitor():
    """Monitor Arbor RL training jobs"""
    app = ArborMonitorApp()
    app.run()


if __name__ == "__main__":
    monitor()
