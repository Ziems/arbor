#!/usr/bin/env python3
"""
Entry point for running Arbor as a module.

This allows running Arbor with: python -m arbor
"""

if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add the project root to Python path for development
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from arbor.cli import cli

    cli()
