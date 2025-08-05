#!/usr/bin/env python3
"""
Entry point for running Arbor as a module.

This allows running Arbor with: python -m arbor
"""

if __name__ == "__main__":
    from arbor.cli import cli

    cli()
