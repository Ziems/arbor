#!/usr/bin/env python3
"""
Script to print a file tree of the arbor directory.
"""

import os
from pathlib import Path


def print_tree(directory, prefix="", is_last=True, max_depth=None, current_depth=0):
    """
    Print a directory tree structure.

    Args:
        directory: Path object or string path to directory
        prefix: String prefix for current level (used for tree formatting)
        is_last: Boolean indicating if this is the last item at current level
        max_depth: Maximum depth to traverse (None for unlimited)
        current_depth: Current depth in the tree
    """
    if max_depth is not None and current_depth >= max_depth:
        return

    directory = Path(directory)

    # Skip hidden directories and common ignore patterns
    ignore_patterns = {
        ".git",
        "__pycache__",
        ".pytest_cache",
        ".venv",
        "venv",
        "node_modules",
        ".DS_Store",
        ".idea",
        ".vscode",
    }

    if directory.name in ignore_patterns:
        return

    # Print current directory/file
    connector = "└── " if is_last else "├── "
    print(f"{prefix}{connector}{directory.name}")

    if directory.is_dir():
        # Get all items in directory, sorted
        try:
            items = sorted(
                [
                    item
                    for item in directory.iterdir()
                    if item.name not in ignore_patterns
                ]
            )
        except PermissionError:
            return

        # Separate directories and files, directories first
        dirs = [item for item in items if item.is_dir()]
        files = [item for item in items if item.is_file()]
        all_items = dirs + files

        # Print each item
        for i, item in enumerate(all_items):
            is_last_item = i == len(all_items) - 1
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_tree(item, new_prefix, is_last_item, max_depth, current_depth + 1)


def main():
    """Main function to print the arbor directory tree."""
    # Get the current directory (should be arbor root)
    current_dir = Path.cwd()

    print(f"File tree for: {current_dir}")
    print("=" * 50)

    # Print the tree with a reasonable max depth to avoid overwhelming output
    print_tree(current_dir, max_depth=5)

    print("\n" + "=" * 50)
    print("Note: Limited to 4 levels deep and excludes common hidden/cache directories")


if __name__ == "__main__":
    main()
