"""Path resolution utilities for tools.

Provides simple path resolution utilities for the coding-agent tool system.
"""

from __future__ import annotations

import os
import os.path
from pathlib import Path


def resolve_path(path_str: str, cwd: str | None = None) -> str:
    """Resolve a path relative to the given cwd.

    Handles ~ expansion and absolute paths.
    """
    # Expand user home directory
    expanded = os.path.expanduser(path_str)

    # If it's already absolute, normalize it
    if os.path.isabs(expanded):
        return os.path.normpath(expanded)

    # Resolve relative to cwd
    base = cwd if cwd is not None else os.getcwd()
    return os.path.normpath(os.path.join(base, expanded))


def normalize_path(path_str: str) -> str:
    """Normalize a path string.

    Removes redundant separators and up-level references.
    """
    return os.path.normpath(os.path.expanduser(path_str))


def canonicalize_path(path_str: str) -> str:
    """Canonicalize a path to its absolute, normalized form.

    Resolves symlinks if possible, otherwise falls back to normalized absolute path.
    """
    expanded = os.path.expanduser(path_str)
    if os.path.isabs(expanded):
        try:
            return os.path.realpath(expanded)
        except OSError:
            return os.path.normpath(expanded)
    return os.path.normpath(os.path.join(os.getcwd(), expanded))


def path_exists(file_path: str) -> bool:
    """Check if a path exists on the filesystem."""
    return Path(file_path).exists()