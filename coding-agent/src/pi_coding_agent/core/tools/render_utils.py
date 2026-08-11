"""Simple rendering utilities for tool output.

Provides basic text formatting helpers for tool output rendering.
"""

from __future__ import annotations

import os


def shorten_path(path: str) -> str:
    """Shorten a path by replacing the home directory with ~."""
    home = os.path.expanduser("~")
    if path.startswith(home):
        return f"~{path[len(home):]}"
    return path


def str_value(value: object) -> str | None:
    """Convert a value to string, returning None for unexpected types."""
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return None


def replace_tabs(text: str) -> str:
    """Replace tabs with spaces."""
    return text.replace("\t", "   ")


def normalize_display_text(text: str) -> str:
    """Normalize display text by removing carriage returns."""
    return text.replace("\r", "")