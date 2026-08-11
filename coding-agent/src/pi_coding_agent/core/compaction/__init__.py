"""Compaction and summarization utilities.

Provides:
- ``compaction``: Context compaction for long sessions
- ``branch_summarization``: Branch summarization for tree navigation
- ``utils``: Shared utilities (file tracking, serialization)
"""

from . import branch_summarization, compaction, utils

__all__ = [
    "branch_summarization",
    "compaction",
    "utils",
]