"""SQLite 会话后端。"""

from __future__ import annotations

from ._adapter import AiosqliteDatabase, AiosqliteStatement, create_aiosqlite_factory
from ._migrations import apply_migrations
from ._repo import SqliteSessionRepository, SqliteSessionRepositoryOptions, SqliteWriterLeaseOptions
from ._search import SqliteSessionSearch, SqliteSessionSearchOptions, create_sqlite_session_search
from ._types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteRunResult,
    SqliteSessionCreateOptions,
    SqliteSessionListOptions,
    SqliteSessionMetadata,
    SqliteSessionRepositoryEnv,
    SqliteStatement,
)

__all__ = [
    "AiosqliteDatabase",
    "AiosqliteStatement",
    "SqliteDatabase",
    "SqliteDatabaseFactory",
    "SqliteRunResult",
    "SqliteSessionCreateOptions",
    "SqliteSessionListOptions",
    "SqliteSessionMetadata",
    "SqliteSessionRepository",
    "SqliteSessionRepositoryEnv",
    "SqliteSessionRepositoryOptions",
    "SqliteSessionSearch",
    "SqliteSessionSearchOptions",
    "SqliteStatement",
    "SqliteWriterLeaseOptions",
    "apply_migrations",
    "create_aiosqlite_factory",
    "create_sqlite_session_search",
]