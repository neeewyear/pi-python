"""SQLite 会话后端类型定义（对应 TS ``sqlite/types.ts``）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Protocol

from pi_session.types import SessionMetadata


class SqliteSessionMetadata(SessionMetadata):
    """SQLite 会话元数据，扩展了 ``SessionMetadata``。"""

    cwd: str = ""
    path: str = ""
    metadata: dict[str, object] | None = None


class SqliteSessionCreateOptions:
    """SQLite 会话创建选项。"""

    def __init__(
        self,
        *,
        id: str | None = None,
        cwd: str = "",
        parent_session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.id = id
        self.cwd = cwd
        self.parent_session_id = parent_session_id
        self.metadata = metadata


class SqliteSessionListOptions:
    """SQLite 会话列表选项。"""

    cwd: str | None = None


class SqliteWriterLeaseOptions:
    """写租约选项。"""

    def __init__(
        self,
        *,
        ttl_ms: int = 30_000,
        heartbeat_interval_ms: int = 10_000,
    ) -> None:
        self.ttl_ms = ttl_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms


class SqliteRunResult:
    """SQLite 执行结果。"""

    def __init__(self, changes: int, last_insert_rowid: int | None = None) -> None:
        self.changes = changes
        self.last_insert_rowid = last_insert_rowid


class SqliteStatement(Protocol):
    """SQLite 预编译语句协议。"""

    async def run(self, *params: object) -> SqliteRunResult: ...
    async def get(self, *params: object) -> dict[str, object] | None: ...
    async def all(self, *params: object) -> list[dict[str, object]]: ...


class SqliteDatabase(Protocol):
    """SQLite 数据库协议。"""

    async def exec(self, sql: str) -> None: ...
    async def prepare(self, sql: str) -> SqliteStatement: ...
    async def transaction(self, fn: Callable[[], Awaitable[object]]) -> object: ...
    async def close(self) -> None: ...


class SqliteDatabaseFactory(Protocol):
    """SQLite 数据库工厂。"""

    async def open(self, path: str) -> SqliteDatabase: ...


class SqliteSessionRepositoryEnv:
    """SQLite 会话仓库环境。"""

    async def absolute_path(self, path: str) -> str:
        return path

    async def create_dir(self, path: str, *, recursive: bool = False) -> None:
        return None

    async def exists(self, path: str) -> bool:
        return True


__all__ = [
    "SqliteDatabase",
    "SqliteDatabaseFactory",
    "SqliteRunResult",
    "SqliteSessionCreateOptions",
    "SqliteSessionListOptions",
    "SqliteSessionMetadata",
    "SqliteSessionRepositoryEnv",
    "SqliteStatement",
    "SqliteWriterLeaseOptions",
]
