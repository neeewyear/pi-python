"""aiosqlite 适配器（对应 TS ``sqlite/index.ts`` 的适配层）。

将 ``aiosqlite`` 的异步 API 包装为异步的 ``SqliteDatabase`` 协议。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import aiosqlite

from ._types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteRunResult,
    SqliteStatement,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class AiosqliteStatement:
    """aiosqlite 预编译语句适配器（每次调用都会执行独立的 execute）。"""

    def __init__(self, conn: aiosqlite.Connection, sql: str) -> None:
        self._conn = conn
        self._sql = sql

    async def run(self, *params: object) -> SqliteRunResult:
        """执行并返回影响行数。"""
        cursor = await self._conn.execute(self._sql, params)
        return SqliteRunResult(
            changes=cursor.rowcount or 0,
            last_insert_rowid=cursor.lastrowid,
        )

    async def get(self, *params: object) -> dict[str, object] | None:
        """执行并返回单行。"""
        cursor = await self._conn.execute(self._sql, params)
        row = await cursor.fetchone()
        if row is None:
            return None
        return dict(row)

    async def all(self, *params: object) -> list[dict[str, object]]:
        """执行并返回所有行。"""
        cursor = await self._conn.execute(self._sql, params)
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


class AiosqliteDatabase:
    """aiosqlite 数据库适配器，实现异步的 ``SqliteDatabase`` 协议。"""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn = conn

    async def exec(self, sql: str) -> None:
        """执行 SQL（支持多语句）。

        使用 ``execute`` 而非 ``executescript`` 以避免 ``isolation_level=None``
        下 ``executescript`` 隐式 BEGIN/COMMIT 与外部事务冲突的问题。
        """
        for statement in sql.split(";"):
            stripped = statement.strip()
            if stripped:
                await self._conn.execute(stripped)

    async def prepare(self, sql: str) -> SqliteStatement:
        """预编译 SQL（实际返回每次独立执行的 statement 适配器）。"""
        return AiosqliteStatement(self._conn, sql)

    async def transaction(self, fn: Callable[[], object]) -> object:
        """执行异步事务。"""
        await self._conn.execute("BEGIN IMMEDIATE")
        try:
            result = fn()
            if hasattr(result, "__await__"):
                result = await result
            await self._conn.execute("COMMIT")
            return result
        except BaseException:
            await self._conn.execute("ROLLBACK")
            raise

    async def close(self) -> None:
        """关闭数据库连接。"""
        await self._conn.close()


def create_aiosqlite_factory() -> SqliteDatabaseFactory:
    """创建 aiosqlite 数据库工厂。

    返回一个工厂对象，其 ``open`` 方法异步创建数据库连接并返回异步适配器。
    """

    class _AiosqliteFactory:
        """aiosqlite 数据库工厂。"""

        async def open(self, path: str) -> SqliteDatabase:
            """打开数据库并返回适配器。"""
            conn = await aiosqlite.connect(path, isolation_level=None)
            conn.row_factory = aiosqlite.Row
            return AiosqliteDatabase(conn)

    return _AiosqliteFactory()


__all__ = [
    "AiosqliteDatabase",
    "AiosqliteStatement",
    "create_aiosqlite_factory",
]
