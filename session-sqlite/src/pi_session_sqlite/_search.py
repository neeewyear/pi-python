"""SQLite FTS5 搜索后端（对应 TS ``sqlite/search-backend.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ._migrations import apply_migrations
from .storage.sessions import row_to_metadata

if TYPE_CHECKING:
    from ._types import (
        SqliteDatabase,
        SqliteDatabaseFactory,
        SqliteSessionRepositoryEnv,
    )


class SqliteSessionSearchOptions:
    """SQLite 会话搜索选项。"""

    def __init__(
        self,
        *,
        env: SqliteSessionRepositoryEnv,
        sqlite: SqliteDatabaseFactory,
        database_path: str,
    ) -> None:
        self.env = env
        self.sqlite = sqlite
        self.database_path = database_path


def _get_parent_path(path: str) -> str:
    """获取父目录路径。"""
    import os

    normalized = path.rstrip("/\\")
    return os.path.dirname(normalized) or "."


async def _configure_sqlite_database(db: SqliteDatabase) -> None:
    """配置 SQLite 数据库。"""
    await db.exec("PRAGMA journal_mode=WAL")
    await db.exec("PRAGMA synchronous=FULL")
    await db.exec("PRAGMA busy_timeout=5000")


async def _table_exists(db: SqliteDatabase, name: str) -> bool:
    """检查表是否存在。"""
    return bool(
        await (
            await db.prepare(
                "SELECT 1 AS found FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1"
            )
        ).get(name)
    )


async def _ensure_search_schema(db: SqliteDatabase) -> None:
    """确保搜索模式存在。"""
    fts_exists = await _table_exists(db, "session_search_fts")
    await db.exec(
        "CREATE VIRTUAL TABLE IF NOT EXISTS session_search_fts USING fts5(\n"
        "  payload,\n"
        "  content = 'entries',\n"
        "  content_rowid = 'rowid',\n"
        "  tokenize = 'trigram remove_diacritics 1'\n"
        ");\n"
        "CREATE TRIGGER IF NOT EXISTS session_search_fts_ai AFTER INSERT ON entries BEGIN\n"
        "  INSERT INTO session_search_fts(rowid, payload) VALUES (new.rowid, new.payload);\n"
        "END;\n"
        "CREATE TRIGGER IF NOT EXISTS session_search_fts_ad AFTER DELETE ON entries BEGIN\n"
        "  INSERT INTO session_search_fts(session_search_fts, rowid, payload) VALUES('delete', old.rowid, old.payload);\n"
        "END;\n"
        "CREATE TRIGGER IF NOT EXISTS session_search_fts_au AFTER UPDATE OF payload ON entries BEGIN\n"
        "  INSERT INTO session_search_fts(session_search_fts, rowid, payload) VALUES('delete', old.rowid, old.payload);\n"
        "  INSERT INTO session_search_fts(rowid, payload) VALUES (new.rowid, new.payload);\n"
        "END;\n"
    )
    if not fts_exists:
        await db.exec(
            "INSERT INTO session_search_fts(session_search_fts) VALUES('rebuild')"
        )


class SqliteSessionSearch:
    """SQLite FTS5 会话搜索。"""

    def __init__(self, options: SqliteSessionSearchOptions) -> None:
        self._options = options
        self._database_path: str | None = None

    async def _get_database_path(self) -> str:
        """获取数据库路径。"""
        if self._database_path is None:
            self._database_path = await self._options.env.absolute_path(
                self._options.database_path
            )
        return self._database_path

    async def _open_database(self) -> SqliteDatabase:
        """打开数据库。"""
        path = await self._get_database_path()
        directory = _get_parent_path(path)
        await self._options.env.create_dir(directory, recursive=True)
        db = await self._options.sqlite.open(path)
        try:
            await _configure_sqlite_database(db)
            await apply_migrations(db)
            await _ensure_search_schema(db)
            return db
        except BaseException:
            await db.close()
            raise

    async def search(
        self,
        text: str,
        cwd: str | None = None,
    ) -> list[dict[str, object]]:
        """搜索会话。"""
        normalized = text.strip()
        if not normalized:
            return []
        db = await self._open_database()
        try:
            query = f'"{normalized.replace(chr(34), chr(34) + chr(34))}"'
            rows = await (
                await db.prepare(
                    "SELECT s.id, s.created_at, s.metadata, s.cwd, s.parent_session_id, "
                    "se.id AS entry_id, se.timestamp, bm25(session_search_fts) AS score "
                    "FROM session_search_fts "
                    "JOIN entries se ON se.rowid = session_search_fts.rowid "
                    "JOIN sessions s ON s.id = se.session_id "
                    "WHERE session_search_fts MATCH ? AND (? IS NULL OR s.cwd = ?) "
                    "ORDER BY score"
                )
            ).all(query, cwd, cwd)
            path = await self._get_database_path()
            result: list[dict[str, object]] = []
            for row in rows:
                metadata = row_to_metadata(row, path)
                result.append(
                    {
                        "metadata": metadata,
                        "entry_id": str(row["entry_id"]),
                        "timestamp": str(row["timestamp"]),
                        "score": float(cast(float, row["score"])),
                    }
                )
            return result
        finally:
            await db.close()


def create_sqlite_session_search(
    options: SqliteSessionSearchOptions,
) -> SqliteSessionSearch:
    """创建 SQLite 会话搜索。"""
    return SqliteSessionSearch(options)


__all__ = [
    "SqliteSessionSearch",
    "SqliteSessionSearchOptions",
    "create_sqlite_session_search",
]
