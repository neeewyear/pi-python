"""SQLite 迁移运行器。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._sql import INITIAL_SQL

if TYPE_CHECKING:
    from ._types import SqliteDatabase


async def apply_migrations(db: SqliteDatabase) -> None:
    """应用 SQLite 迁移。

    确保 ``migrations`` 表存在，然后按顺序应用所有未执行的迁移。
    """
    await db.exec(
        "CREATE TABLE IF NOT EXISTS migrations ("
        "id TEXT PRIMARY KEY, "
        "applied_at TEXT NOT NULL"
        ")"
    )
    applied_rows = await (
        await db.prepare("SELECT id FROM migrations ORDER BY applied_at, id")
    ).all()
    applied = {str(r["id"]) for r in applied_rows}

    if "001_initial" not in applied:
        await db.exec(INITIAL_SQL)
        await (
            await db.prepare(
                "INSERT INTO migrations (id, applied_at) VALUES (?, datetime('now'))"
            )
        ).run("001_initial")


__all__ = ["apply_migrations"]
