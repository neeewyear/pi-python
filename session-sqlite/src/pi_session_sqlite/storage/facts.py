"""SQLite 事实表存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._types import SqliteDatabase


async def append_fact(
    db: SqliteDatabase,
    session_id: str,
    seq: int,
    kind: str,
    key: str | None,
    value: str | None,
) -> None:
    """追加事实行。"""
    await (
        await db.prepare(
            "INSERT INTO facts (session_id, seq, kind, key, value) VALUES (?, ?, ?, ?, ?)"
        )
    ).run(session_id, seq, kind, key, value)


async def read_latest_fact(
    db: SqliteDatabase, session_id: str, kind: str, key: str | None
) -> dict[str, object] | None:
    """读取最新事实。"""
    return await (
        await db.prepare(
            "SELECT session_id, seq, kind, key, value "
            "FROM facts "
            "WHERE session_id = ? AND kind = ? AND key IS ? "
            "ORDER BY seq DESC "
            "LIMIT 1"
        )
    ).get(session_id, kind, key)


async def read_latest_label_facts(
    db: SqliteDatabase, session_id: str
) -> list[dict[str, object]]:
    """读取最新 label 事实。"""
    return await (
        await db.prepare(
            "SELECT key, value FROM ("
            "SELECT key, value, ROW_NUMBER() OVER (PARTITION BY key ORDER BY seq DESC) AS rank "
            "FROM facts "
            "WHERE session_id = ? AND kind = 'label'"
            ") WHERE rank = 1 AND value IS NOT NULL "
            "ORDER BY key"
        )
    ).all(session_id)


async def read_fact_rows(
    db: SqliteDatabase, session_id: str, options: dict[str, object] | None = None
) -> list[dict[str, object]]:
    """读取事实行。"""
    opts: dict[str, object] = options or {}
    predicates: list[str] = ["session_id = ?"]
    params: list[object] = [session_id]
    after_seq = opts.get("after_seq")
    if after_seq is not None:
        predicates.append("seq > ?")
        params.append(after_seq)
    return await (
        await db.prepare(
            f"SELECT session_id, seq, kind, key, value "
            f"FROM facts "
            f"WHERE {' AND '.join(predicates)} "
            f"ORDER BY seq"
        )
    ).all(*params)


async def delete_fact_rows(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有事实行。"""
    await (await db.prepare("DELETE FROM facts WHERE session_id = ?")).run(session_id)


__all__ = [
    "append_fact",
    "delete_fact_rows",
    "read_fact_rows",
    "read_latest_fact",
    "read_latest_label_facts",
]
