"""SQLite 会话统计存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pi_session.types import SessionError

if TYPE_CHECKING:
    from .._types import SqliteDatabase


async def create_stats(
    db: SqliteDatabase, session_id: str, message_count: int = 0
) -> None:
    """创建会话统计。"""
    await (
        await db.prepare(
            "INSERT INTO session_stats (session_id, message_count, cached_tokens, uncached_tokens, total_tokens, cost_total) "
            "VALUES (?, ?, 0, 0, 0, 0)"
        )
    ).run(session_id, message_count)


async def read_stats(db: SqliteDatabase, session_id: str) -> dict[str, object]:
    """读取会话统计。"""
    row = await (
        await db.prepare(
            "SELECT session_id, message_count, cached_tokens, uncached_tokens, total_tokens, cost_total "
            "FROM session_stats WHERE session_id = ?"
        )
    ).get(session_id)
    if not row:
        raise SessionError("storage", f"Missing stats row for session {session_id}")
    return {
        "message_count": int(cast(int, row["message_count"])),
        "cached_tokens": int(cast(int, row["cached_tokens"])),
        "uncached_tokens": int(cast(int, row["uncached_tokens"])),
        "total_tokens": int(cast(int, row["total_tokens"])),
        "cost_total": float(cast(float, row["cost_total"])),
    }


async def increment_message_count(db: SqliteDatabase, session_id: str) -> None:
    """增加消息计数。"""
    result = await (
        await db.prepare(
            "UPDATE session_stats SET message_count = message_count + 1 WHERE session_id = ?"
        )
    ).run(session_id)
    if result.changes != 1:
        raise SessionError("storage", f"Missing stats row for session {session_id}")


async def add_usage_to_stats(
    db: SqliteDatabase, session_id: str, usage: dict[str, object]
) -> None:
    """添加用量到统计。"""
    cache_read = int(cast(int, usage.get("cache_read", 0)))
    input_val = int(cast(int, usage.get("input", 0)))
    cache_write = int(cast(int, usage.get("cache_write", 0)))
    total_tokens = int(cast(int, usage.get("total_tokens", 0)))
    cost_val = usage.get("cost")
    cost_total = 0.0
    if isinstance(cost_val, dict):
        cost_total = float(cast(float, cost_val.get("total", 0.0)))
    result = await (
        await db.prepare(
            "UPDATE session_stats SET "
            "cached_tokens = cached_tokens + ?, "
            "uncached_tokens = uncached_tokens + ?, "
            "total_tokens = total_tokens + ?, "
            "cost_total = cost_total + ? "
            "WHERE session_id = ?"
        )
    ).run(
        cache_read,
        input_val + cache_write,
        total_tokens,
        cost_total,
        session_id,
    )
    if result.changes != 1:
        raise SessionError("storage", f"Missing stats row for session {session_id}")


async def delete_stats(db: SqliteDatabase, session_id: str) -> None:
    """删除会话统计。"""
    await (await db.prepare("DELETE FROM session_stats WHERE session_id = ?")).run(
        session_id
    )


__all__ = [
    "add_usage_to_stats",
    "create_stats",
    "delete_stats",
    "increment_message_count",
    "read_stats",
]
