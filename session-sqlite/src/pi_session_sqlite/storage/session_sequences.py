"""SQLite 会话序列存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pi_session.types import SessionError

if TYPE_CHECKING:
    from .._types import SqliteDatabase


async def create_sequence(db: SqliteDatabase, session_id: str, next_seq: int = 1) -> None:
    """创建会话序列。"""
    await (await db.prepare(
        "INSERT INTO session_sequences (session_id, next_seq) VALUES (?, ?)"
    )).run(session_id, next_seq)


async def get_next_sequence(db: SqliteDatabase, session_id: str) -> int:
    """获取下一个序列号。"""
    row = await (await db.prepare("SELECT next_seq FROM session_sequences WHERE session_id = ?")).get(
        session_id
    )
    if not row:
        raise SessionError("storage", f"Missing sequence row for session {session_id}")
    return int(cast(int, row["next_seq"]))


async def set_next_sequence(db: SqliteDatabase, session_id: str, next_seq: int) -> None:
    """设置下一个序列号。"""
    await (await db.prepare("UPDATE session_sequences SET next_seq = ? WHERE session_id = ?")).run(
        next_seq, session_id
    )


async def advance_sequence(db: SqliteDatabase, session_id: str, seq: int) -> None:
    """推进序列号。"""
    await set_next_sequence(db, session_id, seq + 1)


async def delete_sequence(db: SqliteDatabase, session_id: str) -> None:
    """删除会话序列。"""
    await (await db.prepare("DELETE FROM session_sequences WHERE session_id = ?")).run(session_id)


__all__ = [
    "advance_sequence",
    "create_sequence",
    "delete_sequence",
    "get_next_sequence",
    "set_next_sequence",
]
