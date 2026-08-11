"""SQLite 分支 tips 表存储（对应 TS ``sqlite/storage/branch-tips.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._types import SqliteDatabase


async def read_branch_tip_ids(db: SqliteDatabase, session_id: str) -> list[str]:
    """读取所有分支 tip ID。"""
    rows = await (
        await db.prepare(
            "SELECT tip_id FROM branch_tips WHERE session_id = ? ORDER BY tip_id"
        )
    ).all(session_id)
    return [str(r["tip_id"]) for r in rows]


async def read_branch_tip_branch_id(
    db: SqliteDatabase, session_id: str, tip_id: str
) -> str | None:
    """读取分支 tip 的分支 ID。"""
    row = await (
        await db.prepare(
            "SELECT branch_id FROM branch_tips WHERE session_id = ? AND tip_id = ?"
        )
    ).get(session_id, tip_id)
    if not row:
        return None
    return str(row["branch_id"])


async def insert_branch_tip(
    db: SqliteDatabase, session_id: str, tip_id: str, branch_id: str
) -> None:
    """插入分支 tip。"""
    await (
        await db.prepare(
            "INSERT INTO branch_tips (session_id, tip_id, branch_id) VALUES (?, ?, ?)"
        )
    ).run(session_id, tip_id, branch_id)


async def update_branch_tip(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    old_tip_id: str,
    new_tip_id: str,
) -> bool:
    """更新分支 tip。"""
    result = await (
        await db.prepare(
            "UPDATE branch_tips SET tip_id = ? WHERE session_id = ? AND branch_id = ? AND tip_id = ?"
        )
    ).run(new_tip_id, session_id, branch_id, old_tip_id)
    return result.changes == 1


async def delete_branch_tips(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有分支 tips。"""
    await (await db.prepare("DELETE FROM branch_tips WHERE session_id = ?")).run(
        session_id
    )


__all__ = [
    "delete_branch_tips",
    "insert_branch_tip",
    "read_branch_tip_branch_id",
    "read_branch_tip_ids",
    "update_branch_tip",
]
