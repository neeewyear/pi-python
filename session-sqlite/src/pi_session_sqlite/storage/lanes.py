"""SQLite lane 表存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_session.types import SessionError

if TYPE_CHECKING:
    from .._types import SqliteDatabase


async def create_initial_lane(
    db: SqliteDatabase, session_id: str, lane: str = "main", leaf_id: str | None = None
) -> None:
    """创建初始 lane。"""
    await (
        await db.prepare(
            "INSERT INTO lanes (session_id, lane, leaf_id) VALUES (?, ?, ?)"
        )
    ).run(session_id, lane, leaf_id)


async def read_lanes(db: SqliteDatabase, session_id: str) -> list[dict[str, object]]:
    """读取所有 lanes。"""
    rows = await (
        await db.prepare(
            "SELECT l.session_id, l.lane, l.leaf_id, "
            "(l.leaf_id IS NULL OR EXISTS ("
            "SELECT 1 FROM entries AS e WHERE e.session_id = l.session_id AND e.id = l.leaf_id"
            ")) AS leaf_exists "
            "FROM lanes AS l WHERE l.session_id = ? ORDER BY l.lane"
        )
    ).all(session_id)
    for row in rows:
        if not row.get("leaf_exists"):
            raise SessionError(
                "storage",
                f"Lane {row['lane']} points at missing entry {row['leaf_id']}",
            )
    return [
        {"session_id": r["session_id"], "lane": r["lane"], "leaf_id": r["leaf_id"]}
        for r in rows
    ]


async def read_lane(
    db: SqliteDatabase, session_id: str, lane: str
) -> dict[str, object] | None:
    """读取单个 lane。"""
    return await (
        await db.prepare(
            "SELECT session_id, lane, leaf_id FROM lanes WHERE session_id = ? AND lane = ?"
        )
    ).get(session_id, lane)


async def read_lane_head(
    db: SqliteDatabase, session_id: str, lane: str
) -> dict[str, object]:
    """读取 lane 的头部（叶子）。"""
    row = await (
        await db.prepare(
            "SELECT l.leaf_id, "
            "(l.leaf_id IS NULL OR EXISTS ("
            "SELECT 1 FROM entries AS e WHERE e.session_id = l.session_id AND e.id = l.leaf_id"
            ")) AS leaf_exists "
            "FROM lanes AS l WHERE l.session_id = ? AND l.lane = ?"
        )
    ).get(session_id, lane)
    if not row:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    if not row.get("leaf_exists"):
        raise SessionError("storage", f"Entry {row['leaf_id']} not found")
    return {"leaf_id": row["leaf_id"]}


async def create_lane(
    db: SqliteDatabase, session_id: str, seq: int, lane: str, leaf_id: str | None
) -> None:
    """创建 lane 并记录移动。"""
    await (
        await db.prepare(
            "INSERT INTO lanes (session_id, lane, leaf_id) VALUES (?, ?, ?)"
        )
    ).run(session_id, lane, leaf_id)
    await _append_lane_move(db, session_id, seq, lane, leaf_id)


async def move_lane(
    db: SqliteDatabase, session_id: str, seq: int, lane: str, leaf_id: str | None
) -> None:
    """移动 lane 并记录移动。"""
    result = await (
        await db.prepare(
            "UPDATE lanes SET leaf_id = ? WHERE session_id = ? AND lane = ?"
        )
    ).run(leaf_id, session_id, lane)
    if result.changes != 1:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")
    await _append_lane_move(db, session_id, seq, lane, leaf_id)


async def set_lane_leaf(
    db: SqliteDatabase, session_id: str, lane: str, leaf_id: str | None
) -> None:
    """设置 lane 的叶子（不记录移动）。"""
    result = await (
        await db.prepare(
            "UPDATE lanes SET leaf_id = ? WHERE session_id = ? AND lane = ?"
        )
    ).run(leaf_id, session_id, lane)
    if result.changes != 1:
        raise SessionError("invalid_lane", f"Lane not found: {lane}")


async def read_lane_move_rows(
    db: SqliteDatabase, session_id: str, options: dict[str, object] | None = None
) -> list[dict[str, object]]:
    """读取 lane 移动记录。"""
    opts: dict[str, object] = options or {}
    predicates: list[str] = ["session_id = ?"]
    params: list[object] = [session_id]
    after_seq = opts.get("after_seq")
    if after_seq is not None:
        predicates.append("seq > ?")
        params.append(after_seq)
    return await (
        await db.prepare(
            f"SELECT session_id, seq, lane, leaf_id "
            f"FROM lane_moves "
            f"WHERE {' AND '.join(predicates)} "
            f"ORDER BY seq"
        )
    ).all(*params)


async def delete_lane_rows(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有 lane 相关行。"""
    await (await db.prepare("DELETE FROM lane_moves WHERE session_id = ?")).run(
        session_id
    )
    await (await db.prepare("DELETE FROM lanes WHERE session_id = ?")).run(session_id)


async def _append_lane_move(
    db: SqliteDatabase, session_id: str, seq: int, lane: str, leaf_id: str | None
) -> None:
    """追加 lane 移动记录。"""
    await (
        await db.prepare(
            "INSERT INTO lane_moves (session_id, seq, lane, leaf_id) VALUES (?, ?, ?, ?)"
        )
    ).run(session_id, seq, lane, leaf_id)


__all__ = [
    "create_initial_lane",
    "create_lane",
    "delete_lane_rows",
    "move_lane",
    "read_lane",
    "read_lane_head",
    "read_lane_move_rows",
    "read_lanes",
    "set_lane_leaf",
]
