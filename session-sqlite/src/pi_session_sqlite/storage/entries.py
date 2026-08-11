"""SQLite 条目表存储（对应 TS ``sqlite/storage/entries.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from .._types import SqliteDatabase


class NewEntryRow:
    """新条目行参数。"""

    def __init__(
        self,
        *,
        seq: int,
        id: str,
        parent_id: str | None,
        type: str,
        timestamp: str,
        payload: str,
    ) -> None:
        self.seq = seq
        self.id = id
        self.parent_id = parent_id
        self.type = type
        self.timestamp = timestamp
        self.payload = payload


def entry_payload(entry: Any) -> dict[str, object]:
    """从条目中提取负载（排除公共字段）。"""
    excluded = {"type", "id", "seq", "parent_id", "parentId", "timestamp"}
    return {k: v for k, v in entry.items() if k not in excluded}


def _ordered_sql(order: str | None) -> str:
    """返回 SQL 排序方向。"""
    return "ASC" if order == "oldestFirst" else "DESC"


async def insert_entry_row(db: SqliteDatabase, session_id: str, entry: NewEntryRow) -> None:
    """插入条目行。"""
    await (await db.prepare(
        "INSERT INTO entries (session_id, id, seq, parent_id, type, timestamp, payload) VALUES (?, ?, ?, ?, ?, ?, ?)"
    )).run(
        session_id,
        entry.id,
        entry.seq,
        entry.parent_id,
        entry.type,
        entry.timestamp,
        entry.payload,
    )


async def read_entry_row(
    db: SqliteDatabase, session_id: str, entry_id: str
) -> dict[str, object] | None:
    """读取单个条目行。"""
    return await (await db.prepare(
        "SELECT session_id, seq, id, parent_id, type, timestamp, payload FROM entries WHERE session_id = ? AND id = ?"
    )).get(session_id, entry_id)


async def read_entry_rows(
    db: SqliteDatabase,
    session_id: str,
    options: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """读取条目行列表。"""
    opts: dict[str, object] = options or {}
    predicates: list[str] = ["session_id = ?"]
    params: list[object] = [session_id]
    after_seq = opts.get("after_seq")
    if after_seq is not None:
        predicates.append("seq > ?")
        params.append(after_seq)
    order_val = opts.get("order")
    order_str = str(cast(str, order_val)) if order_val is not None else None
    return await (await db.prepare(
        f"SELECT session_id, seq, id, parent_id, type, timestamp, payload "
        f"FROM entries "
        f"WHERE {' AND '.join(predicates)} "
        f"ORDER BY seq {_ordered_sql(order_str)}"
    )).all(*params)


async def id_exists_in_entries(db: SqliteDatabase, session_id: str, entry_id: str) -> bool:
    """检查条目 ID 是否存在。"""
    return bool(
        await (await db.prepare(
            "SELECT 1 AS found FROM entries WHERE session_id = ? AND id = ? LIMIT 1"
        )).get(session_id, entry_id)
    )


async def delete_entry_rows(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有条目行。"""
    await (await db.prepare("DELETE FROM entries WHERE session_id = ?")).run(session_id)


__all__ = [
    "NewEntryRow",
    "delete_entry_rows",
    "entry_payload",
    "id_exists_in_entries",
    "insert_entry_row",
    "read_entry_row",
    "read_entry_rows",
]
