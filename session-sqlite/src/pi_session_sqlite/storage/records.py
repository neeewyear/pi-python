"""SQLite 记录表存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .._types import SqliteDatabase


class NewRecordRow:
    """新记录行参数。"""

    def __init__(
        self,
        *,
        seq: int,
        id: str,
        lane: str,
        run_id: str | None = None,
        type: str,
        op_kind: str | None = None,
        timestamp: str,
        payload: str,
    ) -> None:
        self.seq = seq
        self.id = id
        self.lane = lane
        self.run_id = run_id
        self.type = type
        self.op_kind = op_kind
        self.timestamp = timestamp
        self.payload = payload


async def append_record_row(
    db: SqliteDatabase, session_id: str, record: NewRecordRow
) -> None:
    """追加记录行。"""
    await (
        await db.prepare(
            "INSERT INTO records (session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
        )
    ).run(
        session_id,
        record.seq,
        record.id,
        record.lane,
        record.run_id,
        record.type,
        record.op_kind,
        record.timestamp,
        record.payload,
    )


async def id_exists_in_records(
    db: SqliteDatabase, session_id: str, record_id: str
) -> bool:
    """检查记录 ID 是否存在。"""
    return bool(
        await (
            await db.prepare(
                "SELECT 1 AS found FROM records WHERE session_id = ? AND id = ? LIMIT 1"
            )
        ).get(session_id, record_id)
    )


async def delete_record_rows(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有记录行。"""
    await (await db.prepare("DELETE FROM records WHERE session_id = ?")).run(session_id)


async def read_record_rows(
    db: SqliteDatabase,
    session_id: str,
    query: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """读取记录行。"""
    q: dict[str, object] = query or {}
    predicates: list[str] = ["session_id = ?"]
    params: list[object] = [session_id]
    lane = q.get("lane")
    if lane is not None:
        predicates.append("lane = ?")
        params.append(lane)
    rtype = q.get("type")
    if rtype is not None:
        predicates.append("type = ?")
        params.append(rtype)
    run_id = q.get("run_id")
    if run_id is not None:
        predicates.append("run_id = ?")
        params.append(run_id)
    op_kind = q.get("operation_kind")
    if op_kind is not None:
        predicates.append("op_kind = ?")
        params.append(op_kind)
    after_seq = q.get("after_seq")
    if after_seq is not None:
        predicates.append("seq > ?")
        params.append(after_seq)
    limit_val = q.get("limit")
    limit_sql = ""
    if limit_val is not None:
        limit_sql = " LIMIT ?"
        params.append(limit_val)
    direction = "ASC" if q.get("order") == "oldestFirst" else "DESC"
    return await (
        await db.prepare(
            f"SELECT session_id, seq, id, lane, run_id, type, op_kind, timestamp, payload "
            f"FROM records "
            f"WHERE {' AND '.join(predicates)} "
            f"ORDER BY seq {direction}{limit_sql}"
        )
    ).all(*params)


async def read_open_operation_rows(
    db: SqliteDatabase,
    session_id: str,
    lane: str,
    options: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """读取未结束的操作记录。"""
    opts: dict[str, object] = options or {}
    params: list[object] = [session_id, lane]
    limit_val = opts.get("limit")
    limit_sql = ""
    if limit_val is not None:
        limit_sql = " LIMIT ?"
        params.append(limit_val)
    return await (
        await db.prepare(
            f"SELECT started.session_id, started.seq, started.id, started.lane, started.run_id, "
            f"started.type, started.op_kind, started.timestamp, started.payload "
            f"FROM records AS started "
            f"WHERE started.session_id = ? "
            f"AND started.lane = ? "
            f"AND started.type = 'operation_started' "
            f"AND NOT EXISTS ("
            f"SELECT 1 FROM records AS finished "
            f"WHERE finished.session_id = started.session_id "
            f"AND finished.lane = started.lane "
            f"AND finished.run_id = started.id "
            f"AND finished.type = 'operation_finished' "
            f"AND finished.seq > started.seq"
            f") "
            f"ORDER BY started.seq DESC{limit_sql}"
        )
    ).all(*params)


__all__ = [
    "NewRecordRow",
    "append_record_row",
    "delete_record_rows",
    "id_exists_in_records",
    "read_open_operation_rows",
    "read_record_rows",
]
