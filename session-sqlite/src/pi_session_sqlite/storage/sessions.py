"""SQLite 会话表存储（对应 TS ``sqlite/storage/sessions.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pi_session.types import SessionError

if TYPE_CHECKING:
    from .._types import SqliteDatabase, SqliteSessionMetadata


class SessionRow(dict[str, object]):
    """SQLite sessions 表行。"""

    id: str
    created_at: str
    metadata: str | None
    cwd: str
    parent_session_id: str | None


class NewSessionRow:
    """新会话行参数。"""

    def __init__(
        self,
        *,
        id: str,
        created_at: str,
        cwd: str,
        parent_session_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> None:
        self.id = id
        self.created_at = created_at
        self.cwd = cwd
        self.parent_session_id = parent_session_id
        self.metadata = metadata


def _parse_metadata(metadata: str | None, session_id: str) -> dict[str, object] | None:
    """解析 JSON 元数据。"""
    if metadata is None:
        return None
    import orjson

    try:
        parsed = orjson.loads(metadata)
    except orjson.JSONDecodeError as error:
        raise SessionError(
            "storage",
            f"Invalid SQLite session {session_id}: metadata is not valid JSON",
            error,
        ) from error
    if not isinstance(parsed, dict):
        raise SessionError(
            "storage",
            f"Invalid SQLite session {session_id}: metadata must be an object",
        )
    return parsed


def _serialize_metadata(metadata: dict[str, object] | None) -> str | None:
    """序列化元数据为 JSON。"""
    if metadata is None:
        return None
    import orjson

    return orjson.dumps(metadata).decode("utf-8")


async def session_exists(db: SqliteDatabase, session_id: str) -> bool:
    """检查会话是否存在。"""
    return bool(
        await (await db.prepare("SELECT 1 AS found FROM sessions WHERE id = ?")).get(
            session_id
        )
    )


async def insert_session_row(db: SqliteDatabase, session: NewSessionRow) -> None:
    """插入会话行。"""
    await (
        await db.prepare(
            "INSERT INTO sessions (id, created_at, metadata, cwd, parent_session_id) VALUES (?, ?, ?, ?, ?)"
        )
    ).run(
        session.id,
        session.created_at,
        _serialize_metadata(session.metadata),
        session.cwd,
        session.parent_session_id,
    )


async def read_session_row(
    db: SqliteDatabase, session_id: str
) -> dict[str, object] | None:
    """读取会话行。"""
    return await (
        await db.prepare(
            "SELECT id, created_at, metadata, cwd, parent_session_id FROM sessions WHERE id = ?"
        )
    ).get(session_id)


async def read_session_rows(
    db: SqliteDatabase, options: dict[str, object] | None = None
) -> list[dict[str, object]]:
    """读取所有会话行。"""
    opts: dict[str, object] = options or {}
    cwd = opts.get("cwd")
    if cwd is not None:
        return await (
            await db.prepare(
                "SELECT id, created_at, metadata, cwd, parent_session_id FROM sessions WHERE cwd = ? ORDER BY created_at DESC"
            )
        ).all(cwd)
    return await (
        await db.prepare(
            "SELECT id, created_at, metadata, cwd, parent_session_id FROM sessions ORDER BY created_at DESC"
        )
    ).all()


async def delete_session_row(db: SqliteDatabase, session_id: str) -> None:
    """删除会话行。"""
    await (await db.prepare("DELETE FROM sessions WHERE id = ?")).run(session_id)


def row_to_metadata(row: dict[str, object], path: str) -> SqliteSessionMetadata:
    """将行转换为元数据。"""
    from .._types import SqliteSessionMetadata

    created_at_str = row.get("created_at", "")
    assert isinstance(created_at_str, str)
    import datetime

    created_at = int(
        datetime.datetime.fromisoformat(
            created_at_str.replace("Z", "+00:00")
        ).timestamp()
        * 1000
    )
    return SqliteSessionMetadata(
        id=str(row["id"]),
        created_at=created_at,
        cwd=str(row.get("cwd", "")),
        path=path,
        parent_session_id=str(row["parent_session_id"])
        if row.get("parent_session_id")
        else None,
        metadata=_parse_metadata(
            str(row.get("metadata")) if row.get("metadata") is not None else None,
            str(row["id"]),
        ),
    )


__all__ = [
    "NewSessionRow",
    "SessionRow",
    "delete_session_row",
    "insert_session_row",
    "read_session_row",
    "read_session_rows",
    "row_to_metadata",
    "session_exists",
]
