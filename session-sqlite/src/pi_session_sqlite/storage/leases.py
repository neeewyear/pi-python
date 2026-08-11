"""SQLite 租约表存储（对应 TS ``sqlite/storage/leases.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .._types import SqliteDatabase


class SessionLease:
    """会话租约。"""

    def __init__(self, owner_id: str, fence: int, expires_at_ms: int) -> None:
        self.owner_id = owner_id
        self.fence = fence
        self.expires_at_ms = expires_at_ms


async def acquire_session_lease(
    db: SqliteDatabase,
    session_id: str,
    owner_id: str,
    now: int,
    expires_at_ms: int,
) -> SessionLease | None:
    """获取会话租约。"""
    row = await (
        await db.prepare(
            "INSERT INTO leases (session_id, owner_id, fence, expires_at_ms) "
            "VALUES (?, ?, 1, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET "
            "owner_id = excluded.owner_id, "
            "fence = leases.fence + 1, "
            "expires_at_ms = excluded.expires_at_ms "
            "WHERE leases.expires_at_ms <= ? "
            "RETURNING owner_id, fence, expires_at_ms"
        )
    ).get(session_id, owner_id, expires_at_ms, now)
    if not row:
        return None
    return SessionLease(
        owner_id=str(row["owner_id"]),
        fence=int(cast(int, row["fence"])),
        expires_at_ms=int(cast(int, row["expires_at_ms"])),
    )


async def renew_session_lease(
    db: SqliteDatabase,
    session_id: str,
    lease: SessionLease,
    now: int,
    expires_at_ms: int,
) -> bool:
    """续期会话租约。"""
    result = await (
        await db.prepare(
            "UPDATE leases SET expires_at_ms = ? "
            "WHERE session_id = ? AND owner_id = ? AND fence = ? AND expires_at_ms > ?"
        )
    ).run(expires_at_ms, session_id, lease.owner_id, lease.fence, now)
    if result.changes == 1:
        lease.expires_at_ms = expires_at_ms
    return result.changes == 1


async def release_session_lease(
    db: SqliteDatabase, session_id: str, lease: SessionLease
) -> None:
    """释放会话租约。"""
    await (
        await db.prepare(
            "DELETE FROM leases WHERE session_id = ? AND owner_id = ? AND fence = ?"
        )
    ).run(session_id, lease.owner_id, lease.fence)


async def delete_session_lease(db: SqliteDatabase, session_id: str) -> None:
    """删除会话租约。"""
    await (await db.prepare("DELETE FROM leases WHERE session_id = ?")).run(session_id)


__all__ = [
    "SessionLease",
    "acquire_session_lease",
    "delete_session_lease",
    "release_session_lease",
    "renew_session_lease",
]
