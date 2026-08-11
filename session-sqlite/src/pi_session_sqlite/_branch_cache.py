"""SQLite 分支缓存管理（对应 TS ``sqlite/branch-cache.ts``）。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from pi_session.types import SessionError

from .storage.branch_entries import (
    copy_branch_entries_through_seq,
    delete_branch_entries,
    insert_branch_entries_for_path,
    insert_branch_entry,
    read_branch_containing_entry,
)
from .storage.branch_tips import (
    delete_branch_tips,
    insert_branch_tip,
    read_branch_tip_branch_id,
    update_branch_tip,
)

if TYPE_CHECKING:
    from ._types import SqliteDatabase


async def delete_branch_cache(db: SqliteDatabase, session_id: str) -> None:
    """删除分支缓存。"""
    await delete_branch_tips(db, session_id)
    await delete_branch_entries(db, session_id)


async def rebuild_branch_cache(db: SqliteDatabase, session_id: str) -> None:
    """重建分支缓存。"""
    tips = await (await db.prepare(
        "SELECT leaf.id FROM entries AS leaf "
        "WHERE leaf.session_id = ? "
        "AND NOT EXISTS ("
        "SELECT 1 FROM entries AS child "
        "WHERE child.session_id = leaf.session_id AND child.parent_id = leaf.id"
        ") "
        "ORDER BY leaf.seq"
    )).all(session_id)
    await delete_branch_cache(db, session_id)
    for tip in tips:
        await build_cached_branch(db, session_id, str(tip["id"]))


async def build_cached_branch(db: SqliteDatabase, session_id: str, leaf_id: str) -> None:
    """构建缓存分支。"""

    from pi_agent.uuid7 import uuidv7

    branch_id = uuidv7()
    await db.exec("SAVEPOINT build_branch_cache")
    try:
        await insert_branch_entries_for_path(db, session_id, branch_id, leaf_id)
        await insert_branch_tip(db, session_id, leaf_id, branch_id)
        await db.exec("RELEASE SAVEPOINT build_branch_cache")
    except BaseException as error:
        try:
            await db.exec("ROLLBACK TO SAVEPOINT build_branch_cache")
            await db.exec("RELEASE SAVEPOINT build_branch_cache")
        except BaseException:
            pass
        if isinstance(error, SessionError):
            raise
        raise SessionError(
            "storage",
            f"Failed to build SQLite branch cache at entry {leaf_id}",
            error if isinstance(error, Exception) else None,
        ) from error


async def _extend_branch(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    parent_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
) -> None:
    """扩展分支。"""
    await insert_branch_entry(
        db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
    )
    if not await update_branch_tip(db, session_id, branch_id, parent_id, entry_id):
        raise SessionError(
            "invalid_entry", f"Branch tip {parent_id} changed during append"
        )


async def append_entry_to_branch_cache(
    db: SqliteDatabase,
    session_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
    parent_id: str | None,
) -> None:
    """追加条目到分支缓存。"""
    from pi_agent.uuid7 import uuidv7

    if parent_id is None:
        branch_id = uuidv7()
        await insert_branch_entry(
            db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
        )
        await insert_branch_tip(db, session_id, entry_id, branch_id)
        return

    tip_branch_id = await read_branch_tip_branch_id(db, session_id, parent_id)
    if tip_branch_id is not None:
        await _extend_branch(
            db,
            session_id,
            tip_branch_id,
            parent_id,
            entry_id,
            entry_seq,
            entry_type,
            custom_type,
        )
        return

    source = await read_branch_containing_entry(db, session_id, parent_id)
    if not source:
        raise SessionError(
            "invalid_entry",
            f"Branch cache has no branch containing parent entry {parent_id}",
        )
    branch_id = uuidv7()
    await copy_branch_entries_through_seq(
        db,
        session_id,
        branch_id,
        str(source["branch_id"]),
        int(cast(int, source["entry_seq"])),
    )
    await insert_branch_entry(
        db, session_id, branch_id, entry_id, entry_seq, entry_type, custom_type
    )
    await insert_branch_tip(db, session_id, entry_id, branch_id)


__all__ = [
    "append_entry_to_branch_cache",
    "build_cached_branch",
    "delete_branch_cache",
    "rebuild_branch_cache",
]