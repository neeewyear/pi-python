"""SQLite 分支条目表存储。"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from .._types import SqliteDatabase


class CachedBranch:
    """缓存分支信息。"""

    def __init__(self, branch_id: str, leaf_seq: int) -> None:
        self.branch_id = branch_id
        self.leaf_seq = leaf_seq


async def read_cached_branch(
    db: SqliteDatabase, session_id: str, leaf_id: str
) -> CachedBranch | None:
    """读取缓存分支。"""
    row = await (
        await db.prepare(
            "SELECT branch_id, entry_seq FROM branch_entries "
            "WHERE session_id = ? AND entry_id = ? ORDER BY branch_id LIMIT 1"
        )
    ).get(session_id, leaf_id)
    if not row:
        return None
    return CachedBranch(
        branch_id=str(row["branch_id"]), leaf_seq=int(cast(int, row["entry_seq"]))
    )


async def query_cached_branch_rows(
    db: SqliteDatabase,
    session_id: str,
    branch: CachedBranch,
    query: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """查询缓存分支行。"""
    q: dict[str, object] = query or {}
    oldest_first = q.get("order") == "oldestFirst"
    boundary_params: list[object] = [session_id, branch.branch_id, branch.leaf_seq]
    stop_predicates: list[str] = []
    stop_at_type = q.get("stop_at_type")
    if stop_at_type is not None:
        stop_predicates.append("stop_entry.type = ?")
        boundary_params.append(stop_at_type)
    stop_at_id = q.get("stop_at_id")
    if stop_at_id is not None:
        stop_predicates.append("stop.entry_id = ?")
        boundary_params.append(stop_at_id)

    boundary = ""
    range_clause = ""
    if stop_predicates:
        boundary = (
            f"WITH boundary AS ("
            f"SELECT {'MIN' if oldest_first else 'MAX'}(stop.entry_seq) AS entry_seq "
            f"FROM branch_entries AS stop "
            f"JOIN entries AS stop_entry ON stop_entry.session_id = stop.session_id AND stop_entry.id = stop.entry_id "
            f"WHERE stop.session_id = ? AND stop.branch_id = ? AND stop.entry_seq <= ? "
            f"AND ({' OR '.join(stop_predicates)})"
            f")"
        )
        range_clause = (
            f"AND b.entry_seq {'<=' if oldest_first else '>='} COALESCE("
            f"(SELECT entry_seq FROM boundary), {branch.leaf_seq if oldest_first else 0}"
            f")"
        )

    params = (
        boundary_params
        if not stop_predicates
        else [*boundary_params, session_id, branch.branch_id, branch.leaf_seq]
    )
    sql = (
        f"{boundary} "
        f"SELECT e.session_id, e.id, e.seq AS entry_seq, e.parent_id, e.type, e.timestamp, e.payload "
        f"FROM branch_entries AS b "
        f"JOIN entries AS e ON e.session_id = b.session_id AND e.id = b.entry_id "
        f"WHERE b.session_id = ? AND b.branch_id = ? AND b.entry_seq <= ? "
        f"{range_clause} "
        f"ORDER BY b.entry_seq {'ASC' if oldest_first else 'DESC'}"
    )
    return await (await db.prepare(sql)).all(*params)


async def delete_branch_entries(db: SqliteDatabase, session_id: str) -> None:
    """删除会话的所有分支条目。"""
    await (await db.prepare("DELETE FROM branch_entries WHERE session_id = ?")).run(
        session_id
    )


async def insert_branch_entry(
    db: SqliteDatabase,
    session_id: str,
    branch_id: str,
    entry_id: str,
    entry_seq: int,
    entry_type: str,
    custom_type: str | None,
) -> None:
    """插入分支条目。"""
    await (
        await db.prepare(
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type, custom_type) "
            "VALUES (?, ?, ?, ?, ?, ?)"
        )
    ).run(session_id, branch_id, entry_id, entry_seq, entry_type, custom_type)


async def insert_branch_entries_for_path(
    db: SqliteDatabase, session_id: str, branch_id: str, leaf_id: str
) -> None:
    """插入从叶子到根的分支路径。"""
    await (
        await db.prepare(
            "WITH RECURSIVE path(id, entry_seq, parent_id, type, custom_type) AS ("
            "SELECT id, seq, parent_id, type, "
            "CASE WHEN type = 'custom' THEN json_extract(payload, '$.customType') ELSE NULL END "
            "FROM entries "
            "WHERE session_id = ? AND id = ? "
            "UNION ALL "
            "SELECT parent.id, parent.seq, parent.parent_id, parent.type, "
            "CASE WHEN parent.type = 'custom' THEN json_extract(parent.payload, '$.customType') ELSE NULL END "
            "FROM entries AS parent "
            "JOIN path AS child ON child.parent_id = parent.id "
            "WHERE parent.session_id = ?"
            ") "
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type, custom_type) "
            "SELECT ?, ?, id, entry_seq, type, custom_type FROM path"
        )
    ).run(session_id, leaf_id, session_id, session_id, branch_id)


async def read_branch_containing_entry(
    db: SqliteDatabase, session_id: str, entry_id: str
) -> dict[str, object] | None:
    """读取包含指定条目的分支。"""
    row = await (
        await db.prepare(
            "SELECT b.branch_id, b.entry_seq "
            "FROM branch_entries AS b "
            "WHERE b.session_id = ? AND b.entry_id = ? "
            "ORDER BY b.branch_id "
            "LIMIT 1"
        )
    ).get(session_id, entry_id)
    if not row:
        return None
    return {
        "branch_id": str(row["branch_id"]),
        "entry_seq": int(cast(int, row["entry_seq"])),
    }


async def copy_branch_entries_through_seq(
    db: SqliteDatabase,
    session_id: str,
    target_branch_id: str,
    source_branch_id: str,
    through_seq: int,
) -> None:
    """复制分支条目到指定序列。"""
    await (
        await db.prepare(
            "INSERT INTO branch_entries (session_id, branch_id, entry_id, entry_seq, entry_type, custom_type) "
            "SELECT session_id, ?, entry_id, entry_seq, entry_type, custom_type "
            "FROM branch_entries "
            "WHERE session_id = ? AND branch_id = ? AND entry_seq <= ?"
        )
    ).run(target_branch_id, session_id, source_branch_id, through_seq)


__all__ = [
    "CachedBranch",
    "copy_branch_entries_through_seq",
    "delete_branch_entries",
    "insert_branch_entries_for_path",
    "insert_branch_entry",
    "query_cached_branch_rows",
    "read_branch_containing_entry",
    "read_cached_branch",
]
