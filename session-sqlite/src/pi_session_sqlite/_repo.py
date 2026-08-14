"""SQLite 会话仓库与存储实现。"""


from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any, cast

from pi_session.session import Session
from pi_session.types import SessionError

from ._branch_cache import (
    append_entry_to_branch_cache,
    build_cached_branch,
    delete_branch_cache,
    rebuild_branch_cache,
)
from ._migrations import apply_migrations
from ._types import (
    SqliteDatabase,
    SqliteDatabaseFactory,
    SqliteSessionCreateOptions,
    SqliteSessionListOptions,
    SqliteSessionMetadata,
    SqliteSessionRepositoryEnv,
)
from .storage.branch_entries import (
    query_cached_branch_rows,
    read_cached_branch,
)
from .storage.branch_tips import read_branch_tip_ids
from .storage.entries import (
    NewEntryRow,
    delete_entry_rows,
    entry_payload,
    id_exists_in_entries,
    insert_entry_row,
    read_entry_row,
    read_entry_rows,
)
from .storage.facts import (
    append_fact,
    delete_fact_rows,
    read_fact_rows,
    read_latest_fact,
    read_latest_label_facts,
)
from .storage.lanes import (
    create_initial_lane,
    create_lane,
    delete_lane_rows,
    move_lane,
    read_lane,
    read_lane_head,
    read_lane_move_rows,
    read_lanes,
    set_lane_leaf,
)
from .storage.leases import (
    SessionLease,
    acquire_session_lease,
    delete_session_lease,
    release_session_lease,
    renew_session_lease,
)
from .storage.records import (
    NewRecordRow,
    append_record_row,
    delete_record_rows,
    id_exists_in_records,
    read_open_operation_rows,
    read_record_rows,
)
from .storage.session_sequences import (
    advance_sequence,
    create_sequence,
    delete_sequence,
    get_next_sequence,
    set_next_sequence,
)
from .storage.session_stats import (
    add_usage_to_stats,
    create_stats,
    delete_stats,
    increment_message_count,
    read_stats,
)
from .storage.sessions import (
    NewSessionRow,
    delete_session_row,
    insert_session_row,
    read_session_row,
    read_session_rows,
    row_to_metadata,
    session_exists,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from pi_session.types import (
        BranchBounds,
        Entry,
        EntryQuery,
        LanePointer,
        LaneRecord,
        LogItem,
        NewRecord,
        OperationStartedRecord,
        ProvisionedEntry,
        RecordQuery,
        SessionStats,
    )


class SqliteWriterLeaseOptions:
    """写租约选项。"""

    def __init__(
        self,
        *,
        ttl_ms: int = 30_000,
        heartbeat_interval_ms: int = 10_000,
    ) -> None:
        self.ttl_ms = ttl_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms


class ResolvedWriterLeaseOptions:
    """已解析的写租约选项。"""

    def __init__(self, ttl_ms: int, heartbeat_interval_ms: int) -> None:
        self.ttl_ms = ttl_ms
        self.heartbeat_interval_ms = heartbeat_interval_ms


class SqliteSessionRepositoryOptions:
    """SQLite 会话仓库选项。"""

    def __init__(
        self,
        *,
        env: SqliteSessionRepositoryEnv,
        sqlite: SqliteDatabaseFactory,
        database_path: str,
        writer_lease: SqliteWriterLeaseOptions | None = None,
    ) -> None:
        self.env = env
        self.sqlite = sqlite
        self.database_path = database_path
        self.writer_lease = writer_lease


def _resolve_writer_lease_options(
    options: SqliteWriterLeaseOptions | None,
) -> ResolvedWriterLeaseOptions:
    """解析写租约选项。"""
    ttl_ms = options.ttl_ms if options and options.ttl_ms else 30_000
    heartbeat_interval_ms = (
        options.heartbeat_interval_ms
        if options and options.heartbeat_interval_ms
        else 10_000
    )
    if not isinstance(ttl_ms, int) or ttl_ms <= 0:
        raise RangeError("writerLease.ttlMs must be positive")
    if (
        not isinstance(heartbeat_interval_ms, int)
        or heartbeat_interval_ms <= 0
        or heartbeat_interval_ms >= ttl_ms
    ):
        raise RangeError(
            "writerLease.heartbeatIntervalMs must be positive and less than ttlMs"
        )
    return ResolvedWriterLeaseOptions(ttl_ms, heartbeat_interval_ms)


class RangeError(Exception):
    """范围错误。"""


def _active_writer_error(session_id: str) -> SessionError:
    return SessionError(
        "storage", f"SQLite session {session_id} already has an active writer"
    )


def _lost_writer_error(session_id: str) -> SessionError:
    return SessionError("storage", f"SQLite session {session_id} writer lease was lost")


async def _acquire_writer_lease(
    db: SqliteDatabase,
    session_id: str,
    options: ResolvedWriterLeaseOptions,
) -> SessionLease:
    """获取写租约。"""
    from pi_agent.uuid7 import uuidv7

    now = _now_ms()
    lease = await acquire_session_lease(
        db, session_id, uuidv7(), now, now + options.ttl_ms
    )
    if not lease:
        raise _active_writer_error(session_id)
    return lease


class SerialOperationQueue:
    """串行操作队列。"""

    def __init__(self) -> None:
        self._tail: asyncio.Future[Any] = asyncio.Future()
        self._tail.set_result(None)

    async def enqueue(self, operation: Callable[[], Any]) -> Any:
        """将操作加入队列（串行执行）。"""
        await self._tail
        new_tail: asyncio.Future[Any] = asyncio.Future()
        self._tail = new_tail
        try:
            res = await operation()
            new_tail.set_result(None)
            return res
        except BaseException:
            new_tail.set_result(None)
            raise

    async def drain(self) -> None:
        """等待队列清空。"""
        await self._tail


def _now_ms() -> int:
    """返回当前时间戳（毫秒）。"""
    import time

    return int(time.time() * 1000)


def _timestamp_to_text(timestamp: int) -> str:
    """将时间戳转为 ISO 文本。"""
    import datetime

    return datetime.datetime.fromtimestamp(
        timestamp / 1000, tz=datetime.timezone.utc
    ).isoformat()


def _timestamp_from_text(timestamp: str) -> int:
    """将 ISO 文本转为时间戳。"""
    import datetime

    dt = datetime.datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return int(dt.timestamp() * 1000)


def _get_parent_path(path: str) -> str:
    """获取父目录路径。"""
    import os

    normalized = path.rstrip("/\\")
    return os.path.dirname(normalized) or "."


async def _configure_sqlite_database(db: SqliteDatabase) -> None:
    """配置 SQLite 数据库。"""
    await db.exec("PRAGMA journal_mode=WAL")
    await db.exec("PRAGMA synchronous=FULL")
    await db.exec("PRAGMA busy_timeout=5000")


def _entry_row_from_cached(row: dict[str, object]) -> dict[str, object]:
    """从缓存行转换为条目行。"""
    result = dict(row)
    result["seq"] = row["entry_seq"]
    return result


def _read_object_payload(row: dict[str, object]) -> dict[str, object]:
    """解析条目负载为对象。"""
    payload_str = row.get("payload", "")
    assert isinstance(payload_str, str)
    import orjson

    payload = orjson.loads(payload_str)
    if not isinstance(payload, dict):
        raise SessionError("invalid_entry", "Payload is not an object")
    return payload


def _decode_entry(row: dict[str, object]) -> Entry:
    """解码条目行。"""
    from pi_session.types import (
        ActiveToolsEntry,
        BranchSummaryEntry,
        CompactionEntry,
        CustomEntry,
        MessageEntry,
        ModelChangeEntry,
        ThinkingLevelEntry,
    )

    try:
        payload = _read_object_payload(row)
        timestamp = _timestamp_from_text(str(row["timestamp"]))
        if timestamp == 0 and str(row["timestamp"]) != "0":
            raise SessionError("invalid_entry", f"Invalid timestamp {row['timestamp']}")
        entry_id = str(cast(str, row["id"]))
        entry_seq = int(cast(int, row["seq"]))
        entry_parent_id = (
            str(cast(str, row["parent_id"])) if row.get("parent_id") else None
        )
        entry_type = str(cast(str, row["type"]))
        if entry_type == "message":
            msg = payload.get("message")
            if not isinstance(msg, dict):
                raise SessionError("invalid_entry", "Missing message")
            terminate_val = payload.get("terminate")
            return MessageEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                message=cast(Any, msg),
                terminate=cast(Any, terminate_val)
                if terminate_val is not None
                else None,
            )
        elif entry_type == "model_change":
            return ModelChangeEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                provider=str(payload["provider"]),
                model_id=str(payload["modelId"]),
            )
        elif entry_type == "thinking_level_change":
            return ThinkingLevelEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                thinking_level=str(payload["thinkingLevel"]),
            )
        elif entry_type == "active_tools_change":
            active_tool_names = payload.get("activeToolNames", [])
            if not isinstance(active_tool_names, list):
                raise SessionError(
                    "invalid_entry", "Invalid active_tools_change payload"
                )
            return ActiveToolsEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                active_tool_names=active_tool_names,
            )
        elif entry_type == "compaction":
            retained_tail = cast(Any, payload.get("retainedTail", []))
            usage = cast(Any, payload.get("usage"))
            return CompactionEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                summary=str(payload["summary"]),
                retained_tail=retained_tail,
                tokens_before=int(cast(int, payload["tokensBefore"])),
                details=payload.get("details"),
                usage=usage,
            )
        elif entry_type == "branch_summary":
            return BranchSummaryEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                from_id=str(payload["fromId"]),
                summary=str(payload["summary"]),
                details=payload.get("details"),
                usage=cast(Any, payload.get("usage")),
            )
        elif entry_type == "custom":
            return CustomEntry(
                id=entry_id,
                seq=entry_seq,
                parent_id=entry_parent_id,
                timestamp=timestamp,
                custom_type=str(payload["customType"]),
                data=payload.get("data"),
            )
        else:
            raise SessionError("invalid_entry", f"Unknown entry type: {entry_type}")
    except SessionError:
        raise
    except Exception as error:
        raise SessionError(
            "invalid_entry",
            f"Invalid SQLite session entry {row.get('id')}: failed to decode",
            error,
        ) from error


def _decode_record(row: dict[str, object]) -> LaneRecord:
    """解码记录行。"""
    try:
        timestamp = _timestamp_from_text(str(cast(str, row["timestamp"])))
        if timestamp == 0 and str(row["timestamp"]) != "0":
            raise SessionError("storage", f"Invalid timestamp {row['timestamp']}")
        payload_str = str(row.get("payload", "{}"))
        import orjson

        payload = orjson.loads(payload_str)
        if not isinstance(payload, dict):
            raise SessionError("storage", "Invalid record payload")
        payload["seq"] = int(cast(int, row["seq"]))
        payload["timestamp"] = timestamp
        from pi_session.types import LaneRecord as LaneRecordAlias
        from pydantic import TypeAdapter

        adapter: TypeAdapter[LaneRecordAlias] = TypeAdapter(LaneRecordAlias)
        return adapter.validate_python(payload)
    except SessionError:
        raise
    except Exception as error:
        raise SessionError(
            "storage",
            f"Invalid SQLite session record at sequence {row.get('seq')}: failed to decode payload",
            error,
        ) from error


def _validate_cached_branch_rows(
    rows: list[dict[str, object]],
    query: BranchBounds,
) -> None:
    """验证缓存分支行。"""
    if not rows:
        return
    path = sorted(rows, key=lambda r: int(cast(int, r["entry_seq"])))
    if (
        query.stop_at_id is None
        and query.stop_at_type is None
        and path[0].get("parent_id") is not None
    ):
        raise SessionError(
            "invalid_entry", f"Entry {path[0].get('parent_id')} not found"
        )
    for index in range(1, len(path)):
        previous = path[index - 1]
        current = path[index]
        if current.get("parent_id") != previous.get("id"):
            raise SessionError(
                "invalid_entry", f"Entry {current.get('parent_id')} not found"
            )


def _matches_entry_query(entry: Entry, query: EntryQuery) -> bool:
    """检查条目是否匹配查询。"""
    if query.type is not None and entry.type != query.type:
        return False
    if query.custom_type is not None:
        if entry.type != "custom":
            return False
        if hasattr(entry, "custom_type") and entry.custom_type != query.custom_type:
            return False
    if query.cursor is not None:
        if query.order == "oldestFirst":
            if entry.seq <= query.cursor.after_seq:
                return False
        else:
            if entry.seq >= query.cursor.after_seq:
                return False
    return True


async def _assert_unused_id(db: SqliteDatabase, session_id: str, entry_id: str) -> None:
    """断言 ID 未被使用。"""
    if await id_exists_in_entries(
        db, session_id, entry_id
    ) or await id_exists_in_records(db, session_id, entry_id):
        raise SessionError("already_exists", f"ID already exists: {entry_id}")


async def _require_session_row(
    db: SqliteDatabase, session_id: str
) -> dict[str, object]:
    """要求会话行存在。"""
    row = await read_session_row(db, session_id)
    if not row:
        raise SessionError("not_found", f"Session not found: {session_id}")
    return row


def _metadata_from_row(row: dict[str, object], path: str) -> SqliteSessionMetadata:
    """从行创建元数据。"""
    return row_to_metadata(row, path)


class SqliteSessionStorage:
    """SQLite 会话存储实现。"""

    def __init__(
        self,
        db: SqliteDatabase,
        metadata: SqliteSessionMetadata,
        lease: SessionLease,
        lease_options: ResolvedWriterLeaseOptions,
        on_release: Callable[[], None],
    ) -> None:
        self._db = db
        self._metadata = metadata
        self._lease = lease
        self._lease_options = lease_options
        self._on_release = on_release
        self._operations = SerialOperationQueue()
        self._heartbeat_timer: asyncio.TimerHandle | None = None
        self._lease_error: SessionError | None = None
        self._closing = False
        self._release_promise: asyncio.Future[None] | None = None
        self._schedule_heartbeat()

    async def release(self) -> None:
        """释放存储。"""
        if self._release_promise is None:
            self._release_promise = asyncio.ensure_future(self._finish_release())
        await self._release_promise

    async def _finish_release(self) -> None:
        """完成释放。"""
        self._closing = True
        if self._heartbeat_timer is not None:
            self._heartbeat_timer.cancel()
            self._heartbeat_timer = None
        try:

            async def _release_tx() -> None:
                await release_session_lease(self._db, self._metadata.id, self._lease)

            await self._operations.enqueue(lambda: self._db.transaction(_release_tx))
        finally:
            self._on_release()

    async def _enqueue_write(self, operation: Callable[[], Awaitable[object]]) -> Any:
        """将写操作入队。"""
        if self._closing:
            raise SessionError(
                "storage", f"SQLite session {self._metadata.id} is closed"
            )

        async def _tx_wrapper() -> object:
            if self._lease_error:
                raise self._lease_error
            now = _now_ms()
            if not await renew_session_lease(
                self._db,
                self._metadata.id,
                self._lease,
                now,
                now + self._lease_options.ttl_ms,
            ):
                self._lease_error = _lost_writer_error(self._metadata.id)
                if self._heartbeat_timer is not None:
                    self._heartbeat_timer.cancel()
                    self._heartbeat_timer = None
                raise self._lease_error
            return await operation()

        return await self._operations.enqueue(lambda: self._db.transaction(_tx_wrapper))

    def _schedule_heartbeat(self) -> None:
        """调度心跳。"""
        if self._closing or self._lease_error:
            return

        async def _heartbeat() -> None:
            try:
                await self._operations.enqueue(
                    lambda: self._db.transaction(self._renew_lease_inner)
                )
            except BaseException:
                pass
            finally:
                self._schedule_heartbeat()

        loop = asyncio.get_event_loop()
        self._heartbeat_timer = loop.call_later(
            self._lease_options.heartbeat_interval_ms / 1000,
            lambda: asyncio.ensure_future(_heartbeat()),
        )

    async def _renew_lease_inner(self) -> None:
        """内部续期租约。"""
        if self._closing or self._lease_error:
            return
        now = _now_ms()
        if not await renew_session_lease(
            self._db,
            self._metadata.id,
            self._lease,
            now,
            now + self._lease_options.ttl_ms,
        ):
            self._lease_error = _lost_writer_error(self._metadata.id)

    async def get_metadata(self) -> SqliteSessionMetadata:
        """获取元数据。"""
        return self._metadata.model_copy(deep=True)

    def is_for_session(self, session_id: str) -> bool:
        """检查是否属于指定会话。"""
        return self._metadata.id == session_id

    async def get_lanes(self) -> list[LanePointer]:
        """获取所有 lanes。"""
        from pi_session.types import LanePointer

        result: list[LanePointer] = []
        for r in await read_lanes(self._db, self._metadata.id):
            leaf_id_val = r.get("leaf_id")
            result.append(
                LanePointer(
                    lane=str(cast(str, r["lane"])),
                    leaf_id=str(cast(str, leaf_id_val))
                    if leaf_id_val is not None
                    else None,
                )
            )
        return result

    async def create_lane(self, lane: str, at: str | None) -> None:
        """创建 lane。"""
        return await self._enqueue_write(lambda: self._create_lane_tx(lane, at))  # type: ignore[no-any-return]

    async def _create_lane_tx(self, lane: str, at: str | None) -> None:
        """创建 lane 事务。"""
        if await read_lane(self._db, self._metadata.id, lane):
            raise SessionError("already_exists", f"Lane already exists: {lane}")
        if at is not None and not await read_entry_row(self._db, self._metadata.id, at):
            raise SessionError("not_found", f"Entry not found: {at}")
        seq = await get_next_sequence(self._db, self._metadata.id)
        await create_lane(self._db, self._metadata.id, seq, lane, at)
        await advance_sequence(self._db, self._metadata.id, seq)

    async def move_lane(self, lane: str, to: str | None) -> None:
        """移动 lane。"""
        return await self._enqueue_write(lambda: self._move_lane_tx(lane, to))  # type: ignore[no-any-return]

    async def _move_lane_tx(self, lane: str, to: str | None) -> None:
        """移动 lane 事务。"""
        if not await read_lane(self._db, self._metadata.id, lane):
            raise SessionError("invalid_lane", f"Lane not found: {lane}")
        if to is not None and not await read_entry_row(self._db, self._metadata.id, to):
            raise SessionError("not_found", f"Entry not found: {to}")
        seq = await get_next_sequence(self._db, self._metadata.id)
        await move_lane(self._db, self._metadata.id, seq, lane, to)
        await advance_sequence(self._db, self._metadata.id, seq)

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        """追加条目。"""
        return await self._enqueue_write(lambda: self._append_entry_tx(entry, lane))  # type: ignore[no-any-return]

    async def _append_entry_tx(self, entry: ProvisionedEntry, lane: str) -> Entry:
        """追加条目事务。"""
        import orjson
        from pi_session.types import Entry as EntryType

        parent_id = (await read_lane_head(self._db, self._metadata.id, lane)).get(
            "leaf_id"
        )
        await _assert_unused_id(self._db, self._metadata.id, entry.id)
        seq = await get_next_sequence(self._db, self._metadata.id)
        committed: EntryType = entry.model_copy(
            update={
                "parent_id": parent_id,
                "seq": seq,
                "timestamp": _now_ms(),
            }
        )
        entry_dict = committed.model_dump(mode="json")
        await insert_entry_row(
            self._db,
            self._metadata.id,
            NewEntryRow(
                seq=seq,
                id=committed.id,
                parent_id=committed.parent_id,
                type=committed.type,
                timestamp=_timestamp_to_text(committed.timestamp),
                payload=orjson.dumps(entry_payload(entry_dict)).decode("utf-8"),
            ),
        )
        await set_lane_leaf(self._db, self._metadata.id, lane, committed.id)
        await append_entry_to_branch_cache(
            self._db,
            self._metadata.id,
            committed.id,
            seq,
            committed.type,
            committed.custom_type
            if hasattr(committed, "custom_type") and committed.type == "custom"
            else None,
            committed.parent_id,
        )
        if committed.type == "message":
            await increment_message_count(self._db, self._metadata.id)
        await advance_sequence(self._db, self._metadata.id, seq)
        return committed.model_copy(deep=True)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        """追加记录。"""
        return await self._enqueue_write(lambda: self._append_record_tx(record))  # type: ignore[no-any-return]

    async def _append_record_tx(self, record: NewRecord) -> LaneRecord:
        """追加记录事务。"""
        if not await read_lane(self._db, self._metadata.id, record.lane):
            raise SessionError("invalid_lane", f"Lane not found: {record.lane}")
        await _assert_unused_id(self._db, self._metadata.id, record.id)
        seq = await get_next_sequence(self._db, self._metadata.id)
        committed: LaneRecord = record.model_copy(
            update={
                "seq": seq,
                "timestamp": _now_ms(),
            }
        )

        def _record_run_id(r: NewRecord) -> str | None:
            if r.type == "operation_started":
                return r.id
            return getattr(r, "run_id", None)

        def _record_op_kind(r: NewRecord) -> str | None:
            if r.type == "operation_started":
                return r.intent.kind if hasattr(r, "intent") else None
            return None

        await append_record_row(
            self._db,
            self._metadata.id,
            NewRecordRow(
                seq=seq,
                id=record.id,
                lane=record.lane,
                run_id=_record_run_id(record),
                type=record.type,
                op_kind=_record_op_kind(record),
                timestamp=_timestamp_to_text(committed.timestamp),
                payload=committed.model_dump_json(),
            ),
        )
        if record.type == "usage":
            await add_usage_to_stats(
                self._db, self._metadata.id, record.usage.model_dump()
            )
        await advance_sequence(self._db, self._metadata.id, seq)
        return committed.model_copy(deep=True)

    async def get_entry(self, entry_id: str) -> Entry | None:
        """获取条目。"""
        row = await read_entry_row(self._db, self._metadata.id, entry_id)
        return _decode_entry(row) if row else None

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        """查找条目。"""
        q = query or EntryQuery()
        rows = await read_entry_rows(self._db, self._metadata.id, {"order": q.order})
        entries = [
            _decode_entry(r) for r in rows if _matches_entry_query(_decode_entry(r), q)
        ]
        return entries[: q.limit] if q.limit is not None else entries

    async def find_entries_on_branch(
        self,
        query: Any,  # BranchBounds & EntryQuery
    ) -> list[Entry]:
        """查找分支上的条目。"""
        cached = await read_cached_branch(self._db, self._metadata.id, query.start)
        if not cached:
            if not await read_entry_row(self._db, self._metadata.id, query.start):
                raise SessionError("not_found", f"Entry not found: {query.start}")
            raise SessionError(
                "invalid_entry", f"Branch cache missing entry {query.start}"
            )
        rows = await query_cached_branch_rows(
            self._db, self._metadata.id, cached, query.model_dump(exclude_none=True)
        )
        _validate_cached_branch_rows(rows, query)
        entries = [
            _decode_entry(_entry_row_from_cached(r))
            for r in rows
            if _matches_entry_query(_decode_entry(_entry_row_from_cached(r)), query)
        ]
        return entries[: query.limit] if query.limit is not None else entries

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        """查找记录。"""
        q = query or RecordQuery()
        rows = await read_record_rows(
            self._db, self._metadata.id, q.model_dump(exclude_none=True)
        )
        return [_decode_record(r) for r in rows]

    async def find_open_operations(
        self, lane: str, options: dict[str, object] | None = None
    ) -> list[OperationStartedRecord]:
        """查找未结束的操作。"""
        opts: dict[str, object] = options or {}
        rows = await read_open_operation_rows(self._db, self._metadata.id, lane, opts)
        result: list[OperationStartedRecord] = []
        for row in rows:
            record = _decode_record(row)
            if record.type != "operation_started":
                raise SessionError("storage", "Expected operation_started record")
            result.append(record)
        return result

    async def get_log(self, options: dict[str, object] | None = None) -> list[LogItem]:
        """获取日志。"""
        import orjson
        from pi_session.types import (
            LogEntryItem,
            LogLabelFactItem,
            LogLaneItem,
            LogNameFactItem,
            LogRecordItem,
        )

        opts: dict[str, object] = options or {}
        after_seq = cast(int, opts.get("after_seq", 0))
        entry_rows = await read_entry_rows(
            self._db,
            self._metadata.id,
            {"after_seq": after_seq, "order": "oldestFirst"},
        )
        record_rows = await read_record_rows(
            self._db, self._metadata.id, {"after_seq": after_seq}
        )
        lane_rows = await read_lane_move_rows(
            self._db, self._metadata.id, {"after_seq": after_seq}
        )
        fact_rows = await read_fact_rows(
            self._db, self._metadata.id, {"after_seq": after_seq}
        )

        log: list[LogItem] = []
        for row in entry_rows:
            log.append(
                LogEntryItem(seq=int(cast(int, row["seq"])), entry=_decode_entry(row))
            )
        for row in record_rows:
            log.append(
                LogRecordItem(
                    seq=int(cast(int, row["seq"])), record=_decode_record(row)
                )
            )
        for row in lane_rows:
            leaf_id_val = row.get("leaf_id")
            log.append(
                LogLaneItem(
                    seq=int(cast(int, row["seq"])),
                    lane=str(cast(str, row["lane"])),
                    leaf_id=str(cast(str, leaf_id_val))
                    if leaf_id_val is not None
                    else None,
                )
            )
        for row in fact_rows:
            kind = str(cast(str, row.get("kind", "")))
            if kind == "name":
                value = row.get("value")
                name = orjson.loads(cast(str, value)) if value else ""
                log.append(LogNameFactItem(seq=int(cast(int, row["seq"])), name=name))
            else:
                value = row.get("value")
                label = orjson.loads(cast(str, value)) if value is not None else None
                log.append(
                    LogLabelFactItem(
                        seq=int(cast(int, row["seq"])),
                        target_id=str(cast(str, row.get("key", ""))),
                        label=label,
                    )
                )

        log.sort(key=lambda item: item.seq)
        limit_val = opts.get("limit")
        limit = cast(int, limit_val) if limit_val is not None else None
        return log[:limit] if limit is not None else log

    async def get_name(self) -> str | None:
        """获取会话名称。"""
        row = await read_latest_fact(self._db, self._metadata.id, "name", None)
        if not row or row.get("value") is None:
            return None
        import orjson

        return orjson.loads(cast(str, row["value"]))  # type: ignore[no-any-return]

    async def set_name(self, name: str) -> None:
        """设置会话名称。"""
        return await self._enqueue_write(lambda: self._set_name_tx(name))  # type: ignore[no-any-return]

    async def _set_name_tx(self, name: str) -> None:
        """设置名称事务。"""
        import orjson

        seq = await get_next_sequence(self._db, self._metadata.id)
        await append_fact(
            self._db,
            self._metadata.id,
            seq,
            "name",
            None,
            orjson.dumps(name).decode("utf-8"),
        )
        await advance_sequence(self._db, self._metadata.id, seq)

    async def get_label(self, entry_id: str) -> str | None:
        """获取条目标签。"""
        row = await read_latest_fact(self._db, self._metadata.id, "label", entry_id)
        if not row or row.get("value") is None:
            return None
        import orjson

        return orjson.loads(cast(str, row["value"]))  # type: ignore[no-any-return]

    async def set_label(self, entry_id: str, label: str | None) -> None:
        """设置条目标签。"""
        return await self._enqueue_write(lambda: self._set_label_tx(entry_id, label))  # type: ignore[no-any-return]

    async def _set_label_tx(self, entry_id: str, label: str | None) -> None:
        """设置标签事务。"""
        if not await read_entry_row(self._db, self._metadata.id, entry_id):
            raise SessionError("not_found", f"Entry not found: {entry_id}")
        seq = await get_next_sequence(self._db, self._metadata.id)
        import orjson

        value = orjson.dumps(label).decode("utf-8") if label is not None else None
        await append_fact(self._db, self._metadata.id, seq, "label", entry_id, value)
        await advance_sequence(self._db, self._metadata.id, seq)

    async def get_stats(self) -> SessionStats:
        """获取统计。"""
        row = await read_stats(self._db, self._metadata.id)
        from pi_session.types import SessionStats

        return SessionStats(
            message_count=int(cast(int, row["message_count"])),
            cached_tokens=int(cast(int, row["cached_tokens"])),
            uncached_tokens=int(cast(int, row["uncached_tokens"])),
            total_tokens=int(cast(int, row["total_tokens"])),
            cost_total=float(cast(float, row["cost_total"])),
        )


async def _claim_storage(
    db: SqliteDatabase,
    metadata: SqliteSessionMetadata,
    lease_options: ResolvedWriterLeaseOptions,
    on_release: Callable[[], None],
) -> SqliteSessionStorage:
    """获取存储。"""
    await _require_session_row(db, metadata.id)

    async def _claim_tx() -> dict[str, object]:
        lease = await _acquire_writer_lease(db, metadata.id, lease_options)
        row = await _require_session_row(db, metadata.id)
        await read_lanes(db, metadata.id)
        return {"lease": lease, "row": row}

    claimed = cast(dict[str, object], await db.transaction(_claim_tx))
    return SqliteSessionStorage(
        db,
        _metadata_from_row(cast(dict[str, object], claimed["row"]), metadata.path),
        cast(SessionLease, claimed["lease"]),
        lease_options,
        on_release,
    )


class SqliteSessionRepository:
    """SQLite 会话仓库。"""

    def __init__(self, options: SqliteSessionRepositoryOptions) -> None:
        self._options = options
        self._lease_options = _resolve_writer_lease_options(options.writer_lease)
        self._database_path: str | None = None
        self._database: SqliteDatabase | None = None
        self._database_promise: asyncio.Future[SqliteDatabase] | None = None
        self._operations = SerialOperationQueue()
        self._active_storages: set[SqliteSessionStorage] = set()

    async def _release_storages_for_session(self, session_id: str) -> None:
        """释放会话的所有存储。"""
        for storage in list(self._active_storages):
            if storage.is_for_session(session_id):
                await storage.release()

    def _session_from_lease(
        self,
        db: SqliteDatabase,
        metadata: SqliteSessionMetadata,
        lease: SessionLease,
    ) -> Session:
        """从租约创建会话。"""
        storage = SqliteSessionStorage(
            db,
            metadata,
            lease,
            self._lease_options,
            lambda: self._active_storages.discard(storage),
        )
        self._active_storages.add(storage)
        return Session(storage)

    async def _claim_session(
        self, db: SqliteDatabase, metadata: SqliteSessionMetadata
    ) -> Session:
        """获取会话。"""
        for storage in self._active_storages:
            if storage.is_for_session(metadata.id):
                await read_lanes(db, metadata.id)
                return Session(storage)
        storage = await _claim_storage(
            db,
            metadata,
            self._lease_options,
            lambda: self._active_storages.discard(storage),
        )
        self._active_storages.add(storage)
        return Session(storage)

    async def create(self, options: SqliteSessionCreateOptions) -> Session:
        """创建会话。"""
        from pi_agent.uuid7 import uuidv7

        async def _create() -> Session:
            db = await self._get_database()
            path = await self._get_database_path()
            session_id = options.id or uuidv7()
            if await session_exists(db, session_id):
                raise SessionError(
                    "already_exists", f"Session already exists: {session_id}"
                )
            created_at = _now_ms()

            async def _create_tx() -> SessionLease:
                await insert_session_row(
                    db,
                    NewSessionRow(
                        id=session_id,
                        created_at=_timestamp_to_text(created_at),
                        cwd=options.cwd,
                        parent_session_id=options.parent_session_id,
                        metadata=options.metadata,
                    ),
                )
                await create_sequence(db, session_id)
                await create_stats(db, session_id)
                await create_initial_lane(db, session_id)
                return await _acquire_writer_lease(db, session_id, self._lease_options)

            lease = cast(SessionLease, await db.transaction(_create_tx))
            row = await _require_session_row(db, session_id)
            return self._session_from_lease(db, _metadata_from_row(row, path), lease)

        return await self._operations.enqueue(_create)  # type: ignore[no-any-return]

    async def open(self, metadata: SqliteSessionMetadata) -> Session:
        """打开会话。"""

        async def _open() -> Session:
            return await self._claim_session(await self._get_database(), metadata)

        return await self._operations.enqueue(_open)  # type: ignore[no-any-return]

    async def repair_branch_cache(self, metadata: SqliteSessionMetadata) -> None:
        """修复分支缓存。"""

        async def _repair() -> None:
            await self._release_storages_for_session(metadata.id)
            db = await self._get_database()

            async def _repair_tx() -> None:
                lease = await _acquire_writer_lease(
                    db, metadata.id, self._lease_options
                )
                await _require_session_row(db, metadata.id)
                await rebuild_branch_cache(db, metadata.id)
                await release_session_lease(db, metadata.id, lease)

            await db.transaction(_repair_tx)

        return await self._operations.enqueue(_repair)  # type: ignore[no-any-return]

    async def list(
        self, options: SqliteSessionListOptions | None = None
    ) -> list[SqliteSessionMetadata]:
        """列出会话。"""
        opts = options or SqliteSessionListOptions()

        async def _list() -> list[SqliteSessionMetadata]:
            path = await self._get_database_path()
            if not await self._options.env.exists(path):
                return []
            db = await self._get_database()
            rows = await read_session_rows(db, {"cwd": opts.cwd} if opts.cwd else {})
            return [_metadata_from_row(r, path) for r in rows]

        return await self._operations.enqueue(_list)  # type: ignore[no-any-return]

    async def delete(self, metadata: SqliteSessionMetadata) -> None:
        """删除会话。"""

        async def _delete() -> None:
            await self._release_storages_for_session(metadata.id)
            db = await self._get_database()

            async def _delete_tx() -> None:
                if not await session_exists(db, metadata.id):
                    await delete_session_lease(db, metadata.id)
                    return
                await _acquire_writer_lease(db, metadata.id, self._lease_options)
                await delete_branch_cache(db, metadata.id)
                await delete_fact_rows(db, metadata.id)
                await delete_lane_rows(db, metadata.id)
                await delete_record_rows(db, metadata.id)
                await delete_entry_rows(db, metadata.id)
                await delete_session_lease(db, metadata.id)
                await delete_stats(db, metadata.id)
                await delete_sequence(db, metadata.id)
                await delete_session_row(db, metadata.id)

            await db.transaction(_delete_tx)

        return await self._operations.enqueue(_delete)  # type: ignore[no-any-return]

    async def fork(
        self,
        source: SqliteSessionMetadata,
        options: Any,  # SqliteSessionCreateOptions & ForkOptions
    ) -> Session:
        """fork 会话。"""
        from pi_agent.uuid7 import uuidv7

        async def _fork() -> Session:
            db = await self._get_database()
            path = await self._get_database_path()
            source_metadata = _metadata_from_row(
                await _require_session_row(db, source.id), path
            )
            session_id = options.id or uuidv7()
            if await session_exists(db, session_id):
                raise SessionError(
                    "already_exists", f"Session already exists: {session_id}"
                )

            entries: list[dict[str, object]] = []
            lanes_list: list[dict[str, object]] = []
            branch_tips: list[str] = []
            branch_fork_target_id: str | None = None

            if options.scope == "tree":
                entries = await read_entry_rows(db, source.id, {"order": "oldestFirst"})
                lanes_list = [
                    {"lane": r["lane"], "leaf_id": r.get("leaf_id")}
                    for r in await read_lanes(db, source.id)
                ]
                branch_tips = await read_branch_tip_ids(db, source.id)
            else:
                main_lane = await read_lane(db, source.id, "main")
                if not main_lane:
                    raise SessionError("invalid_lane", "Lane not found: main")
                selected_entry_id = (
                    options.entry_id if options.entry_id else main_lane.get("leaf_id")
                )
                if selected_entry_id is not None:
                    target = await read_entry_row(
                        db, source.id, cast(str, selected_entry_id)
                    )
                    if not target or str(cast(str, target.get("type"))) != "message":
                        raise SessionError(
                            "invalid_fork_target",
                            f"Fork target is not a message entry: {selected_entry_id}",
                        )
                    position = (
                        options.position if options.entry_id is not None else "at"
                    )
                    if position is None:
                        position = "at"
                    branch_fork_target_id = (
                        str(cast(str, target["id"]))
                        if position == "at"
                        else str(cast(str, target.get("parent_id")))
                        if target.get("parent_id")
                        else None
                    )
                lanes_list.append({"lane": "main", "leaf_id": branch_fork_target_id})
                if branch_fork_target_id is not None:
                    cached = await read_cached_branch(
                        db, source.id, branch_fork_target_id
                    )
                    if not cached:
                        raise SessionError(
                            "invalid_fork_target",
                            f"Fork target is not on a cached branch: {branch_fork_target_id}",
                        )
                    cached_rows = await query_cached_branch_rows(
                        db, source.id, cached, {"order": "oldestFirst"}
                    )
                    entries = [_entry_row_from_cached(r) for r in cached_rows]
                    branch_tips.append(branch_fork_target_id)

            copied_ids = {str(cast(str, e["id"])) for e in entries}
            latest_name = await read_latest_fact(db, source.id, "name", None)
            latest_labels = await read_latest_label_facts(db, source.id)
            labels_to_copy = [
                r
                for r in latest_labels
                if options.scope == "tree"
                or (r.get("key") is not None and cast(str, r["key"]) in copied_ids)
            ]
            created_at = _now_ms()
            fork_metadata = (
                options.metadata
                if options.metadata is not None
                else source_metadata.metadata
            )

            try:

                async def _fork_tx() -> SessionLease:
                    await insert_session_row(
                        db,
                        NewSessionRow(
                            id=session_id,
                            created_at=_timestamp_to_text(created_at),
                            cwd=options.cwd,
                            parent_session_id=options.parent_session_id or source.id,
                            metadata=fork_metadata,
                        ),
                    )
                    await create_sequence(db, session_id)
                    message_count = sum(
                        1 for e in entries if str(cast(str, e.get("type"))) == "message"
                    )
                    await create_stats(db, session_id, message_count)

                    next_seq = 1
                    for entry in entries:
                        await insert_entry_row(
                            db,
                            session_id,
                            NewEntryRow(
                                seq=next_seq,
                                id=str(cast(str, entry["id"])),
                                parent_id=str(cast(str, entry.get("parent_id")))
                                if entry.get("parent_id")
                                else None,
                                type=str(cast(str, entry["type"])),
                                timestamp=str(cast(str, entry["timestamp"])),
                                payload=str(cast(str, entry.get("payload", "{}"))),
                            ),
                        )
                        next_seq += 1

                    if options.scope == "tree":
                        for lane in lanes_list:
                            lane_leaf = lane.get("leaf_id")
                            await create_lane(
                                db,
                                session_id,
                                next_seq,
                                str(cast(str, lane["lane"])),
                                str(cast(str, lane_leaf))
                                if lane_leaf is not None
                                else None,
                            )
                            next_seq += 1
                    else:
                        await create_initial_lane(
                            db, session_id, "main", branch_fork_target_id
                        )

                    if latest_name and latest_name.get("value") is not None:
                        await append_fact(
                            db,
                            session_id,
                            next_seq,
                            "name",
                            None,
                            str(cast(str, latest_name["value"])),
                        )
                        next_seq += 1
                    for label in labels_to_copy:
                        await append_fact(
                            db,
                            session_id,
                            next_seq,
                            "label",
                            str(cast(str, label.get("key"))),
                            str(cast(str, label.get("value"))),
                        )
                        next_seq += 1

                    await set_next_sequence(db, session_id, next_seq)
                    for tip in branch_tips:
                        await build_cached_branch(db, session_id, tip)
                    return await _acquire_writer_lease(
                        db, session_id, self._lease_options
                    )

                lease = cast(SessionLease, await db.transaction(_fork_tx))
            except SessionError:
                raise
            except Exception as error:
                raise SessionError(
                    "storage",
                    f"Failed to fork SQLite session {session_id}",
                    error,
                ) from error

            row = await _require_session_row(db, session_id)
            return self._session_from_lease(db, _metadata_from_row(row, path), lease)

        return await self._operations.enqueue(_fork)  # type: ignore[no-any-return]

    async def close(self) -> None:
        """关闭仓库。"""
        await self._operations.drain()
        for storage in list(self._active_storages):
            await storage.release()
        if self._database:
            await self._database.close()
            self._database = None
        self._database_promise = None

    async def _get_database_path(self) -> str:
        """获取数据库路径。"""
        if self._database_path is None:
            result = await self._options.env.absolute_path(self._options.database_path)
            self._database_path = result
        return self._database_path

    async def _get_database(self) -> SqliteDatabase:
        """获取数据库。"""
        if self._database_promise is None:
            self._database_promise = asyncio.ensure_future(self._open_database())
        self._database = await self._database_promise
        return self._database

    async def _open_database(self) -> SqliteDatabase:
        """打开数据库。"""
        path = await self._get_database_path()
        await self._options.env.create_dir(_get_parent_path(path), recursive=True)
        db = await self._options.sqlite.open(path)
        try:
            await _configure_sqlite_database(db)
            await apply_migrations(db)
            return db
        except BaseException:
            await db.close()
            raise


__all__ = [
    "SqliteSessionRepository",
    "SqliteSessionRepositoryOptions",
    "SqliteSessionStorage",
    "SqliteWriterLeaseOptions",
]
