"""内存会话存储。

``InMemorySessionStorage`` 实现 ``SessionStorage`` 契约；``InMemorySessionRepo``
实现 ``SessionRepo``（create / open / list / delete / fork）。
"""

from __future__ import annotations

import time
from collections.abc import Iterator

from pi_agent.uuid7 import uuidv7
from .session import Session
from .types import (
    BranchEntryQuery,
    CustomEntry,
    Entry,
    EntryQuery,
    ForkOptions,
    LanePointer,
    LaneRecord,
    LogEntryItem,
    LogItem,
    LogLabelFactItem,
    LogLaneItem,
    LogNameFactItem,
    LogRecordItem,
    NewRecord,
    OperationFinishedRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionCreateOptions,
    SessionError,
    SessionMetadata,
    SessionRepo,
    SessionStats,
    SessionStorage,
    UsageRecord,
)


class InMemorySessionStorage(SessionStorage):
    """基于 dict/list 的内存实现（单进程、非持久化）。"""

    def __init__(self, metadata: SessionMetadata) -> None:
        self._metadata = metadata
        self._entries: dict[str, Entry] = {}
        self._records: list[LaneRecord] = []
        self._open_operations_by_lane: dict[str, dict[str, OperationStartedRecord]] = {}
        self._lanes: dict[str, str | None] = {"main": None}
        self._name: str | None = None
        self._labels: dict[str, str] = {}
        self._seq = 0
        self._stats = SessionStats()

    # -- 元数据 ------------------------------------------------------------

    async def get_metadata(self) -> SessionMetadata:
        return self._metadata

    async def get_stats(self) -> SessionStats:
        """返回会话统计（增量维护）。"""
        return self._stats

    # -- lanes -------------------------------------------------------------

    async def get_lanes(self) -> list[LanePointer]:
        return [
            LanePointer(lane=lane, leaf_id=leaf) for lane, leaf in self._lanes.items()
        ]

    async def create_lane(self, lane: str, at: str | None) -> None:
        if lane in self._lanes:
            raise SessionError("already_exists", f'lane "{lane}" already exists')
        self._lanes[lane] = at

    async def move_lane(self, lane: str, to: str | None) -> None:
        if lane not in self._lanes:
            raise SessionError("not_found", f'lane "{lane}" not found')
        self._lanes[lane] = to

    # -- entries / records -------------------------------------------------

    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry:
        if lane not in self._lanes:
            raise SessionError("invalid_lane", f'lane "{lane}" not found')
        if entry.id in self._entries:
            raise SessionError("already_exists", f'entry "{entry.id}" already exists')
        self._seq += 1
        stored = entry.model_copy(
            update={
                "seq": self._seq,
                "timestamp": _now_ms(),
                "parent_id": self._lanes[lane],
            }
        )
        self._entries[stored.id] = stored
        self._lanes[lane] = stored.id
        if stored.type == "message":
            self._stats.message_count += 1
        return stored

    async def append_record(self, record: NewRecord) -> LaneRecord:
        if record.id in {r.id for r in self._records}:
            raise SessionError("already_exists", f'record "{record.id}" already exists')
        self._seq += 1
        stored = record.model_copy(update={"seq": self._seq, "timestamp": _now_ms()})
        self._records.append(stored)
        # 增量维护恢复投影，避免恢复时全量扫描。
        if isinstance(stored, OperationStartedRecord):
            lane_ops = self._open_operations_by_lane.setdefault(stored.lane, {})
            lane_ops[stored.id] = stored
        elif isinstance(stored, OperationFinishedRecord):
            finished_lane_ops = self._open_operations_by_lane.get(stored.lane)
            if finished_lane_ops is not None:
                finished_lane_ops.pop(stored.run_id, None)
        elif isinstance(stored, UsageRecord):
            self._stats.cached_tokens += stored.usage.cache_read
            self._stats.uncached_tokens += stored.usage.input + stored.usage.cache_write
            self._stats.total_tokens += stored.usage.total_tokens
            self._stats.cost_total += stored.usage.cost.total
        return stored

    # -- reads -------------------------------------------------------------

    async def get_entry(self, id: str) -> Entry | None:
        return self._entries.get(id)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        """基础查询：过滤 type/custom_type，按 seq 排序，可选 limit。"""
        query = query or EntryQuery()
        entries = list(self._entries.values())
        if query.type is not None:
            entries = [e for e in entries if e.type == query.type]
        if query.custom_type is not None:
            entries = [
                e
                for e in entries
                if e.type == "custom" and e.custom_type == query.custom_type
            ]
        entries.sort(key=lambda e: e.seq, reverse=query.order == "newestFirst")
        if query.limit is not None:
            entries = entries[: query.limit]
        return entries

    async def find_entries_on_branch(self, query: BranchEntryQuery) -> list[Entry]:
        """沿 parent 链从 ``query.start`` 回溯收集条目。"""
        if query.start is None:
            raise SessionError(
                "invalid_query", "find_entries_on_branch requires a start"
            )
        if query.order == "oldestFirst":
            entries = list(self._walk_to_root(query.start))
            entries.reverse()
            results: list[Entry] = []
            for entry in entries:
                reached_bound = (
                    entry.id == query.stop_at_id or entry.type == query.stop_at_type
                )
                if self._matches_entry_query(entry, query):
                    results.append(entry)
                if reached_bound or (
                    query.limit is not None and len(results) == query.limit
                ):
                    break
            return results
        results = []
        for entry in self._walk_to_root(query.start, query):
            if self._matches_entry_query(entry, query):
                results.append(entry)
            if query.limit is not None and len(results) == query.limit:
                break
        return results

    def _walk_to_root(
        self, start: str, bounds: BranchEntryQuery | None = None
    ) -> Iterator[Entry]:
        """从 ``start`` 沿 parent 链向上产出条目；检测环；在边界处停止。"""
        visited: set[str] = set()
        current = self._entries.get(start)
        if current is None:
            raise SessionError("not_found", f"Entry not found: {start}")
        while current is not None:
            if current.id in visited:
                raise SessionError(
                    "invalid_entry", f"Session branch contains a cycle at {current.id}"
                )
            visited.add(current.id)
            yield current
            if bounds is not None and (
                current.id == bounds.stop_at_id or current.type == bounds.stop_at_type
            ):
                break
            if current.parent_id is None:
                break
            current = self._entries.get(current.parent_id)

    def _matches_entry_query(self, entry: Entry, query: BranchEntryQuery) -> bool:
        """条目是否匹配查询过滤（type / custom_type / cursor）。"""
        if query.type is not None and entry.type != query.type:
            return False
        if query.custom_type is not None and not (
            isinstance(entry, CustomEntry) and entry.custom_type == query.custom_type
        ):
            return False
        if query.cursor is not None:
            if query.order == "oldestFirst":
                if entry.seq <= query.cursor.after_seq:
                    return False
            elif entry.seq >= query.cursor.after_seq:
                return False
        return True

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        """基础查询：过滤 lane/type/run_id，按 seq 排序，可选 limit。"""
        query = query or RecordQuery()
        records = list(self._records)
        if query.lane is not None:
            records = [r for r in records if r.lane == query.lane]
        if query.type is not None:
            records = [r for r in records if r.type == query.type]
        if query.run_id is not None:
            records = [r for r in records if getattr(r, "run_id", None) == query.run_id]
        records.sort(key=lambda r: r.seq, reverse=query.order == "newestFirst")
        if query.limit is not None:
            records = records[: query.limit]
        return records

    async def find_open_operations(
        self, lane: str, options: dict[str, object] | None = None
    ) -> list[OperationStartedRecord]:
        """查找未结束的操作（读取增量维护的投影，按写入逆序）。"""
        open_operations = list(self._open_operations_by_lane.get(lane, {}).values())
        open_operations.reverse()
        limit = options.get("limit") if options else None
        if isinstance(limit, int):
            open_operations = open_operations[:limit]
        return open_operations

    async def get_log(self, options: dict[str, object] | None = None) -> list[LogItem]:
        """全量日志（entry / record / lane / fact 按 seq 合并排序）。"""
        items: list[LogItem] = []
        for entry in self._entries.values():
            items.append(LogEntryItem(seq=entry.seq, entry=entry))
        for record in self._records:
            items.append(LogRecordItem(seq=record.seq, record=record))
        for lane, leaf_id in self._lanes.items():
            items.append(LogLaneItem(seq=0, lane=lane, leaf_id=leaf_id))
        if self._name is not None:
            items.append(LogNameFactItem(seq=0, fact="name", name=self._name))
        for target_id, label in self._labels.items():
            items.append(
                LogLabelFactItem(seq=0, fact="label", target_id=target_id, label=label)
            )
        items.sort(key=lambda item: item.seq)
        if options is not None:
            after_seq = options.get("after_seq")
            if isinstance(after_seq, int):
                items = [i for i in items if i.seq > after_seq]
            limit = options.get("limit")
            if isinstance(limit, int):
                items = items[:limit]
        return items

    # -- facts -------------------------------------------------------------

    async def get_name(self) -> str | None:
        return self._name

    async def set_name(self, name: str) -> None:
        self._name = name

    async def get_label(self, id: str) -> str | None:
        return self._labels.get(id)

    async def set_label(self, id: str, label: str | None) -> None:
        if label is None:
            self._labels.pop(id, None)
        else:
            self._labels[id] = label


class InMemorySessionRepo(SessionRepo):
    """基于内存的会话仓库。"""

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}

    async def create(self, options: SessionCreateOptions | None = None) -> Session:
        options = options or SessionCreateOptions()
        session_id = options.id or uuidv7()
        if session_id in self._sessions:
            raise SessionError(
                "already_exists", f'session "{session_id}" already exists'
            )
        metadata = SessionMetadata(
            id=session_id,
            created_at=_now_ms(),
            parent_session_id=options.parent_session_id,
        )
        session = Session(InMemorySessionStorage(metadata))
        self._sessions[session_id] = session
        return session

    async def open(self, metadata: SessionMetadata) -> Session:
        session = self._sessions.get(metadata.id)
        if session is None:
            raise SessionError("not_found", f'session "{metadata.id}" not found')
        return session

    async def list(self, options: object | None = None) -> list[SessionMetadata]:
        sessions = list(self._sessions.values())
        return [await session.storage.get_metadata() for session in sessions]

    async def delete(self, metadata: SessionMetadata) -> None:
        if metadata.id not in self._sessions:
            raise SessionError("not_found", f'session "{metadata.id}" not found')
        del self._sessions[metadata.id]

    async def fork(self, source: SessionMetadata, options: ForkOptions) -> Session:
        raise NotImplementedError("InMemorySessionRepo.fork 尚未实现（TODO）")


def _now_ms() -> int:
    """当前 Unix 毫秒时间戳。"""
    return int(time.time() * 1000)
