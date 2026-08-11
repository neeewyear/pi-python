"""Session 类（对应 ``harness/session/session.ts``）。

``Session`` 是会话树视图：封装 ``SessionStorage`` 并绑定到 ``main`` lane，
提供树查询、追加、lane 管理、记录查询与全局事实的便捷接口。
"""

from __future__ import annotations

import orjson

from pi_agent.types import AgentMessage
from .types import (
    BranchEntryQuery,
    CustomEntry,
    Entry,
    EntryQuery,
    LaneRecord,
    LogItem,
    MessageEntry,
    NewRecord,
    OperationStartedRecord,
    ProvisionedEntry,
    RecordQuery,
    SessionError,
    SessionStats,
    SessionStorage,
)

DEFAULT_LANE = "main"


def _assert_json_serializable(value: object) -> None:
    """严格 JSON 校验：不可序列化的负载抛出 ``invalid_payload``。"""
    try:
        orjson.dumps(value)
    except orjson.JSONEncodeError as exc:
        raise SessionError("invalid_payload", f"value is not JSON-serializable: {exc}") from exc


class Session:
    """绑定到 ``main`` lane 的会话树视图。"""

    def __init__(self, storage: SessionStorage) -> None:
        self._storage = storage
        self._lane = DEFAULT_LANE

    @property
    def storage(self) -> SessionStorage:
        """底层存储后端。"""
        return self._storage

    # -- 树查询 ------------------------------------------------------------

    async def get_leaf_id(self) -> str | None:
        """返回当前 lane 的叶子条目 ID。"""
        lanes = await self._storage.get_lanes()
        for pointer in lanes:
            if pointer.lane == self._lane:
                return pointer.leaf_id
        return None

    async def get_entry(self, id: str) -> Entry | None:
        """按 ID 获取条目。"""
        return await self._storage.get_entry(id)

    async def get_stats(self) -> SessionStats:
        """返回会话统计。"""
        return await self._storage.get_stats()

    async def get_name(self) -> str | None:
        """返回会话名称。"""
        return await self._storage.get_name()

    async def set_name(self, name: str) -> None:
        """设置会话名称。"""
        await self._storage.set_name(name)

    async def get_label(self, target_id: str) -> str | None:
        """获取条目标签。"""
        return await self._storage.get_label(target_id)

    async def set_label(self, target_id: str, label: str | None) -> None:
        """设置条目标签（None 删除）。"""
        await self._storage.set_label(target_id, label)

    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]:
        """查询条目（按 seq 倒序，支持过滤/limit/cursor）。"""
        return await self._storage.find_entries(query)

    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None:
        """查询单条条目。"""
        entries = await self._storage.find_entries(query)
        return entries[0] if entries else None

    async def find_entries_on_branch(self, query: BranchEntryQuery | None = None) -> list[Entry]:
        """查询分支上的条目（对应 TS ``EntryQuery & BranchBounds``，默认从当前叶子回溯）。"""
        start = query.start if query and query.start else await self.get_leaf_id()
        if start is None:
            return []
        merged: dict[str, object] = query.model_dump(exclude_none=True) if query else {}
        merged["start"] = start
        return await self._storage.find_entries_on_branch(BranchEntryQuery.model_validate(merged))

    async def find_entry_on_branch(self, query: BranchEntryQuery | None = None) -> Entry | None:
        """查询分支上的单条条目。"""
        entries = await self.find_entries_on_branch(query)
        return entries[0] if entries else None

    # -- 追加 --------------------------------------------------------------

    async def append_message(self, message: AgentMessage) -> str:
        """追加消息条目，返回条目 ID。"""
        entry = MessageEntry(id=_next_id(message), message=message)
        stored = await self._storage.append_entry(entry, self._lane)
        return stored.id

    async def append_custom_entry(self, custom_type: str, data: object | None = None) -> str:
        """追加自定义条目，返回条目 ID。"""
        _assert_json_serializable(data)
        entry = CustomEntry(id=_next_id(data), custom_type=custom_type, data=data)
        stored = await self._storage.append_entry(entry, self._lane)
        return stored.id

    # -- lane 管理 ---------------------------------------------------------

    def view(self, lane: str) -> Session:
        """返回绑定到指定 lane 的视图。"""
        session = Session(self._storage)
        session._lane = lane
        return session

    async def create_lane(self, lane: str, at: str | None) -> None:
        """创建 lane 并锚定到指定条目。"""
        await self._storage.create_lane(lane, at)

    async def move_lane(self, lane: str, to: str | None) -> None:
        """移动 lane 到指定条目（None 表示根）。"""
        await self._storage.move_lane(lane, to)

    # -- 记录 --------------------------------------------------------------

    async def append_entry(self, entry: ProvisionedEntry, lane: str | None = None) -> Entry:
        """追加条目到指定 lane（默认当前 lane）。"""
        return await self._storage.append_entry(entry, lane or self._lane)

    async def append_record(self, record: NewRecord) -> LaneRecord:
        """追加记录到当前 lane。"""
        return await self._storage.append_record(record)

    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]:
        """查询记录。"""
        return await self._storage.find_records(query)

    async def find_open_operations(self, lane: str, options: dict[str, object] | None = None) -> list[OperationStartedRecord]:
        """查找未结束的操作。"""
        return await self._storage.find_open_operations(lane, options)

    async def get_log(self, options: dict[str, object] | None = None) -> list[LogItem]:
        """获取全量日志（按 seq 排序）。"""
        return await self._storage.get_log(options)


def _next_id(payload: object) -> str:
    """生成条目 ID（TODO：接入 uuidv7 后替换）。"""
    from pi_agent.uuid7 import uuidv7

    return uuidv7()
