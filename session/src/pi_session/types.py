"""会话数据模型。

- ``Entry``：7 种会话树条目（message / model_change / thinking_level_change /
  active_tools_change / compaction / branch_summary / custom）
- ``LaneRecord``：9 种 lane 操作日志记录
- ``SessionStorage`` / ``SessionTree`` / ``SessionRepo``：存储契约

存储契约约定：**方法永不抛异常**（除 ``SessionError`` 代表的逻辑错误），
所有后端失败编码为 ``SessionError(code="storage")``。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from pi_agent.types import AgentMessage, Usage

if TYPE_CHECKING:
    from .session import Session

SessionStopReason: TypeAlias = Literal[
    "stop",
    "max_tokens",
    "length",
    "tool_use",
    "error",
    "aborted",
    "deferred",
]
"""会话可持久化的停止原因（排除 ``pending``，追加 ``deferred``）。"""


class IdGenerator(Protocol):
    """ID 生成器。"""

    def next(self) -> str: ...


# ---------------------------------------------------------------------------
# Entry（会话树条目）
# ---------------------------------------------------------------------------

class EntryBase(BaseModel):
    """条目公共字段。

    存储层赋值前（``ProvisionedEntry`` 状态）``parent_id``/``seq``/``timestamp``
    使用默认占位；存储层负责分配真实值并返回完整条目。
    """

    id: str
    type: str
    parent_id: str | None = None
    seq: int = 0
    timestamp: int = 0


class MessageEntry(EntryBase):
    """消息条目。"""

    type: Literal["message"] = "message"
    message: AgentMessage
    terminate: bool | None = None


class ModelChangeEntry(EntryBase):
    """模型切换条目。"""

    type: Literal["model_change"] = "model_change"
    provider: str
    model_id: str


class ThinkingLevelEntry(EntryBase):
    """思考级别切换条目。"""

    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str


class ActiveToolsEntry(EntryBase):
    """活跃工具集合切换条目。"""

    type: Literal["active_tools_change"] = "active_tools_change"
    active_tool_names: list[str]


class CompactionEntry(EntryBase):
    """上下文压缩条目。"""

    type: Literal["compaction"] = "compaction"
    summary: str
    retained_tail: list[AgentMessage] = Field(default_factory=list)
    tokens_before: int
    details: object | None = None
    usage: Usage | None = None


class BranchSummaryEntry(EntryBase):
    """分支摘要条目。"""

    type: Literal["branch_summary"] = "branch_summary"
    from_id: str
    summary: str
    details: object | None = None
    usage: Usage | None = None


class CustomEntry(EntryBase):
    """应用自定义条目。"""

    type: Literal["custom"] = "custom"
    custom_type: str
    data: object | None = None


Entry: TypeAlias = Annotated[
    MessageEntry
    | ModelChangeEntry
    | ThinkingLevelEntry
    | ActiveToolsEntry
    | CompactionEntry
    | BranchSummaryEntry
    | CustomEntry,
    Field(discriminator="type"),
]
"""会话树条目判别联合。"""

ProvisionedEntry: TypeAlias = Entry
"""待存储条目：与 ``Entry`` 同构，存储前 ``parent_id``/``seq``/``timestamp`` 为占位。"""

EntryType: TypeAlias = Literal[
    "message",
    "model_change",
    "thinking_level_change",
    "active_tools_change",
    "compaction",
    "branch_summary",
    "custom",
]
"""``Entry`` 判别字段的取值。"""


# ---------------------------------------------------------------------------
# LaneRecord（lane 操作日志）
# ---------------------------------------------------------------------------

class RecordBase(BaseModel):
    """记录公共字段（同 Entry，存储前为占位值）。"""

    id: str
    type: str
    lane: str
    seq: int = 0
    timestamp: int = 0


class RunIntent(BaseModel):
    """run 操作意图。"""

    kind: Literal["run"] = "run"
    original_prompt: list[AgentMessage] = Field(default_factory=list)
    initial_messages: list[ProvisionedEntry] = Field(default_factory=list)
    system_prompt_override: str | None = None
    resume_data: dict[str, JsonValue] | None = None


class CompactionIntent(BaseModel):
    """compaction 操作意图。"""

    kind: Literal["compaction"] = "compaction"
    custom_instructions: str | None = None
    result_entry_id: str


class NavigationIntent(BaseModel):
    """navigation 操作意图。"""

    kind: Literal["navigation"] = "navigation"
    target_id: str | None
    summarize: bool = False
    custom_instructions: str | None = None
    label: str | None = None
    summary_entry_id: str | None = None


OperationIntent: TypeAlias = Annotated[
    RunIntent | CompactionIntent | NavigationIntent, Field(discriminator="kind")
]
"""操作意图判别联合。"""


class OperationStartedRecord(RecordBase):
    """操作开始记录：接受即持久化，崩溃后可恢复。"""

    type: Literal["operation_started"] = "operation_started"
    source_leaf_id: str | None = None
    intent: OperationIntent


class AbortRequestedRecord(RecordBase):
    """中止请求记录。"""

    type: Literal["abort_requested"] = "abort_requested"
    run_id: str


class OperationFinishedRecord(RecordBase):
    """操作结束记录。"""

    type: Literal["operation_finished"] = "operation_finished"
    run_id: str
    outcome: Literal["completed", "aborted", "failed", "declined"]
    error: dict[str, str] | None = None


CompactionReason: TypeAlias = Literal["manual", "threshold", "overflow"]
"""压缩触发原因。"""


class StepAttemptRecord(RecordBase):
    """步骤尝试记录（step 为 assistant/branch_summary 时无 compaction_reason）。"""

    type: Literal["step_attempt"] = "step_attempt"
    run_id: str
    step: Literal["assistant", "branch_summary", "compaction"]
    attempt: int
    result_entry_id: str
    compaction_reason: CompactionReason | None = None


class ToolStartedRecord(RecordBase):
    """工具开始记录。"""

    type: Literal["tool_started"] = "tool_started"
    run_id: str
    assistant_entry_id: str
    tool_index: int
    tool_call_id: str
    tool_name: str
    effective_args: dict[str, object] = Field(default_factory=dict)
    result_entry_id: str
    replay: Literal["never", "safe"] = "never"


class QueueEnqueuedRecord(RecordBase):
    """队列入队记录（steer/followUp 携带 run_id；nextRun 不携带）。"""

    type: Literal["queue_enqueued"] = "queue_enqueued"
    queue: Literal["steer", "followUp", "nextRun"]
    run_id: str | None = None
    target: ProvisionedEntry


class QueueCancelledRecord(RecordBase):
    """队列取消记录（不消费目标条目）。"""

    type: Literal["queue_cancelled"] = "queue_cancelled"
    run_id: str | None = None
    entry_id: str


class WriteDeferredRecord(RecordBase):
    """延迟写入记录（步骤在途时请求的事实，取消后仍生效）。"""

    type: Literal["write_deferred"] = "write_deferred"
    run_id: str
    target: ProvisionedEntry


class UsageRecord(RecordBase):
    """用量记录。"""

    type: Literal["usage"] = "usage"
    usage: Usage
    cause: Literal["assistant", "compaction", "branch_summary", "deferred_fetch", "tool", "hook", "adjustment"]
    run_id: str | None = None
    entry_id: str | None = None
    attempt: int | None = None
    stop_reason: SessionStopReason | None = None
    tool_call_id: str | None = None
    details: JsonValue | None = None


LaneRecord: TypeAlias = Annotated[
    OperationStartedRecord
    | AbortRequestedRecord
    | OperationFinishedRecord
    | StepAttemptRecord
    | ToolStartedRecord
    | QueueEnqueuedRecord
    | QueueCancelledRecord
    | WriteDeferredRecord
    | UsageRecord,
    Field(discriminator="type"),
]
"""lane 操作日志记录判别联合。"""

NewRecord: TypeAlias = LaneRecord
"""待存储记录：与 ``LaneRecord`` 同构，存储前 ``seq``/``timestamp`` 为占位。"""

LaneRecordType: TypeAlias = Literal[
    "operation_started",
    "abort_requested",
    "operation_finished",
    "step_attempt",
    "tool_started",
    "queue_enqueued",
    "queue_cancelled",
    "write_deferred",
    "usage",
]
"""``LaneRecord`` 判别字段的取值。"""


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------

EntryOrder: TypeAlias = Literal["newestFirst", "oldestFirst"]
"""条目顺序。"""


class EntryCursor(BaseModel):
    """基于 seq 的光标。"""

    after_seq: int


class EntryQuery(BaseModel):
    """条目查询。"""

    type: EntryType | None = None
    custom_type: str | None = None
    order: EntryOrder = "newestFirst"
    limit: int | None = None
    cursor: EntryCursor | None = None


class BranchBounds(BaseModel):
    """分支范围。"""

    start: str | None = None
    stop_at_type: EntryType | None = None
    stop_at_id: str | None = None


class BranchEntryQuery(EntryQuery, BranchBounds):
    """分支条目查询（``EntryQuery`` + ``BranchBounds``）。"""


class RecordQuery(BaseModel):
    """记录查询。"""

    lane: str | None = None
    type: LaneRecordType | None = None
    run_id: str | None = None
    operation_kind: Literal["run", "compaction", "navigation"] | None = None
    after_seq: int | None = None
    order: EntryOrder = "newestFirst"
    limit: int | None = None


# ---------------------------------------------------------------------------
# 元数据 / 统计 / 日志
# ---------------------------------------------------------------------------

class SessionMetadata(BaseModel):
    """会话元数据。"""

    id: str
    created_at: int
    parent_session_id: str | None = None


class SessionStats(BaseModel):
    """会话统计。"""

    message_count: int = 0
    cached_tokens: int = 0
    uncached_tokens: int = 0
    total_tokens: int = 0
    cost_total: float = 0.0


class LanePointer(BaseModel):
    """lane 位置指针。"""

    lane: str
    leaf_id: str | None


class LogEntryItem(BaseModel):
    """日志项：条目。"""

    kind: Literal["entry"] = "entry"
    seq: int
    entry: Entry


class LogRecordItem(BaseModel):
    """日志项：记录。"""

    kind: Literal["record"] = "record"
    seq: int
    record: LaneRecord


class LogLaneItem(BaseModel):
    """日志项：lane 移动。"""

    kind: Literal["lane"] = "lane"
    seq: int
    lane: str
    leaf_id: str | None


class LogNameFactItem(BaseModel):
    """日志项：name 事实。"""

    kind: Literal["fact"] = "fact"
    fact: Literal["name"] = "name"
    seq: int
    name: str


class LogLabelFactItem(BaseModel):
    """日志项：label 事实。"""

    kind: Literal["fact"] = "fact"
    fact: Literal["label"] = "label"
    seq: int
    target_id: str
    label: str | None


LogItem: TypeAlias = (
    LogEntryItem | LogRecordItem | LogLaneItem | LogNameFactItem | LogLabelFactItem
)
"""日志项（kind: entry / record / lane / fact）。"""


# ---------------------------------------------------------------------------
# 契约：SessionStorage / SessionTree / SessionRepo
# ---------------------------------------------------------------------------

class SessionStorage(Protocol):
    """会话存储后端契约。

    实现必须满足：
    - ``append_entry`` / ``append_record`` 分配全局递增 ``seq`` 与时间戳；
    - 重复 ID 抛出 ``SessionError("already_exists")`` 且不改变状态；
    - 所有查询支持 ``EntryQuery`` / ``RecordQuery`` 过滤、排序与 limit。
    """

    async def get_metadata(self) -> SessionMetadata: ...

    # Lanes
    async def get_lanes(self) -> list[LanePointer]: ...
    async def create_lane(self, lane: str, at: str | None) -> None: ...
    async def move_lane(self, lane: str, to: str | None) -> None: ...

    # Entries and Records
    async def append_entry(self, entry: ProvisionedEntry, lane: str) -> Entry: ...
    async def append_record(self, record: NewRecord) -> LaneRecord: ...

    # Reads
    async def get_entry(self, id: str) -> Entry | None: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entries_on_branch(self, query: BranchEntryQuery) -> list[Entry]: ...
    async def find_records(self, query: RecordQuery | None = None) -> list[LaneRecord]: ...
    async def find_open_operations(self, lane: str, options: dict[str, object] | None = None) -> list[OperationStartedRecord]: ...
    async def get_log(self, options: dict[str, object] | None = None) -> list[LogItem]: ...

    # Global facts
    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str) -> None: ...
    async def get_label(self, id: str) -> str | None: ...
    async def set_label(self, id: str, label: str | None) -> None: ...
    async def get_stats(self) -> SessionStats: ...


class SessionTree(Protocol):
    """会话树视图契约（面向调用方的便捷接口）。"""

    async def get_leaf_id(self) -> str | None: ...
    async def get_entry(self, id: str) -> Entry | None: ...
    async def get_stats(self) -> SessionStats: ...
    async def get_name(self) -> str | None: ...
    async def set_name(self, name: str) -> None: ...
    async def get_label(self, target_id: str) -> str | None: ...
    async def set_label(self, target_id: str, label: str | None) -> None: ...
    async def find_entries(self, query: EntryQuery | None = None) -> list[Entry]: ...
    async def find_entry(self, query: EntryQuery | None = None) -> Entry | None: ...
    async def find_entries_on_branch(self, query: BranchEntryQuery | None = None) -> list[Entry]: ...
    async def find_entry_on_branch(self, query: BranchEntryQuery | None = None) -> Entry | None: ...
    async def append_message(self, message: AgentMessage) -> str: ...
    async def append_custom_entry(self, custom_type: str, data: object | None = None) -> str: ...


class SessionCreateOptions(BaseModel):
    """创建会话选项。"""

    id: str | None = None
    parent_session_id: str | None = None


class ForkOptions(BaseModel):
    """fork 选项（branch 或 tree 作用域）。"""

    scope: Literal["branch", "tree"] = "tree"
    entry_id: str | None = None
    position: Literal["before", "at"] | None = None


class SessionRepo(Protocol):
    """会话仓库契约（create / open / list / delete / fork）。"""

    async def create(self, options: SessionCreateOptions | None = None) -> Session: ...
    async def open(self, metadata: SessionMetadata) -> Session: ...
    async def list(self, options: object | None = None) -> list[SessionMetadata]: ...
    async def delete(self, metadata: SessionMetadata) -> None: ...
    async def fork(self, source: SessionMetadata, options: ForkOptions) -> Session: ...


# ---------------------------------------------------------------------------
# SessionError
# ---------------------------------------------------------------------------

SessionErrorCode: TypeAlias = Literal[
    "not_found",
    "already_exists",
    "invalid_entry",
    "invalid_payload",
    "invalid_lane",
    "invalid_query",
    "invalid_fork_target",
    "storage",
]


class SessionError(Exception):
    """会话错误。"""

    def __init__(self, code: SessionErrorCode, message: str, cause: BaseException | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        if cause is not None:
            self.__cause__ = cause
