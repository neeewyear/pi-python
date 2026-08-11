"""Agent Harness 类型与错误体系（对应 ``harness/agent-harness.ts`` 的类型层）。

包含：
- 20 个错误类（17 个 TaggedError + 3 个系统错误）
- 操作结果类型（RunOutcome / CompactionOutcome / NavigationOutcome 等）
- 数据模型（LaneInfo / LaneSnapshot / SessionSnapshot / ActionInfo 等）
- 接口 / 协议（AgentLane / Hooks / Events / WatchHandle 等）
- AgentHarnessOptions
"""

from __future__ import annotations

from typing import (
    Annotated,
    Awaitable,
    Callable,
    ClassVar,
    Generic,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    runtime_checkable,
)

from pydantic import BaseModel, ConfigDict, Field

from ..result import AgentError, Result
from ..types import (
    AgentMessage,
    AgentTool,
    AssistantMessage,
    ImageContent,
    Model,
    QueueMode,
    ThinkingLevel,
    Usage,
)
from .compaction.compaction import CompactionSettings
from pi_session.types import Entry, SessionTree
from .types import (
    AgentHarnessResources,
    AgentHarnessStreamOptions,
    PromptTemplate,
    Skill,
)


# ---------------------------------------------------------------------------
# 错误类（17 个 TaggedError + 3 个系统错误）
# ---------------------------------------------------------------------------


class LaneBusy(AgentError):
    """车道正忙。"""

    code: ClassVar[str] = "LaneBusy"

    def __init__(self, lane: str, operation_id: str, operation_kind: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.operation_id = operation_id
        self.operation_kind = operation_kind


class MissingIdentities(AgentError):
    """缺少工具/模型身份。"""

    code: ClassVar[str] = "MissingIdentities"

    def __init__(self, lane: str, tools: list[str], models: list[str], message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.tools = tools
        self.models = models


class NoActiveRun(AgentError):
    """无活跃运行。"""

    code: ClassVar[str] = "NoActiveRun"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class NoActiveOperation(AgentError):
    """无活跃操作。"""

    code: ClassVar[str] = "NoActiveOperation"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class NothingToResume(AgentError):
    """无可恢复内容。"""

    code: ClassVar[str] = "NothingToResume"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class InvalidMessage(AgentError):
    """无效消息。"""

    code: ClassVar[str] = "InvalidMessage"

    def __init__(self, lane: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.reason = reason


class UnknownSkill(AgentError):
    """未知技能。"""

    code: ClassVar[str] = "UnknownSkill"

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


class UnknownTemplate(AgentError):
    """未知模板。"""

    code: ClassVar[str] = "UnknownTemplate"

    def __init__(self, name: str, message: str) -> None:
        super().__init__(message)
        self.name = name


class UnknownTarget(AgentError):
    """未知目标。"""

    code: ClassVar[str] = "UnknownTarget"

    def __init__(self, target_id: str, message: str) -> None:
        super().__init__(message)
        self.target_id = target_id


class UnknownQueueItem(AgentError):
    """未知队列项。"""

    code: ClassVar[str] = "UnknownQueueItem"

    def __init__(self, lane: str, entry_id: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.entry_id = entry_id


class LaneExists(AgentError):
    """车道已存在。"""

    code: ClassVar[str] = "LaneExists"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class InvalidLane(AgentError):
    """无效车道。"""

    code: ClassVar[str] = "InvalidLane"

    def __init__(self, lane: str, reason: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane
        self.reason = reason


class NothingToCompact(AgentError):
    """无可压缩内容。"""

    code: ClassVar[str] = "NothingToCompact"

    def __init__(self, lane: str, message: str) -> None:
        super().__init__(message)
        self.lane = lane


class Closed(AgentError):
    """已关闭。"""

    code: ClassVar[str] = "Closed"


class HarnessFault(Exception):
    """内部故障（含 cause）。"""

    def __init__(self, message: str, cause: object) -> None:
        super().__init__(message)
        self.cause = cause


class HarnessClosed(Exception):
    """操作期间被关闭。"""

    def __init__(self) -> None:
        super().__init__("AgentHarness was closed while the operation was active")


class HarnessNotImplemented(Exception):
    """操作未实现。"""

    def __init__(self, operation: str) -> None:
        super().__init__(f"AgentHarness.{operation} is not implemented yet")
        self.operation = operation


# ---------------------------------------------------------------------------
# 共享错误模型
# ---------------------------------------------------------------------------


class OperationError(BaseModel):
    """操作错误（对应 TS ``OperationError``）。"""

    code: str
    message: str


# ---------------------------------------------------------------------------
# RetryPolicy（pi-ai 占位）
# ---------------------------------------------------------------------------


class RetryPolicy(BaseModel):
    """重试策略（TODO：与 pi-ai ``RetryPolicy`` 对齐）。"""

    enabled: bool = False
    max_retries: int = 0
    base_delay_ms: int = 1000


# ---------------------------------------------------------------------------
# 操作结果类型（3 种 Outcome 判别联合）
# ---------------------------------------------------------------------------


class RunCompletedOutcome(BaseModel):
    """运行完成。"""

    kind: Literal["completed"] = "completed"
    leaf_id: str
    final_entry_id: str
    final_message: AssistantMessage


class RunAbortedOutcome(BaseModel):
    """运行中止。"""

    kind: Literal["aborted"] = "aborted"
    leaf_id: str
    final_entry_id: str
    final_message: AssistantMessage


class RunFailedOutcome(BaseModel):
    """运行失败。"""

    kind: Literal["failed"] = "failed"
    leaf_id: str
    error: OperationError
    final_entry_id: str | None = None
    final_message: AssistantMessage | None = None


class RunSuspendedOutcome(BaseModel):
    """运行挂起。"""

    kind: Literal["suspended"] = "suspended"
    leaf_id: str
    final_entry_id: str
    deferred: object = Field(default=None)  # DeferredHandle 占位


RunOutcome: TypeAlias = Annotated[
    RunCompletedOutcome | RunAbortedOutcome | RunFailedOutcome | RunSuspendedOutcome,
    Field(discriminator="kind"),
]
"""运行结果（对应 TS ``RunOutcome``）。"""


class CompactionCompletedOutcome(BaseModel):
    """压缩完成。"""

    kind: Literal["completed"] = "completed"
    leaf_id: str
    entry: object  # CompactionEntry 占位，避免循环导入


class CompactionDeclinedOutcome(BaseModel):
    """压缩被拒绝。"""

    kind: Literal["declined"] = "declined"
    leaf_id: str


class CompactionAbortedOutcome(BaseModel):
    """压缩中止。"""

    kind: Literal["aborted"] = "aborted"
    leaf_id: str


class CompactionFailedOutcome(BaseModel):
    """压缩失败。"""

    kind: Literal["failed"] = "failed"
    leaf_id: str
    error: OperationError


CompactionOutcome: TypeAlias = Annotated[
    CompactionCompletedOutcome
    | CompactionDeclinedOutcome
    | CompactionAbortedOutcome
    | CompactionFailedOutcome,
    Field(discriminator="kind"),
]
"""压缩结果（对应 TS ``CompactionOutcome``）。"""


class NavigationCompletedOutcome(BaseModel):
    """导航完成。"""

    kind: Literal["completed"] = "completed"
    new_leaf_id: str | None
    summary_entry: object | None = None  # BranchSummaryEntry 占位


class NavigationDeclinedOutcome(BaseModel):
    """导航被拒绝。"""

    kind: Literal["declined"] = "declined"
    leaf_id: str | None


class NavigationAbortedOutcome(BaseModel):
    """导航中止。"""

    kind: Literal["aborted"] = "aborted"
    leaf_id: str | None


class NavigationFailedOutcome(BaseModel):
    """导航失败。"""

    kind: Literal["failed"] = "failed"
    leaf_id: str | None
    error: OperationError


NavigationOutcome: TypeAlias = Annotated[
    NavigationCompletedOutcome
    | NavigationDeclinedOutcome
    | NavigationAbortedOutcome
    | NavigationFailedOutcome,
    Field(discriminator="kind"),
]
"""导航结果（对应 TS ``NavigationOutcome``）。"""


# ---------------------------------------------------------------------------
# ResumeOutcome（嵌套判别联合）
# ---------------------------------------------------------------------------


class ResumeRunOutcome(BaseModel):
    """恢复为运行。"""

    operation: Literal["run"] = "run"
    run_id: str
    kind: Literal["completed", "aborted", "failed", "suspended"]  # noqa: F722
    leaf_id: str
    final_entry_id: str | None = None
    final_message: AssistantMessage | None = None
    error: OperationError | None = None
    deferred: object | None = None


class ResumeCompactionOutcome(BaseModel):
    """恢复为压缩。"""

    operation: Literal["compaction"] = "compaction"
    run_id: str
    kind: Literal["completed", "declined", "aborted", "failed"]  # noqa: F722
    leaf_id: str
    entry: object | None = None
    error: OperationError | None = None


class ResumeNavigationOutcome(BaseModel):
    """恢复为导航。"""

    operation: Literal["navigation"] = "navigation"
    run_id: str
    kind: Literal["completed", "declined", "aborted", "failed"]  # noqa: F722
    leaf_id: str | None
    new_leaf_id: str | None = None
    summary_entry: object | None = None
    error: OperationError | None = None


ResumeOutcome: TypeAlias = Annotated[
    ResumeRunOutcome | ResumeCompactionOutcome | ResumeNavigationOutcome,
    Field(discriminator="operation"),
]
"""恢复结果（对应 TS ``ResumeOutcome``）。"""


# ---------------------------------------------------------------------------
# 拒绝联合类型别名
# ---------------------------------------------------------------------------

RunRejected: TypeAlias = LaneBusy | InvalidMessage | UnknownSkill | UnknownTemplate | Closed
CompactionRejected: TypeAlias = LaneBusy | NothingToCompact | Closed
NavigationRejected: TypeAlias = LaneBusy | UnknownTarget | Closed
ResumeRejected: TypeAlias = LaneBusy | NothingToResume | MissingIdentities | Closed
QueueRejected: TypeAlias = NoActiveRun | InvalidMessage | Closed
CancelQueuedRejected: TypeAlias = UnknownQueueItem | Closed
AbortRejected: TypeAlias = NoActiveOperation | Closed

# ---------------------------------------------------------------------------
# 结果类型别名
# ---------------------------------------------------------------------------

RunResult: TypeAlias = Result[RunOutcome, RunRejected]
CompactionResult: TypeAlias = Result[CompactionOutcome, CompactionRejected]
NavigationResult: TypeAlias = Result[NavigationOutcome, NavigationRejected]
QueueResult: TypeAlias = Result[object, QueueRejected]
CancelQueuedResult: TypeAlias = Result[object, CancelQueuedRejected]
RecordUsageResult: TypeAlias = Result[None, Closed]
AbortResult: TypeAlias = Result[object, AbortRejected]
ResumeResult: TypeAlias = Result[ResumeOutcome, ResumeRejected]
CreateLaneResult: TypeAlias = Result[object, LaneExists | InvalidLane | UnknownTarget | Closed]


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class NavigateOptions(BaseModel):
    """导航选项（对应 TS ``NavigateOptions``）。"""

    summarize: bool = False
    custom_instructions: str | None = None
    label: str | None = None


class SuspendedOperation(BaseModel):
    """挂起操作（对应 TS ``SuspendedOperation``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lane: str
    kind: Literal["run", "compaction", "navigation"]
    id: str
    started_at: int
    reason: Literal["crash", "deferred"]
    prompt: list[AgentMessage] | None = None
    deferred: object | None = None  # DeferredHandle 占位
    aborting: object | None = None  # { steer: AgentMessage[]; followUp: AgentMessage[] }
    missing: object = Field(default_factory=lambda: {"tools": [], "models": []})


class LaneOperationInfo(BaseModel):
    """车道操作信息。"""

    id: str
    kind: Literal["run", "compaction", "navigation"]
    status: Literal["running", "suspended", "aborting"]


class LaneInfo(BaseModel):
    """车道信息（对应 TS ``LaneInfo``）。"""

    name: str
    leaf_id: str | None
    operation: LaneOperationInfo | None = None


class QueuedItem(BaseModel):
    """队列项（对应 TS ``QueuedItem``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    entry_id: str
    message: AgentMessage


class LaneQueues(BaseModel):
    """车道队列。"""

    steer: list[QueuedItem] = Field(default_factory=list)
    follow_up: list[QueuedItem] = Field(default_factory=list)
    next_run: list[QueuedItem] = Field(default_factory=list)


class PendingWrite(BaseModel):
    """待写入条目。"""

    id: str
    entry: object  # ProvisionedEntry 占位


class LaneSnapshot(BaseModel):
    """车道快照（对应 TS ``LaneSnapshot``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lane: str
    transcript: list[Entry]
    leaf_id: str | None
    operation: LaneOperationInfo | None = None
    queues: LaneQueues = Field(default_factory=LaneQueues)
    pending_writes: list[PendingWrite] = Field(default_factory=list)
    faulted: bool = False


class LaneInfoWithSuspended(LaneInfo):
    """带挂起信息的车道信息。"""

    suspended: SuspendedOperation | None = None


class SessionSnapshot(BaseModel):
    """会话快照（对应 TS ``SessionSnapshot``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lanes: list[LaneInfoWithSuspended] = Field(default_factory=list)
    faulted: bool = False


# ---------------------------------------------------------------------------
# ActionInfo（14 种判别联合）
# ---------------------------------------------------------------------------


class AppendEntryAction(BaseModel):
    """追加条目动作。"""

    kind: Literal["append_entry"] = "append_entry"
    entry_type: str
    entry_id: str


class AppendRecordAction(BaseModel):
    """追加记录动作。"""

    kind: Literal["append_record"] = "append_record"
    record_type: str


class MoveLaneAction(BaseModel):
    """移动车道动作。"""

    kind: Literal["move_lane"] = "move_lane"
    to: str | None


class SetFactAction(BaseModel):
    """设置事实动作。"""

    kind: Literal["set_fact"] = "set_fact"
    fact: Literal["name", "label"]


class TryFinishRunAction(BaseModel):
    """尝试完成运行动作。"""

    kind: Literal["try_finish_run"] = "try_finish_run"
    outcome: Literal["completed", "failed"]


class FinishOperationAction(BaseModel):
    """完成操作动作。"""

    kind: Literal["finish_operation"] = "finish_operation"
    outcome: Literal["completed", "declined", "failed", "aborted"]


class CommitFollowUpAction(BaseModel):
    """提交跟进动作。"""

    kind: Literal["commit_follow_up"] = "commit_follow_up"


class ConsumeQueueItemAction(BaseModel):
    """消费队列项动作。"""

    kind: Literal["consume_queue_item"] = "consume_queue_item"
    queue: Literal["steer", "followUp"]
    entry_id: str


class ApplyPendingWriteAction(BaseModel):
    """应用待写入动作。"""

    kind: Literal["apply_pending_write"] = "apply_pending_write"
    entry_id: str


class StreamAssistantAction(BaseModel):
    """流式 assistant 动作。"""

    kind: Literal["stream_assistant"] = "stream_assistant"
    step: Literal["assistant", "compaction", "branch_summary"]
    attempt: int


class ExecuteToolAction(BaseModel):
    """执行工具动作。"""

    kind: Literal["execute_tool"] = "execute_tool"
    tool_call_id: str
    tool_name: str


class FetchDeferredAction(BaseModel):
    """获取延迟动作。"""

    kind: Literal["fetch_deferred"] = "fetch_deferred"
    provider: str
    id: str


class CancelDeferredAction(BaseModel):
    """取消延迟动作。"""

    kind: Literal["cancel_deferred"] = "cancel_deferred"
    provider: str
    id: str


class HookAction(BaseModel):
    """钩子动作。"""

    kind: Literal["hook"] = "hook"
    name: str  # HookName


class SleepAction(BaseModel):
    """休眠动作。"""

    kind: Literal["sleep"] = "sleep"
    delay_ms: int


ActionInfo: TypeAlias = Annotated[
    AppendEntryAction
    | AppendRecordAction
    | MoveLaneAction
    | SetFactAction
    | TryFinishRunAction
    | FinishOperationAction
    | CommitFollowUpAction
    | ConsumeQueueItemAction
    | ApplyPendingWriteAction
    | StreamAssistantAction
    | ExecuteToolAction
    | FetchDeferredAction
    | CancelDeferredAction
    | HookAction
    | SleepAction,
    Field(discriminator="kind"),
]
"""动作信息（对应 TS ``ActionInfo``，14 种判别联合）。"""


# ---------------------------------------------------------------------------
# HookName
# ---------------------------------------------------------------------------

HookName: TypeAlias = Literal[
    "before_run",
    "before_resume",
    "before_run_end",
    "transform_context",
    "before_request",
    "before_payload",
    "after_response",
    "before_tool",
    "after_tool",
    "before_compaction",
    "before_navigation",
]
"""钩子名称（对应 TS ``HookName``，11 种）。"""


# ---------------------------------------------------------------------------
# 接口 / 协议
# ---------------------------------------------------------------------------


class Hooks(Protocol):
    """钩子注册接口（对应 TS ``Hooks``）。"""

    def on(
        self,
        name: HookName | str,
        handler: Callable[[object], object | Awaitable[object]],
        options: dict[str, str] | None = None,
    ) -> Callable[[], None]: ...


class Events(Protocol):
    """事件注册接口（对应 TS ``Events``）。"""

    def on(
        self,
        type: str,
        listener: Callable[[object], None | Awaitable[None]],
    ) -> Callable[[], None]: ...


class PassiveRegistry:
    """空实现注册器（同时实现 Hooks + Events，对应 TS ``PassiveRegistry``）。"""

    def on(
        self,
        _name: str,
        _handler: Callable[[object], object | Awaitable[object]],
        _options: dict[str, str] | None = None,
    ) -> Callable[[], None]:
        return lambda: None


# ---------------------------------------------------------------------------
# 执行上下文（ExecutionContext / ExecutionSpan）
# ---------------------------------------------------------------------------


SpanAttributes: TypeAlias = dict[str, str | int | bool | None]
"""跨度属性（对应 TS ``SpanAttributes``）。"""


class SpanEnd(BaseModel):
    """跨度结束（对应 TS ``SpanEnd``）。"""

    status: Literal["ok", "error"]
    error: object | None = None  # { name: str; message: str }
    attributes: SpanAttributes | None = None


@runtime_checkable
class ExecutionSpan(Protocol):
    """执行跨度（对应 TS ``ExecutionSpan``）。"""

    def add_event(self, name: str, attributes: SpanAttributes | None = None) -> None: ...
    def set_attributes(self, attributes: SpanAttributes) -> None: ...
    def end(self, result: SpanEnd) -> None: ...


@runtime_checkable
class ExecutionContext(Protocol):
    """执行上下文（对应 TS ``ExecutionContext``）。"""

    def start_span(
        self, name: str, attributes: SpanAttributes | None = None
    ) -> ExecutionSpan: ...


# ---------------------------------------------------------------------------
# 工具与资源类型别名
# ---------------------------------------------------------------------------


class HarnessTool(AgentTool):
    """harness 工具（对应 TS ``HarnessTool``）。"""

    replay: Literal["never", "safe"] | None = None


Resources: TypeAlias = AgentHarnessResources
StreamOptions: TypeAlias = AgentHarnessStreamOptions
StreamOptionsPatch: TypeAlias = dict[str, object]  # Partial<SimpleStreamOptions>

EntryProjector: TypeAlias = Callable[[Entry], list[AgentMessage] | Awaitable[list[AgentMessage]]]
"""条目投影器（对应 TS ``EntryProjector``）。"""


# ---------------------------------------------------------------------------
# AgentHarnessOptions
# ---------------------------------------------------------------------------


class AgentHarnessOptions(BaseModel):
    """harness 构造选项（对应 TS ``AgentHarnessOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session: object  # Session 占位，避免循环导入
    models: object = Field(default=None)  # Models 占位（pi-ai）
    model: Model
    thinking_level: ThinkingLevel = "off"
    active_tool_names: list[str] | None = None
    tools: list[HarnessTool] | None = None
    tool_context: object | None = None
    system_prompt: str | None = None
    resources: Resources | None = None
    stream_options: StreamOptions | None = None
    retry: RetryPolicy | None = None
    compaction: CompactionSettings | None = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    tool_execution: Literal["sequential", "parallel"] | None = None
    drive: Literal["automatic", "manual"] | None = None
    to_provider_messages: Callable[
        [list[AgentMessage]], list[object] | Awaitable[list[object]]
    ] | None = None
    entry_projectors: dict[str, EntryProjector] | None = None
    context: ExecutionContext | None = None


# ---------------------------------------------------------------------------
# WatchHandle
# ---------------------------------------------------------------------------

TSnapshot = TypeVar("TSnapshot")


class WatchHandle(BaseModel, Generic[TSnapshot]):
    """快照监听句柄（对应 TS ``WatchHandle<TSnapshot>``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    snapshot: TSnapshot

    def start(self, listener: Callable[[object], None]) -> None: ...
    def unsubscribe(self) -> None: ...


# ---------------------------------------------------------------------------
# AgentLane 接口（Protocol）
# ---------------------------------------------------------------------------


@runtime_checkable
class AgentLane(Protocol):
    """AgentLane 接口（对应 TS ``AgentLane``，约 35 个方法）。"""

    name: str
    session: SessionTree

    # --- 运行控制 ---
    async def prompt(self, text: str, images: list[ImageContent] | None = None) -> RunResult: ...
    async def skill(self, name: str, additional_instructions: str | None = None) -> RunResult: ...
    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> RunResult: ...

    # --- 操作管理 ---
    async def compact(
        self, options: dict[str, str] | None = None
    ) -> CompactionResult: ...
    async def navigate_tree(
        self, target_id: str | None, options: NavigateOptions | None = None
    ) -> NavigationResult: ...
    async def resume(self) -> ResumeResult: ...
    async def abort(self) -> AbortResult: ...

    # --- 队列操作 ---
    async def steer(self, text: str, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def next_run(self, text: str, images: list[ImageContent] | None = None) -> QueueResult: ...
    async def cancel_queued(self, entry_id: str) -> CancelQueuedResult: ...

    # --- 用量记录 ---
    async def record_usage(
        self, usage: Usage, options: dict[str, object] | None = None
    ) -> RecordUsageResult: ...

    # --- 空闲等待 ---
    async def wait_for_idle(self) -> None: ...
    async def run_when_idle(self, callback: Callable[[], None | Awaitable[None]]) -> None: ...

    # --- 动作执行 ---
    async def peek_action(self) -> ActionInfo | None: ...
    async def execute_action(self) -> ActionInfo | None: ...
    async def run_to_completion(self) -> None: ...

    # --- 属性访问 ---
    async def get_leaf_id(self) -> str | None: ...
    async def get_model(self) -> Model: ...
    async def set_model(self, model: Model) -> None: ...
    async def get_thinking_level(self) -> ThinkingLevel: ...
    async def set_thinking_level(self, level: ThinkingLevel) -> None: ...
    async def get_active_tools(self) -> list[str]: ...
    async def set_active_tools(self, names: list[str]) -> None: ...

    # --- 监听 ---
    async def watch(self) -> WatchHandle[LaneSnapshot]: ...