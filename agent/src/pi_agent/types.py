"""核心类型定义（从 ``pi_ai.types`` 再导出 + agent 专用类型）。

共享类型（ThinkingLevel, StopReason, Model, Message, Context 等）从 pi_ai
导入；agent 独占类型（AgentMessage, AgentState, AgentLoopConfig, AgentEvent 等）
在本模块中定义。

命名约定：Python 使用 ``snake_case``（工作区规则 coding.md），字段名与 TS 的
``camelCase`` 一一对应。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Annotated, Literal, TypeAlias

# ---------------------------------------------------------------------------
# 从 pi_ai 导入共享类型
# ---------------------------------------------------------------------------
from pi_ai.types import (
    # 流事件
    AssistantAbortedEvent,
    AssistantErrorEvent,
    # 消息
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    AssistantTextDelta,
    AssistantThinkingDelta,
    AssistantToolCallEnd,
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    AssistantUsageDelta,
    # 内容块
    ContentBlock,
    # 模型 / 上下文 / 工具
    Context,
    # Usage / Cost
    Cost,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    # 基础标量
    StopReason,
    # StreamFn
    StreamFn,
    TextContent,
    ThinkingBlock,
    ThinkingLevel,
    Tool,
    ToolCallContent,
    ToolExecutionMode,
    ToolResultContent,
    ToolResultMessage,
    Transport,
    Usage,
    UserMessage,
)
from pydantic import BaseModel, ConfigDict, Field

from .cancellation import CancellationToken

# ---------------------------------------------------------------------------
# 基础标量
# ---------------------------------------------------------------------------

QueueMode: TypeAlias = Literal["all", "one-at-a-time"]
"""队列排空模式。"""


# ---------------------------------------------------------------------------
# 自定义消息。
# ---------------------------------------------------------------------------


class BashExecutionMessage(BaseModel):
    """bash 工具执行消息。"""

    role: Literal["bashExecution"] = "bashExecution"
    command: str
    output: str
    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    full_output_path: str | None = None
    timestamp: int
    exclude_from_context: bool = False


class CustomMessage(BaseModel):
    """应用自定义消息。"""

    role: Literal["custom"] = "custom"
    custom_type: str
    content: str | list[TextContent | ImageContent]
    display: bool = True
    details: object | None = None
    timestamp: int


class BranchSummaryMessage(BaseModel):
    """分支摘要消息。"""

    role: Literal["branchSummary"] = "branchSummary"
    summary: str
    from_id: str
    timestamp: int


class CompactionSummaryMessage(BaseModel):
    """上下文压缩摘要消息。"""

    role: Literal["compactionSummary"] = "compactionSummary"
    summary: str
    tokens_before: int
    timestamp: int


AgentMessage: TypeAlias = Annotated[
    UserMessage
    | AssistantMessage
    | ToolResultMessage
    | BashExecutionMessage
    | CustomMessage
    | BranchSummaryMessage
    | CompactionSummaryMessage,
    Field(discriminator="role"),
]
"""应用层消息联合。"""


# ---------------------------------------------------------------------------
# 工具（AgentTool）
# ---------------------------------------------------------------------------


class AgentToolResult(BaseModel):
    """工具执行的最终或部分结果。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    content: list[TextContent | ImageContent]
    details: object
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    terminate: bool = False


AgentToolUpdateCallback: TypeAlias = Callable[[AgentToolResult], None]
"""工具流式进度回调（作用域限定于当前 execute 调用）。"""


class AgentTool(Tool):
    """agent 运行时使用的工具定义。"""

    label: str
    prepare_arguments: Callable[[dict[str, object]], dict[str, object]] | None = None
    execute: Callable[
        [
            str,
            dict[str, object],
            CancellationToken | None,
            AgentToolUpdateCallback | None,
        ],
        Awaitable[AgentToolResult],
    ]
    execution_mode: ToolExecutionMode | None = None


# ---------------------------------------------------------------------------
# AgentContext / AgentState
# ---------------------------------------------------------------------------


class AgentContext(BaseModel):
    """传入低层 agent 循环的上下文快照。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    system_prompt: str
    messages: list[AgentMessage]
    tools: list[AgentTool] | None = None


class AgentState:
    """公共 agent 状态。    

    ``tools`` / ``messages`` 使用属性语义：赋值时拷贝顶层数组，防止外部变异。
    """

    __slots__ = (
        "_messages",
        "_model",
        "_thinking_level",
        "_tools",
        "error_message",
        "is_streaming",
        "pending_tool_calls",
        "streaming_message",
        "system_prompt",
    )

    def __init__(
        self,
        *,
        system_prompt: str,
        model: Model,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        self.system_prompt = system_prompt
        self._model = model
        self._thinking_level = thinking_level
        self._tools = list(tools or [])
        self._messages = list(messages or [])
        self.is_streaming = False
        self.streaming_message: AgentMessage | None = None
        self.pending_tool_calls: frozenset[str] = frozenset()
        self.error_message: str | None = None

    @property
    def model(self) -> Model:
        """当前使用的模型。"""
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        self._model = value

    @property
    def thinking_level(self) -> ThinkingLevel:
        """请求的思考级别。"""
        return self._thinking_level

    @thinking_level.setter
    def thinking_level(self, value: ThinkingLevel) -> None:
        self._thinking_level = value

    @property
    def tools(self) -> list[AgentTool]:
        """可用工具列表（赋值时拷贝顶层数组）。"""
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        """对话记录（赋值时拷贝顶层数组）。"""
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)


# ---------------------------------------------------------------------------
# 钩子上下文与返回值
# ---------------------------------------------------------------------------


class BeforeToolCallResult(BaseModel):
    """``before_tool_call`` 返回值：可阻断工具执行。"""

    block: bool = False
    reason: str | None = None


class AfterToolCallResult(BaseModel):
    """``after_tool_call`` 返回值：逐字段覆盖工具结果（无深合并）。

    使用 ``model_fields_set`` 判断调用方提供了哪些字段。
    """

    content: list[TextContent | ImageContent] | None = None
    details: object | None = None
    is_error: bool | None = None
    usage: Usage | None = None
    terminate: bool | None = None


class BeforeToolCallContext(BaseModel):
    """``before_tool_call`` 入参。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, object]
    context: AgentContext


class AfterToolCallContext(BaseModel):
    """``after_tool_call`` 入参。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    assistant_message: AssistantMessage
    tool_call: ToolCallContent
    args: dict[str, object]
    result: AgentToolResult
    is_error: bool
    context: AgentContext


class ShouldStopAfterTurnContext(BaseModel):
    """``should_stop_after_turn`` 入参。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message: AssistantMessage
    tool_results: list[ToolResultMessage]
    context: AgentContext
    new_messages: list[AgentMessage]


PrepareNextTurnContext: TypeAlias = ShouldStopAfterTurnContext


class AgentLoopTurnUpdate(BaseModel):
    """替换下一回合运行时状态。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    context: AgentContext | None = None
    model: Model | None = None
    thinking_level: ThinkingLevel | None = None


# ---------------------------------------------------------------------------
# AgentLoopConfig
# ---------------------------------------------------------------------------


class AgentLoopConfig(SimpleStreamOptions):
    """低层循环配置。

    必需：``model``、``convert_to_llm``。所有钩子契约：**不得抛异常**，
    失败时返回安全回退值。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    convert_to_llm: Callable[
        [list[AgentMessage]], Awaitable[list[Message]] | list[Message]
    ]
    transform_context: (
        Callable[
            [list[AgentMessage], CancellationToken | None],
            Awaitable[list[AgentMessage]],
        ]
        | None
    ) = None
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext], Awaitable[bool] | bool] | None
    ) = None
    prepare_next_turn: (
        Callable[
            [PrepareNextTurnContext],
            Awaitable[AgentLoopTurnUpdate | None] | AgentLoopTurnUpdate | None,
        ]
        | None
    ) = None
    get_steering_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    get_follow_up_messages: Callable[[], Awaitable[list[AgentMessage]]] | None = None
    tool_execution: ToolExecutionMode = "parallel"
    before_tool_call: (
        Callable[
            [BeforeToolCallContext, CancellationToken | None],
            Awaitable[BeforeToolCallResult | None],
        ]
        | None
    ) = None
    after_tool_call: (
        Callable[
            [AfterToolCallContext, CancellationToken | None],
            Awaitable[AfterToolCallResult | None],
        ]
        | None
    ) = None


# ---------------------------------------------------------------------------
# AgentEvent
# ---------------------------------------------------------------------------


class AgentStartEvent(BaseModel):
    """agent 开始处理。"""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """运行最终事件；其订阅者仍计入运行落定。"""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class TurnStartEvent(BaseModel):
    """新回合开始（一次 LLM 调用 + 工具执行）。"""

    type: Literal["turn_start"] = "turn_start"


class TurnEndEvent(BaseModel):
    """回合完成。"""

    type: Literal["turn_end"] = "turn_end"
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    """任意消息开始（user / assistant / toolResult）。"""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(BaseModel):
    """assistant 消息流式更新。"""

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class MessageEndEvent(BaseModel):
    """消息完成。"""

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    """工具执行开始。"""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: object


class ToolExecutionUpdateEvent(BaseModel):
    """工具执行进度。"""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: object
    partial_result: object


class ToolExecutionEndEvent(BaseModel):
    """工具执行结束。"""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: object
    is_error: bool


AgentEvent: TypeAlias = Annotated[
    AgentStartEvent
    | AgentEndEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent,
    Field(discriminator="type"),
]
"""agent 生命周期事件联合。"""


__all__ = [
    # 从 pi_ai 再导出
    "AssistantAbortedEvent",
    "AssistantErrorEvent",
    "AssistantMessage",
    "AssistantMessageEvent",
    "AssistantMessageSnapshot",
    "AssistantStreamEnd",
    "AssistantTextDelta",
    "AssistantThinkingDelta",
    "AssistantToolCallEnd",
    "AssistantToolCallStart",
    "AssistantToolCallUpdate",
    "AssistantUsageDelta",
    "ContentBlock",
    "Context",
    "Cost",
    "ImageContent",
    "Message",
    "Model",
    "SimpleStreamOptions",
    "StopReason",
    "StreamFn",
    "TextContent",
    "ThinkingBlock",
    "ThinkingLevel",
    "Tool",
    "ToolCallContent",
    "ToolExecutionMode",
    "ToolResultContent",
    "ToolResultMessage",
    "Transport",
    "Usage",
    "UserMessage",
    # Agent 专用
    "AfterToolCallContext",
    "AfterToolCallResult",
    "AgentContext",
    "AgentEndEvent",
    "AgentEvent",
    "AgentLoopConfig",
    "AgentLoopTurnUpdate",
    "AgentMessage",
    "AgentStartEvent",
    "AgentState",
    "AgentTool",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "BashExecutionMessage",
    "BeforeToolCallContext",
    "BeforeToolCallResult",
    "BranchSummaryMessage",
    "CancellationToken",
    "CompactionSummaryMessage",
    "CustomMessage",
    "MessageEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "PrepareNextTurnContext",
    "QueueMode",
    "ShouldStopAfterTurnContext",
    "ToolExecutionEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "TurnEndEvent",
    "TurnStartEvent",
]
