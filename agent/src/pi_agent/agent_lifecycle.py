"""Agent 生命周期与状态管理（对应 ``agent.ts`` 的辅助类型与工厂函数）。

包含：
- ``MutableAgentState``：可变 agent 状态（组合 AgentState 接口，pending_tool_calls 为可变 set）
- ``ActiveRun``：活跃运行句柄（asyncio.Event + CancellationToken）
- ``AgentOptions``：Agent 构造选项（Pydantic BaseModel）
- ``default_convert_to_llm``：默认 LLM 消息转换器
- 常量：``EMPTY_USAGE`` / ``DEFAULT_MODEL``
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from .cancellation import CancellationToken
from .types import (
    AfterToolCallContext,
    AfterToolCallResult,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentState,
    AgentTool,
    BeforeToolCallContext,
    BeforeToolCallResult,
    Message,
    Model,
    PrepareNextTurnContext,
    QueueMode,
    ShouldStopAfterTurnContext,
    ThinkingLevel,
    ToolExecutionMode,
    Transport,
    Usage,
)

# ---------------------------------------------------------------------------
# 默认模型占位
# ---------------------------------------------------------------------------


class _DefaultModel:
    """满足 ``Model`` Protocol 的默认模型占位（对应 TS ``DEFAULT_MODEL``）。"""

    api: str = "unknown"
    provider: str = "unknown"
    model_id: str = "unknown"


DEFAULT_MODEL: Model = _DefaultModel()


# ---------------------------------------------------------------------------
# 零用量占位
# ---------------------------------------------------------------------------

EMPTY_USAGE = Usage(
    input=0,
    output=0,
    cache_read=0,
    cache_write=0,
    total_tokens=0,
)


# ---------------------------------------------------------------------------
# MutableAgentState
# ---------------------------------------------------------------------------


class MutableAgentState:
    """可变 agent 状态（对应 TS ``MutableAgentState``）。

    与 ``AgentState`` 接口一致，但 ``pending_tool_calls`` 为可变 ``set[str]``
    （而非 ``frozenset[str]``），方便在 ``processEvents`` 中直接 add/discard。
    """

    def __init__(
        self,
        *,
        system_prompt: str = "",
        model: Model = DEFAULT_MODEL,
        thinking_level: ThinkingLevel = "off",
        tools: list[AgentTool] | None = None,
        messages: list[AgentMessage] | None = None,
    ) -> None:
        self.system_prompt: str = system_prompt
        self._model: Model = model
        self._thinking_level: ThinkingLevel = thinking_level
        self._tools: list[AgentTool] = list(tools or [])
        self._messages: list[AgentMessage] = list(messages or [])
        self.is_streaming: bool = False
        self.streaming_message: AgentMessage | None = None
        self.pending_tool_calls: set[str] = set()
        self.error_message: str | None = None

    @property
    def model(self) -> Model:
        return self._model

    @model.setter
    def model(self, value: Model) -> None:
        self._model = value

    @property
    def thinking_level(self) -> ThinkingLevel:
        return self._thinking_level

    @thinking_level.setter
    def thinking_level(self, value: ThinkingLevel) -> None:
        self._thinking_level = value

    @property
    def tools(self) -> list[AgentTool]:
        return self._tools

    @tools.setter
    def tools(self, value: list[AgentTool]) -> None:
        self._tools = list(value)

    @property
    def messages(self) -> list[AgentMessage]:
        return self._messages

    @messages.setter
    def messages(self, value: list[AgentMessage]) -> None:
        self._messages = list(value)


def create_mutable_agent_state(
    initial_state: AgentState | None = None,
) -> MutableAgentState:
    """工厂函数：从可选初始状态创建 ``MutableAgentState``（对应 TS ``createMutableAgentState``）。"""
    if initial_state is None:
        return MutableAgentState()
    return MutableAgentState(
        system_prompt=initial_state.system_prompt,
        model=initial_state.model,
        thinking_level=initial_state.thinking_level,
        tools=list(initial_state.tools),
        messages=list(initial_state.messages),
    )


# ---------------------------------------------------------------------------
# ActiveRun
# ---------------------------------------------------------------------------


class ActiveRun:
    """活跃运行句柄（对应 TS ``ActiveRun``）。

    - ``completion_event``：运行完成时 set，``wait_for_idle`` 通过 ``await event.wait()`` 等待
    - ``cancellation_token``：当前运行的取消令牌
    """

    def __init__(self) -> None:
        self.completion_event: asyncio.Event = asyncio.Event()
        self.cancellation_token: CancellationToken = CancellationToken()


# ---------------------------------------------------------------------------
# AgentOptions
# ---------------------------------------------------------------------------


class AgentOptions(BaseModel):
    """Agent 构造选项（对应 TS ``AgentOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    initial_state: AgentState | None = None
    convert_to_llm: (
        Callable[[list[AgentMessage]], Awaitable[list[Message]] | list[Message]] | None
    ) = None
    transform_context: (
        Callable[
            [list[AgentMessage], CancellationToken | None],
            Awaitable[list[AgentMessage]],
        ]
        | None
    ) = None
    stream_fn: object | None = (
        None  # StreamFn | None（Protocol 无法被 Pydantic isinstance 校验）
    )
    get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = None
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
    should_stop_after_turn: (
        Callable[[ShouldStopAfterTurnContext], Awaitable[bool] | bool] | None
    ) = None
    prepare_next_turn: (
        Callable[[], Awaitable[AgentLoopTurnUpdate | None] | AgentLoopTurnUpdate | None]
        | None
    ) = None
    prepare_next_turn_with_context: (
        Callable[
            [PrepareNextTurnContext],
            Awaitable[AgentLoopTurnUpdate | None] | AgentLoopTurnUpdate | None,
        ]
        | None
    ) = None
    steering_mode: QueueMode = "one-at-a-time"
    follow_up_mode: QueueMode = "one-at-a-time"
    session_id: str | None = None
    transport: Transport | None = None
    max_retry_delay_ms: int | None = None
    tool_execution: ToolExecutionMode = "parallel"


# ---------------------------------------------------------------------------
# default_convert_to_llm
# ---------------------------------------------------------------------------


def default_convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """默认 LLM 消息转换器（对应 TS ``defaultConvertToLlm``）。

    过滤出 role 为 ``"user"`` / ``"assistant"`` / ``"toolResult"`` 的消息，
    移除自定义消息类型（如 ``bashExecution`` / ``custom`` / ``branchSummary`` 等）。
    """
    return [msg for msg in messages if msg.role in ("user", "assistant", "toolResult")]
