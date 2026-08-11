"""Agent 循环入口（对应 ``agent-loop.ts`` 入口函数）。

提供两个层次的 API：
- **高层（EventStream）**：``agent_loop`` / ``agent_loop_continue`` — 返回 ``AgentEventStream``
- **低层（直接调用）**：``run_agent_loop`` / ``run_agent_loop_continue`` — 阻塞直到完成
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from .agent_loop_core import run_loop
from .agent_loop_tools import AgentEventSink
from .cancellation import CancellationToken
from .stream_fn import get_default_stream_fn
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AgentStartEvent,
    MessageEndEvent,
    MessageStartEvent,
    StreamFn,
    TurnStartEvent,
)

# ---------------------------------------------------------------------------
# AgentEventStream
# ---------------------------------------------------------------------------


class AgentEventStream:
    """agent 事件流（对应 TS ``EventStream<AgentEvent, AgentMessage[]>``）。

    支持 push 生产和 async for 消费。``agent_end`` 事件自动触发内部 end。
    调用方通过 ``async for`` 消费事件，完成后通过 ``result()`` 获取最终消息列表。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AgentEvent | None] = asyncio.Queue()
        self._final_messages: list[AgentMessage] | None = None
        self._ended = False

    def push(self, event: AgentEvent) -> None:
        """向流中推送事件。``agent_end`` 事件自动触发 ``end()``。"""
        if self._ended:
            return
        self._queue.put_nowait(event)
        if isinstance(event, AgentEndEvent):
            self.end(event.messages)

    def end(self, final_messages: list[AgentMessage]) -> None:
        """结束流。"""
        if self._ended:
            return
        self._ended = True
        self._final_messages = final_messages
        self._queue.put_nowait(None)  # sentinel

    async def __aiter__(self) -> AsyncIterator[AgentEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event

    async def result(self) -> list[AgentMessage]:
        """返回最终消息列表（对应 TS ``response.result()``）。

        流未结束时阻塞等待；已结束时立即返回。
        """
        if self._final_messages is not None:
            return self._final_messages
        # 等待流结束
        async for _event in self:
            pass
        if self._final_messages is None:
            raise RuntimeError("Stream ended without final messages")
        return self._final_messages


# ---------------------------------------------------------------------------
# agent_loop（高层 API）
# ---------------------------------------------------------------------------


def agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """启动新 agent 循环（带 prompt），返回 ``AgentEventStream``（对应 TS ``agentLoop``）。

    调用方通过 ``async for event in stream:`` 消费事件，
    完成后通过 ``await stream.result()`` 获取最终消息列表。
    """
    stream = AgentEventStream()

    async def _emit(event: AgentEvent) -> None:
        stream.push(event)

    asyncio.ensure_future(
        run_agent_loop(prompts, context, config, _emit, signal, stream_fn)
    )

    return stream


def agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    stream_fn: StreamFn | None = None,
) -> AgentEventStream:
    """继续 agent 循环（无新 prompt），返回 ``AgentEventStream``（对应 TS ``agentLoopContinue``）。

    要求 context.messages 非空且最后一条不是 assistant。
    """
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    stream = AgentEventStream()

    async def _emit(event: AgentEvent) -> None:
        stream.push(event)

    asyncio.ensure_future(
        run_agent_loop_continue(context, config, _emit, signal, stream_fn)
    )

    return stream


# ---------------------------------------------------------------------------
# run_agent_loop / run_agent_loop_continue（低层 API）
# ---------------------------------------------------------------------------


async def run_agent_loop(
    prompts: list[AgentMessage],
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: CancellationToken | None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """直接运行 agent 循环（不经过 EventStream），返回所有新消息（对应 TS ``runAgentLoop``）。"""
    new_messages: list[AgentMessage] = list(prompts)
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages) + list(prompts),
        tools=context.tools,
    )

    # 通知消费方：agent 循环正式开始（消费方可据此初始化 UI / 计时）
    await _emit_safe(emit, AgentStartEvent())
    # 通知消费方：第一个对话回合开始（一次 agent 运行可包含多个 turn）
    await _emit_safe(emit, TurnStartEvent())
    for prompt in prompts:
        # 逐条广播用户输入消息的「开始」事件
        await _emit_safe(emit, MessageStartEvent(message=prompt))
        # 紧接着广播同一条消息的「结束」事件——prompt 是既成事实，无需流式
        await _emit_safe(emit, MessageEndEvent(message=prompt))

    # 解析实际使用的 stream_fn：显式传入优先，否则回退到模块默认注册项
    resolved_fn = stream_fn if stream_fn is not None else get_default_stream_fn()
    # 进入主循环 run_loop：内部「LLM 流式生成 → 工具调用 → 结果回写」反复迭代，
    # 全程通过 emit 把事件分流给消费方；此处阻塞直到循环自然结束
    await run_loop(current_context, new_messages, config, signal, emit, resolved_fn)
    return new_messages


async def run_agent_loop_continue(
    context: AgentContext,
    config: AgentLoopConfig,
    emit: AgentEventSink,
    signal: CancellationToken | None,
    stream_fn: StreamFn | None = None,
) -> list[AgentMessage]:
    """直接继续 agent 循环（对应 TS ``runAgentLoopContinue``）。"""
    if not context.messages:
        raise ValueError("Cannot continue: no messages in context")

    if context.messages[-1].role == "assistant":
        raise ValueError("Cannot continue from message role: assistant")

    new_messages: list[AgentMessage] = []
    current_context = AgentContext(
        system_prompt=context.system_prompt,
        messages=list(context.messages),
        tools=context.tools,
    )

    await _emit_safe(emit, AgentStartEvent())
    await _emit_safe(emit, TurnStartEvent())

    resolved_fn = stream_fn if stream_fn is not None else get_default_stream_fn()
    await run_loop(current_context, new_messages, config, signal, emit, resolved_fn)
    return new_messages


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


async def _emit_safe(emit: AgentEventSink, event: AgentEvent) -> None:
    """同步或异步发射事件。"""
    result = emit(event)
    if result is not None:
        await result
