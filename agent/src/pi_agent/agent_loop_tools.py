"""Agent 循环工具执行管线。

三阶段管线：
1. **Clearance**：prepare_tool_call — 工具查找 → before_tool_call 拦截 → 参数校验
2. **Execute**：execute_prepared_tool_call — 执行工具 + 收集进度更新
3. **Finalize**：finalize_executed_tool_call — after_tool_call 钩子逐字段覆盖结果

支持 sequential / parallel 两种执行模式。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Literal, TypeAlias, TypedDict

from .cancellation import CancellationToken
from .types import (
    AfterToolCallContext,
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AssistantMessage,
    BeforeToolCallContext,
    MessageEndEvent,
    MessageStartEvent,
    TextContent,
    ToolCallContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolResultMessage,
)

# ---------------------------------------------------------------------------
# 内部类型
# ---------------------------------------------------------------------------

AgentEventSink: TypeAlias = Callable[[AgentEvent], Awaitable[None] | None]
"""事件发射回调。"""


class ExecutedToolCallBatch(TypedDict):
    """工具批次执行结果。"""

    messages: list[ToolResultMessage]
    terminate: bool


class PreparedToolCall(TypedDict):
    """已准备就绪的工具调用。"""

    kind: Literal["prepared"]
    tool_call: ToolCallContent
    tool: AgentTool
    args: dict[str, object]


class ImmediateToolCallOutcome(TypedDict):
    """即刻完成的工具结果（未找到工具 / 被拦截 / 参数无效）。"""

    kind: Literal["immediate"]
    result: AgentToolResult
    is_error: bool


class ExecutedToolCallOutcome(TypedDict):
    """工具执行输出。"""

    result: AgentToolResult
    is_error: bool


class FinalizedToolCallOutcome(TypedDict):
    """最终化的工具调用结果。"""

    tool_call: ToolCallContent
    result: AgentToolResult
    is_error: bool


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


async def execute_tool_calls(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    """路由到 sequential 或 parallel 执行。"""
    tool_calls = [
        c for c in assistant_message.content if isinstance(c, ToolCallContent)
    ]
    if not tool_calls:
        return ExecutedToolCallBatch(messages=[], terminate=False)

    has_sequential = any(
        _find_tool(current_context, tc.name) is not None
        and _find_tool(current_context, tc.name).execution_mode == "sequential"  # type: ignore[union-attr]
        for tc in tool_calls
    )

    if config.tool_execution == "sequential" or has_sequential:
        return await execute_tool_calls_sequential(
            current_context, assistant_message, tool_calls, config, signal, emit
        )
    return await execute_tool_calls_parallel(
        current_context, assistant_message, tool_calls, config, signal, emit
    )


# ---------------------------------------------------------------------------
# Sequential
# ---------------------------------------------------------------------------


async def execute_tool_calls_sequential(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    """顺序执行工具调用：每个工具等待前一个完成。"""
    finalized_calls: list[FinalizedToolCallOutcome] = []
    messages: list[ToolResultMessage] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.name,
                args=tool_call.args,
            ),
        )

        preparation = await prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if preparation["kind"] == "immediate":
            finalized: FinalizedToolCallOutcome = FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation["result"],
                is_error=preparation["is_error"],
            )
        else:
            executed = await execute_prepared_tool_call(preparation, signal, emit)
            finalized = await finalize_executed_tool_call(
                current_context,
                assistant_message,
                preparation,
                executed,
                config,
                signal,
            )

        await emit_tool_execution_end(finalized, emit)
        tool_result_msg = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_msg, emit)
        finalized_calls.append(finalized)
        messages.append(tool_result_msg)

        if signal is not None and signal.aborted:
            break

    return ExecutedToolCallBatch(
        messages=messages,
        terminate=should_terminate_tool_batch(finalized_calls),
    )


# ---------------------------------------------------------------------------
# Parallel
# ---------------------------------------------------------------------------


async def execute_tool_calls_parallel(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_calls: list[ToolCallContent],
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    """并行执行工具调用：先全部准备，再并发执行。"""
    # 收集所有待执行的 callable（立即完成的用 lambda 包装）
    pending: list[Callable[[], Awaitable[FinalizedToolCallOutcome]]] = []

    for tool_call in tool_calls:
        await _emit(
            emit,
            ToolExecutionStartEvent(
                tool_call_id=tool_call.tool_call_id,
                tool_name=tool_call.name,
                args=tool_call.args,
            ),
        )

        preparation = await prepare_tool_call(
            current_context, assistant_message, tool_call, config, signal
        )
        if preparation["kind"] == "immediate":
            immediate = FinalizedToolCallOutcome(
                tool_call=tool_call,
                result=preparation["result"],
                is_error=preparation["is_error"],
            )
            await emit_tool_execution_end(immediate, emit)
            pending.append(_make_async_return(immediate))
        else:

            async def _deferred_exec(
                prep: PreparedToolCall = preparation,
            ) -> FinalizedToolCallOutcome:
                executed = await execute_prepared_tool_call(prep, signal, emit)
                finalized = await finalize_executed_tool_call(
                    current_context,
                    assistant_message,
                    prep,
                    executed,
                    config,
                    signal,
                )
                await emit_tool_execution_end(finalized, emit)
                return finalized

            pending.append(_deferred_exec)

        if signal is not None and signal.aborted:
            break

    ordered_finalized = await asyncio.gather(*[fn() for fn in pending])

    messages: list[ToolResultMessage] = []
    for finalized in ordered_finalized:
        tool_result_msg = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_msg, emit)
        messages.append(tool_result_msg)

    return ExecutedToolCallBatch(
        messages=messages,
        terminate=should_terminate_tool_batch(list(ordered_finalized)),
    )


# ---------------------------------------------------------------------------
# Phase 1: Clearance（工具查找 + 拦截 + 校验）
# ---------------------------------------------------------------------------


async def prepare_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    tool_call: ToolCallContent,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
) -> PreparedToolCall | ImmediateToolCallOutcome:
    """准备工具调用。

    三阶段：
    1. 查找工具 → 不存在则返回 ImmediateToolCallOutcome(is_error=True)
    2. before_tool_call 钩子 → 可 block 返回 ImmediateToolCallOutcome
    3. 参数校验 → 失败返回 ImmediateToolCallOutcome
    """
    tool = _find_tool(current_context, tool_call.name)
    if tool is None:
        return ImmediateToolCallOutcome(
            kind="immediate",
            result=create_error_tool_result(f"Tool {tool_call.name} not found"),
            is_error=True,
        )

    try:
        prepared_tool_call = prepare_tool_call_arguments(tool, tool_call)
        validated_args = validate_tool_arguments(tool, prepared_tool_call)

        if config.before_tool_call is not None:
            before_ctx = BeforeToolCallContext(
                assistant_message=assistant_message,
                tool_call=tool_call,
                args=validated_args,
                context=current_context,
            )
            before_result = await config.before_tool_call(before_ctx, signal)
            if signal is not None and signal.aborted:
                return ImmediateToolCallOutcome(
                    kind="immediate",
                    result=create_error_tool_result("Operation aborted"),
                    is_error=True,
                )
            if before_result is not None and before_result.block:
                return ImmediateToolCallOutcome(
                    kind="immediate",
                    result=create_error_tool_result(
                        before_result.reason or "Tool execution was blocked"
                    ),
                    is_error=True,
                )

        if signal is not None and signal.aborted:
            return ImmediateToolCallOutcome(
                kind="immediate",
                result=create_error_tool_result("Operation aborted"),
                is_error=True,
            )

        return PreparedToolCall(
            kind="prepared",
            tool_call=prepared_tool_call,
            tool=tool,
            args=validated_args,
        )
    except Exception as exc:
        return ImmediateToolCallOutcome(
            kind="immediate",
            result=create_error_tool_result(str(exc)),
            is_error=True,
        )


def prepare_tool_call_arguments(
    tool: AgentTool, tool_call: ToolCallContent
) -> ToolCallContent:
    """应用工具的 prepare_arguments 钩子。"""
    if tool.prepare_arguments is None:
        return tool_call
    prepared_args = tool.prepare_arguments(tool_call.args)
    if prepared_args is tool_call.args:
        return tool_call
    return tool_call.model_copy(update={"args": prepared_args})


def validate_tool_arguments(
    tool: AgentTool, tool_call: ToolCallContent
) -> dict[str, object]:
    """校验工具调用参数。

    当前实现为基础校验：确保 args 为 dict。
    TODO：基于 tool.parameters JSON Schema 做深度校验。
    """
    if not isinstance(tool_call.args, dict):
        raise ValueError(f"Tool {tool_call.name} arguments must be a dict")
    return tool_call.args


# ---------------------------------------------------------------------------
# Phase 2: Execute
# ---------------------------------------------------------------------------


async def execute_prepared_tool_call(
    prepared: PreparedToolCall,
    signal: CancellationToken | None,
    emit: AgentEventSink,
) -> ExecutedToolCallOutcome:
    """执行已准备的工具调用。"""
    update_events: list[Awaitable[None]] = []
    accepting_updates = True

    def _on_partial_result(partial_result: AgentToolResult) -> None:
        if not accepting_updates:
            return
        update_events.append(
            _emit(
                emit,
                ToolExecutionUpdateEvent(
                    tool_call_id=prepared["tool_call"].tool_call_id,
                    tool_name=prepared["tool_call"].name,
                    args=prepared["tool_call"].args,
                    partial_result=partial_result,
                ),
            )
        )

    try:
        result = await prepared["tool"].execute(
            prepared["tool_call"].tool_call_id,
            prepared["args"],
            signal,
            _on_partial_result,
        )
        accepting_updates = False
        await asyncio.gather(*update_events)
        return ExecutedToolCallOutcome(result=result, is_error=False)
    except Exception as exc:
        accepting_updates = False
        await asyncio.gather(*update_events)
        return ExecutedToolCallOutcome(
            result=create_error_tool_result(str(exc)),
            is_error=True,
        )


# ---------------------------------------------------------------------------
# Phase 3: Finalize
# ---------------------------------------------------------------------------


async def finalize_executed_tool_call(
    current_context: AgentContext,
    assistant_message: AssistantMessage,
    prepared: PreparedToolCall,
    executed: ExecutedToolCallOutcome,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
) -> FinalizedToolCallOutcome:
    """应用 after_tool_call 钩子，逐字段覆盖结果。"""
    result = executed["result"]
    is_error = executed["is_error"]

    if config.after_tool_call is not None:
        try:
            after_ctx = AfterToolCallContext(
                assistant_message=assistant_message,
                tool_call=prepared["tool_call"],
                args=prepared["args"],
                result=result,
                is_error=is_error,
                context=current_context,
            )
            after_result = await config.after_tool_call(after_ctx, signal)
            if after_result is not None:
                result = result.model_copy(
                    update={
                        "content": after_result.content
                        if after_result.content is not None
                        else result.content,
                        "details": after_result.details
                        if after_result.details is not None
                        else result.details,
                        "usage": after_result.usage
                        if after_result.usage is not None
                        else result.usage,
                        "terminate": after_result.terminate
                        if after_result.terminate is not None
                        else result.terminate,
                    }
                )
                is_error = (
                    after_result.is_error
                    if after_result.is_error is not None
                    else is_error
                )
        except Exception as exc:
            result = create_error_tool_result(str(exc))
            is_error = True

    return FinalizedToolCallOutcome(
        tool_call=prepared["tool_call"],
        result=result,
        is_error=is_error,
    )


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def create_error_tool_result(message: str) -> AgentToolResult:
    """创建错误工具结果。"""
    return AgentToolResult(
        content=[TextContent(text=message)],
        details={},
    )


def create_tool_result_message(
    finalized: FinalizedToolCallOutcome,
) -> ToolResultMessage:
    """将 FinalizedToolCallOutcome 转为 ToolResultMessage。"""
    result = finalized["result"]
    msg = ToolResultMessage(
        role="toolResult",
        tool_call_id=finalized["tool_call"].tool_call_id,
        tool_name=finalized["tool_call"].name,
        content=result.content or [],
        details=result.details,
        usage=result.usage,
        is_error=finalized["is_error"],
        timestamp=int(time.time() * 1000),
    )
    if result.added_tool_names:
        msg.added_tool_names = result.added_tool_names
    return msg


async def emit_tool_execution_end(
    finalized: FinalizedToolCallOutcome, emit: AgentEventSink
) -> None:
    """发射 tool_execution_end 事件。"""
    await _emit(
        emit,
        ToolExecutionEndEvent(
            tool_call_id=finalized["tool_call"].tool_call_id,
            tool_name=finalized["tool_call"].name,
            result=finalized["result"],
            is_error=finalized["is_error"],
        ),
    )


async def emit_tool_result_message(
    tool_result_msg: ToolResultMessage, emit: AgentEventSink
) -> None:
    """发射 toolResult 消息的 message_start / message_end 事件。"""
    await _emit(emit, MessageStartEvent(message=tool_result_msg))
    await _emit(emit, MessageEndEvent(message=tool_result_msg))


def should_terminate_tool_batch(
    finalized_calls: list[FinalizedToolCallOutcome],
) -> bool:
    """所有工具调用一致要求终止时返回 True。"""
    return len(finalized_calls) > 0 and all(
        fc["result"].terminate for fc in finalized_calls
    )


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    """同步或异步发射事件。"""
    result = emit(event)
    if result is not None:
        await result


def _find_tool(context: AgentContext, name: str) -> AgentTool | None:
    """按名称查找工具。"""
    if context.tools is None:
        return None
    for tool in context.tools:
        if tool.name == name:
            return tool
    return None


def _make_async_return(
    value: FinalizedToolCallOutcome,
) -> Callable[[], Awaitable[FinalizedToolCallOutcome]]:
    """将立即结果包装为异步 callable（用于 parallel 模式的 asyncio.gather）。"""

    async def _inner() -> FinalizedToolCallOutcome:
        return value

    return _inner
