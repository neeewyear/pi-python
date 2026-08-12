"""Agent 循环核心逻辑（对应 ``agent-loop.ts`` 主循环 + 流式响应）。

- ``run_loop``：外层 while（follow-up）+ 内层 while（tool calls + steering）
- ``stream_assistant_response``：调用 LLM 并发射事件，返回 AssistantMessage
"""

from __future__ import annotations

import asyncio

from .agent_loop_tools import (
    AgentEventSink,
    ExecutedToolCallBatch,
    FinalizedToolCallOutcome,
    create_error_tool_result,
    create_tool_result_message,
    emit_tool_execution_end,
    emit_tool_result_message,
    execute_tool_calls,
)
from .cancellation import CancellationToken
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentMessage,
    AssistantMessage,
    Context,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    PrepareNextTurnContext,
    ShouldStopAfterTurnContext,
    StreamFn,
    ToolCallContent,
    ToolExecutionStartEvent,
    ToolResultMessage,
    TurnEndEvent,
    TurnStartEvent,
)

# ---------------------------------------------------------------------------
# run_loop（主循环）
# ---------------------------------------------------------------------------


async def run_loop(
    initial_context: AgentContext,
    new_messages: list[AgentMessage],
    initial_config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> None:
    """agent 主循环（对应 TS ``runLoop``）。

    外层 while 等待 follow-up 消息；内层 while 处理 tool calls + steering。
    """
    current_context = initial_context
    config = initial_config
    first_turn = True

    pending_messages: list[AgentMessage] = []
    if config.get_steering_messages is not None:
        steering = await config.get_steering_messages()
        if steering:
            pending_messages = list(steering)

    # 外层循环：follow-up 续跑
    while True:
        has_more_tool_calls = True

        # 内层循环：tool calls + steering
        while has_more_tool_calls or len(pending_messages) > 0:
            if not first_turn:
                await _emit(emit, TurnStartEvent())
            else:
                first_turn = False

            # 处理待注入消息
            if pending_messages:
                for message in pending_messages:
                    await _emit(emit, MessageStartEvent(message=message))
                    await _emit(emit, MessageEndEvent(message=message))
                    current_context.messages.append(message)
                    new_messages.append(message)
                pending_messages = []

            # 流式调用 LLM
            message = await stream_assistant_response(
                current_context,
                config,
                signal,
                emit,
                stream_function,
            )
            new_messages.append(message)

            if message.stop_reason in ("error", "aborted"):
                await _emit(emit, TurnEndEvent(message=message, tool_results=[]))
                await _emit(emit, AgentEndEvent(messages=list(new_messages)))
                return

            # 工具调用
            tool_calls = [c for c in message.content if isinstance(c, ToolCallContent)]
            tool_results: list[ToolResultMessage] = []
            has_more_tool_calls = False

            if tool_calls:
                if message.stop_reason == "length":
                    executed_batch = await _fail_tool_calls_from_truncated_message(
                        tool_calls, emit
                    )
                else:
                    executed_batch = await execute_tool_calls(
                        current_context,
                        message,
                        config,
                        signal,
                        emit,
                    )
                tool_results = executed_batch["messages"]
                has_more_tool_calls = not executed_batch["terminate"]

                for result in tool_results:
                    current_context.messages.append(result)
                    new_messages.append(result)

            await _emit(emit, TurnEndEvent(message=message, tool_results=tool_results))

            # prepare_next_turn
            if config.prepare_next_turn is not None:
                next_ctx = PrepareNextTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=list(new_messages),
                )
                snapshot = await config.prepare_next_turn(next_ctx)  # type: ignore[misc]
                if snapshot is not None:
                    if snapshot.context is not None:
                        current_context = snapshot.context
                    config = _rebuild_config(config, snapshot)

            # should_stop_after_turn
            if config.should_stop_after_turn is not None:
                stop_ctx = ShouldStopAfterTurnContext(
                    message=message,
                    tool_results=tool_results,
                    context=current_context,
                    new_messages=list(new_messages),
                )
                if await config.should_stop_after_turn(stop_ctx):  # type: ignore[misc]
                    await _emit(emit, AgentEndEvent(messages=list(new_messages)))
                    return

            if config.get_steering_messages is not None:
                steering = await config.get_steering_messages()
                if steering:
                    pending_messages = list(steering)

        # 内层循环结束，检查 follow-up
        if config.get_follow_up_messages is not None:
            follow_up = await config.get_follow_up_messages()
            if follow_up:
                pending_messages = list(follow_up)
                continue

        break

    await _emit(emit, AgentEndEvent(messages=list(new_messages)))


def _rebuild_config(config: AgentLoopConfig, snapshot: object) -> AgentLoopConfig:
    """从 prepare_next_turn 快照重建配置。"""
    # 使用 model_dump 合并，snapshot 有值的字段覆盖
    snap_dict: dict[str, object] = {}
    if hasattr(snapshot, "model") and snapshot.model is not None:
        snap_dict["model"] = snapshot.model
    return AgentLoopConfig(
        model=snap_dict.get("model", config.model),
        convert_to_llm=config.convert_to_llm,
        transform_context=config.transform_context,
        get_api_key=config.get_api_key,
        should_stop_after_turn=config.should_stop_after_turn,
        prepare_next_turn=config.prepare_next_turn,
        get_steering_messages=config.get_steering_messages,
        get_follow_up_messages=config.get_follow_up_messages,
        tool_execution=config.tool_execution,
        before_tool_call=config.before_tool_call,
        after_tool_call=config.after_tool_call,
    )


# ---------------------------------------------------------------------------
# stream_assistant_response
# ---------------------------------------------------------------------------


async def stream_assistant_response(
    context: AgentContext,
    config: AgentLoopConfig,
    signal: CancellationToken | None,
    emit: AgentEventSink,
    stream_function: StreamFn,
) -> AssistantMessage:
    """流式调用 LLM 并发射事件（对应 TS ``streamAssistantResponse``）。

    这是 AgentMessage[] → Message[] 转换的边界点。
    """
    # 上下文变换
    messages = list(context.messages)
    if config.transform_context is not None:
        messages = await config.transform_context(messages, signal)

    # AgentMessage[] → Message[]
    llm_result = config.convert_to_llm(messages)
    if asyncio.iscoroutine(llm_result):
        llm_messages = await llm_result
    else:
        llm_messages = llm_result

    # 构建 LLM 上下文
    llm_context = Context(
        system_prompt=context.system_prompt,
        messages=llm_messages,
        tools=context.tools,
    )

    # 解析 API key
    resolved_api_key: str | None = None
    if config.get_api_key is not None:
        result = config.get_api_key(config.model.provider)
        if asyncio.iscoroutine(result):
            resolved_api_key = await result
        else:
            resolved_api_key = result  # type: ignore[assignment]

    # 调用 streamFn
    response = stream_function(config.model, llm_context)

    partial_message: AssistantMessage | None = None
    added_partial = False
    last_snapshot: AssistantMessage | None = None
    final_message: AssistantMessage | None = None

    async for event in response:
        if event.type == "message_snapshot":
            last_snapshot = event.message
            if not added_partial:
                partial_message = last_snapshot
                context.messages.append(partial_message)
                added_partial = True
                await _emit(emit, MessageStartEvent(message=partial_message))

        elif event.type in (
            "text_delta",
            "thinking_delta",
            "tool_call_start",
            "tool_call_update",
            "tool_call_end",
        ):
            if partial_message is not None:
                partial_message = last_snapshot if last_snapshot else partial_message
                if context.messages:
                    context.messages[-1] = partial_message
                await _emit(
                    emit,
                    MessageUpdateEvent(
                        message=partial_message,
                        assistant_message_event=event,
                    ),
                )

        elif event.type in ("stream_end", "error", "aborted"):
            if event.type == "error":
                final_message = getattr(event, "error", None)
            if final_message is None:
                final_message = last_snapshot or partial_message
            if final_message is None:
                raise RuntimeError("No final message from stream")
            # 不 break：让 async for 自然消费完生成器（StopAsyncIteration）。
            # 若提前退出，生成器会滞留在事件循环的 pending asyncgen 列表，
            # 关闭时被注入 CancelledError 并再次 yield，触发
            # "async generator ignored GeneratorExit" 异常。

    if final_message is None:
        final_message = last_snapshot or partial_message
    if final_message is None:
        raise RuntimeError("No final message from stream")

    # 将最终消息写入 context
    if added_partial and context.messages:
        context.messages[-1] = final_message
    else:
        context.messages.append(final_message)

    if not added_partial:
        await _emit(emit, MessageStartEvent(message=final_message))
    await _emit(emit, MessageEndEvent(message=final_message))

    return final_message


# ---------------------------------------------------------------------------
# _fail_tool_calls_from_truncated_message
# ---------------------------------------------------------------------------


async def _fail_tool_calls_from_truncated_message(
    tool_calls: list[ToolCallContent],
    emit: AgentEventSink,
) -> ExecutedToolCallBatch:
    """将因 token 限制截断的工具调用全部标记为失败（对应 TS ``failToolCallsFromTruncatedMessage``）。"""
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

        finalized = FinalizedToolCallOutcome(
            tool_call=tool_call,
            result=create_error_tool_result(
                f'Tool call "{tool_call.name}" was not executed: the response hit the output '
                f"token limit, so its arguments may be truncated. Re-issue the tool call "
                f"with complete arguments.",
            ),
            is_error=True,
        )
        await emit_tool_execution_end(finalized, emit)
        tool_result_msg = create_tool_result_message(finalized)
        await emit_tool_result_message(tool_result_msg, emit)
        messages.append(tool_result_msg)

    return ExecutedToolCallBatch(messages=messages, terminate=False)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


async def _emit(emit: AgentEventSink, event: AgentEvent) -> None:
    """同步或异步发射事件。"""
    result = emit(event)
    if result is not None:
        await result
