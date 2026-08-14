"""Agent 高层有状态封装。

提供：
- 消息队列：``steer`` / ``follow_up`` / ``clear_*_queue``
- 运行控制：``prompt`` / ``continue`` / ``abort`` / ``wait_for_idle``
- 事件订阅：``subscribe``
- 状态访问：``state`` / ``steering_mode`` / ``follow_up_mode``
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import overload

from .agent_lifecycle import (
    EMPTY_USAGE,
    ActiveRun,
    AgentOptions,
    MutableAgentState,
    create_mutable_agent_state,
    default_convert_to_llm,
)
from .agent_loop import run_agent_loop, run_agent_loop_continue
from .agent_queue import PendingMessageQueue
from .cancellation import CancellationToken
from .stream_fn import get_default_stream_fn
from .types import (
    AgentContext,
    AgentEndEvent,
    AgentEvent,
    AgentLoopConfig,
    AgentLoopTurnUpdate,
    AgentMessage,
    AssistantMessage,
    ImageContent,
    Message,
    MessageEndEvent,
    MessageStartEvent,
    PrepareNextTurnContext,
    QueueMode,
    StreamFn,
    TextContent,
    TurnEndEvent,
    UserMessage,
)


class Agent:
    """有状态 Agent 高层封装。

    包装低层 agent_loop，维护对话 transcript，发射生命周期事件，
    暴露 steering 和 follow-up 消息队列 API。
    """

    def __init__(self, options: AgentOptions) -> None:
        # 状态
        self._state: MutableAgentState = create_mutable_agent_state(
            options.initial_state
        )

        # 事件监听器
        self._listeners: set[
            Callable[[AgentEvent, CancellationToken], Awaitable[None] | None]
        ] = set()

        # 消息队列
        self._steering_queue: PendingMessageQueue = PendingMessageQueue(
            options.steering_mode,
        )
        self._follow_up_queue: PendingMessageQueue = PendingMessageQueue(
            options.follow_up_mode,
        )

        # 钩子 & 配置
        self.convert_to_llm: Callable[
            [list[AgentMessage]], Awaitable[list[Message]] | list[Message]
        ] = options.convert_to_llm or default_convert_to_llm
        self.transform_context: (
            Callable[
                [list[AgentMessage], CancellationToken | None],
                Awaitable[list[AgentMessage]],
            ]
            | None
        ) = options.transform_context
        self._stream_fn: StreamFn | None = options.stream_fn  # type: ignore[assignment]
        self.get_api_key: Callable[[str], Awaitable[str | None] | str | None] | None = (
            options.get_api_key
        )
        self.before_tool_call = options.before_tool_call
        self.after_tool_call = options.after_tool_call
        self.should_stop_after_turn = options.should_stop_after_turn
        self.prepare_next_turn = options.prepare_next_turn
        self.prepare_next_turn_with_context = options.prepare_next_turn_with_context
        self.session_id: str | None = options.session_id
        self.transport = options.transport
        self.max_retry_delay_ms: int | None = options.max_retry_delay_ms
        self.tool_execution = options.tool_execution

        # 活跃运行
        self._active_run: ActiveRun | None = None

    # ------------------------------------------------------------------
    # 事件订阅
    # ------------------------------------------------------------------

    def subscribe(
        self,
        listener: Callable[[AgentEvent, CancellationToken], Awaitable[None] | None],
    ) -> Callable[[], None]:
        """订阅 agent 生命周期事件，返回取消订阅函数。"""
        self._listeners.add(listener)

        def _unsubscribe() -> None:
            self._listeners.discard(listener)

        return _unsubscribe

    # ------------------------------------------------------------------
    # 状态访问
    # ------------------------------------------------------------------

    @property
    def state(self) -> MutableAgentState:
        """当前 agent 状态。"""
        return self._state

    @property
    def steering_mode(self) -> QueueMode:
        """steering 队列排空模式。"""
        return self._steering_queue.mode

    @steering_mode.setter
    def steering_mode(self, mode: QueueMode) -> None:
        self._steering_queue.mode = mode

    @property
    def follow_up_mode(self) -> QueueMode:
        """follow-up 队列排空模式。"""
        return self._follow_up_queue.mode

    @follow_up_mode.setter
    def follow_up_mode(self, mode: QueueMode) -> None:
        self._follow_up_queue.mode = mode

    # ------------------------------------------------------------------
    # 消息队列
    # ------------------------------------------------------------------

    def steer(self, message: AgentMessage) -> None:
        """入队 steering 消息：在当前 assistant 回合结束后注入。"""
        self._steering_queue.enqueue(message)

    def follow_up(self, message: AgentMessage) -> None:
        """入队 follow-up 消息：在 agent 即将停止时注入。"""
        self._follow_up_queue.enqueue(message)

    def clear_steering_queue(self) -> None:
        """清空 steering 队列。"""
        self._steering_queue.clear()

    def clear_follow_up_queue(self) -> None:
        """清空 follow-up 队列。"""
        self._follow_up_queue.clear()

    def clear_all_queues(self) -> None:
        """清空全部队列。"""
        self.clear_steering_queue()
        self.clear_follow_up_queue()
    def has_queued_messages(self) -> bool:
        """是否有待处理的消息。"""
        return self._steering_queue.has_items() or self._follow_up_queue.has_items()

    # ------------------------------------------------------------------
    # 运行控制
    # ------------------------------------------------------------------

    @property
    def signal(self) -> CancellationToken | None:
        """当前运行的取消令牌。"""
        if self._active_run is None:
            return None
        return self._active_run.cancellation_token

    def abort(self) -> None:
        """中止当前运行。"""
        if self._active_run is not None:
            self._active_run.cancellation_token.cancel()

    async def wait_for_idle(self) -> None:
        """等待当前运行完成。"""
        if self._active_run is not None:
            await self._active_run.completion_event.wait()

    def reset(self) -> None:
        """重置状态和队列。"""
        self._state.messages = []
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        self._state.error_message = None
        self.clear_follow_up_queue()
        self.clear_steering_queue()

    # ------------------------------------------------------------------
    # 核心 API：prompt
    # ------------------------------------------------------------------

    @overload
    async def prompt(
        self, input: str, images: list[ImageContent] | None = None
    ) -> None: ...

    @overload
    async def prompt(self, input: AgentMessage) -> None: ...

    @overload
    async def prompt(self, input: list[AgentMessage]) -> None: ...

    async def prompt(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> None:
        """启动新 prompt。

        支持三种输入形式：
        - ``str``：纯文本，自动包装为 ``UserMessage``
        - ``AgentMessage``：单条消息
        - ``list[AgentMessage]``：多条消息
        """
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing a prompt. "
                "Use steer() or follow_up() to queue messages, or wait for completion.",
            )
        messages = self._normalize_prompt_input(input, images)
        await self._run_prompt_messages(messages)

    # ------------------------------------------------------------------
    # 核心 API：continue
    # ------------------------------------------------------------------

    async def continue_(self) -> None:
        """从当前 transcript 继续。

        最后一条消息必须是 user 或 toolResult。
        """
        if self._active_run is not None:
            raise RuntimeError(
                "Agent is already processing. Wait for completion before continuing.",
            )

        last_message = self._state.messages[-1] if self._state.messages else None
        if last_message is None:
            raise RuntimeError("No messages to continue from")

        if last_message.role == "assistant":
            # 尝试排空 steering 队列
            queued_steering = self._steering_queue.drain()
            if queued_steering:
                await self._run_prompt_messages(
                    queued_steering, skip_initial_steering_poll=True
                )
                return

            # 尝试排空 follow-up 队列
            queued_follow_ups = self._follow_up_queue.drain()
            if queued_follow_ups:
                await self._run_prompt_messages(queued_follow_ups)
                return

            raise RuntimeError("Cannot continue from message role: assistant")

        await self._run_continuation()

    # ------------------------------------------------------------------
    # 内部：prompt 输入标准化
    # ------------------------------------------------------------------

    def _normalize_prompt_input(
        self,
        input: str | AgentMessage | list[AgentMessage],
        images: list[ImageContent] | None = None,
    ) -> list[AgentMessage]:
        """标准化 prompt 输入为消息列表。"""
        if isinstance(input, list):
            return input

        if isinstance(input, str):
            content: list[TextContent | ImageContent] = [TextContent(text=input)]
            if images:
                content.extend(images)
            return [UserMessage(content=content, timestamp=int(time.time() * 1000))]

        # 单条消息
        return [input]

    # ------------------------------------------------------------------
    # 内部：运行 prompt 消息
    # ------------------------------------------------------------------

    async def _run_prompt_messages(
        self,
        messages: list[AgentMessage],
        *,
        skip_initial_steering_poll: bool = False,
        ) -> None:
        """执行 prompt 消息。"""
        skip_flag = skip_initial_steering_poll

        async def _executor(signal: CancellationToken) -> None:
            stream_fn = self._resolve_stream_fn()
            await run_agent_loop(
                messages,
                self._create_context_snapshot(),
                self._create_loop_config(skip_initial_steering_poll=skip_flag),
                lambda event: self._process_events(event),
                signal,
                stream_fn,
            )

        await self._run_with_lifecycle(_executor)

    # ------------------------------------------------------------------
    # 内部：运行 continue
    # ------------------------------------------------------------------

    async def _run_continuation(self) -> None:
        """执行 continue。"""

        async def _executor(signal: CancellationToken) -> None:
            stream_fn = self._resolve_stream_fn()
            await run_agent_loop_continue(
                self._create_context_snapshot(),
                self._create_loop_config(),
                lambda event: self._process_events(event),
                signal,
                stream_fn,
            )

        await self._run_with_lifecycle(_executor)

    # ------------------------------------------------------------------
    # 内部：上下文快照
    # ------------------------------------------------------------------

    def _create_context_snapshot(self) -> AgentContext:
        """创建上下文快照。"""
        return AgentContext(
            system_prompt=self._state.system_prompt,
            messages=list(self._state.messages),
            tools=list(self._state.tools),
        )

    # ------------------------------------------------------------------
    # 内部：循环配置
    # ------------------------------------------------------------------

    def _create_loop_config(
        self,
        *,
        skip_initial_steering_poll: bool = False,
    ) -> AgentLoopConfig:
        """创建低层循环配置。"""
        skip_flag = skip_initial_steering_poll

        # prepare_next_turn 适配：优先使用 with_context 版本
        _prepare_next_turn = self.prepare_next_turn
        _prepare_next_turn_with_ctx = self.prepare_next_turn_with_context

        async def _prepare_next_turn_wrapper(
            context: PrepareNextTurnContext,
        ) -> AgentLoopTurnUpdate | None:
            if _prepare_next_turn_with_ctx is not None:
                result = _prepare_next_turn_with_ctx(context)
                if asyncio.iscoroutine(result):
                    return await result  # type: ignore[no-any-return]
                return result  # type: ignore[return-value]
            if _prepare_next_turn is not None:
                result = _prepare_next_turn()
                if asyncio.iscoroutine(result):
                    return await result  # type: ignore[no-any-return]
                return result  # type: ignore[return-value]
            return None

        _should_stop = self.should_stop_after_turn

        async def _should_stop_wrapper(ctx: object) -> bool:
            if _should_stop is None:
                return False
            result = _should_stop(ctx)  # type: ignore[arg-type]
            if asyncio.iscoroutine(result):
                return await result  # type: ignore[no-any-return]
            return result  # type: ignore[return-value]

        async def _get_steering_messages() -> list[AgentMessage]:
            nonlocal skip_flag
            if skip_flag:
                skip_flag = False
                return []
            return self._steering_queue.drain()

        async def _get_follow_up_messages() -> list[AgentMessage]:
            return self._follow_up_queue.drain()

        return AgentLoopConfig(
            model=self._state.model,
            convert_to_llm=self.convert_to_llm,
            transform_context=self.transform_context,
            get_api_key=self.get_api_key,
            should_stop_after_turn=_should_stop_wrapper
            if _should_stop is not None
            else None,
            prepare_next_turn=(
                _prepare_next_turn_wrapper
                if (
                    _prepare_next_turn is not None
                    or _prepare_next_turn_with_ctx is not None
                )
                else None
            ),
            get_steering_messages=_get_steering_messages,
            get_follow_up_messages=_get_follow_up_messages,
            tool_execution=self.tool_execution,
            before_tool_call=self.before_tool_call,
            after_tool_call=self.after_tool_call,
        )

    # ------------------------------------------------------------------
    # 内部：生命周期包装
    # ------------------------------------------------------------------

    async def _run_with_lifecycle(
        self,
        executor: Callable[[CancellationToken], Awaitable[None]],
    ) -> None:
        """生命周期包装。"""
        if self._active_run is not None:
            raise RuntimeError("Agent is already processing.")

        active_run = ActiveRun()
        self._active_run = active_run

        self._state.is_streaming = True
        self._state.streaming_message = None
        self._state.error_message = None

        try:
            await executor(active_run.cancellation_token)
        except Exception as exc:
            await self._handle_run_failure(exc, active_run.cancellation_token.aborted)
        finally:
            self._finish_run()

    # ------------------------------------------------------------------
    # 内部：失败处理
    # ------------------------------------------------------------------

    async def _handle_run_failure(self, error: Exception, aborted: bool) -> None:
        """失败处理：注入错误 assistant 消息。"""
        error_message_text = str(error) if str(error) else "Unknown error"
        failure_message = AssistantMessage(
            content=[TextContent(text="")],
            api=self._state.model.api,
            provider=self._state.model.provider,
            model=self._state.model.model_id,
            usage=EMPTY_USAGE,
            stop_reason="aborted" if aborted else "error",
            error_message=error_message_text,
            timestamp=int(time.time() * 1000),
        )

        await self._process_events(
            MessageStartEvent(type="message_start", message=failure_message)
        )
        await self._process_events(
            MessageEndEvent(type="message_end", message=failure_message)
        )
        await self._process_events(
            TurnEndEvent(type="turn_end", message=failure_message, tool_results=[])
        )
        await self._process_events(
            AgentEndEvent(type="agent_end", messages=[failure_message])
        )

    # ------------------------------------------------------------------
    # 内部：结束运行
    # ------------------------------------------------------------------

    def _finish_run(self) -> None:
        """结束运行。"""
        self._state.is_streaming = False
        self._state.streaming_message = None
        self._state.pending_tool_calls = set()
        if self._active_run is not None:
            self._active_run.completion_event.set()
            self._active_run = None

    # ------------------------------------------------------------------
    # 内部：事件处理
    # ------------------------------------------------------------------

    async def _process_events(self, event: AgentEvent) -> None:
        """处理循环事件并分发给监听器。"""      
        event_type = (
            event["type"] if isinstance(event, dict) else getattr(event, "type", None)
        )

        if event_type == "message_start" or event_type == "message_update":
            msg = (
                event["message"]
                if isinstance(event, dict)
                else getattr(event, "message", None)
            )
            self._state.streaming_message = msg

        elif event_type == "message_end":
            msg = (
                event["message"]
                if isinstance(event, dict)
                else getattr(event, "message", None)
            )
            self._state.streaming_message = None
            if msg is not None:
                self._state.messages.append(msg)

        elif event_type == "tool_execution_start":
            tc_id = (
                event["tool_call_id"]
                if isinstance(event, dict)
                else getattr(event, "tool_call_id", "")
            )
            self._state.pending_tool_calls.add(tc_id)

        elif event_type == "tool_execution_end":
            tc_id = (
                event["tool_call_id"]
                if isinstance(event, dict)
                else getattr(event, "tool_call_id", "")
            )
            self._state.pending_tool_calls.discard(tc_id)

        elif event_type == "turn_end":
            msg = (
                event["message"]
                if isinstance(event, dict)
                else getattr(event, "message", None)
            )
            if msg is not None and hasattr(msg, "role") and msg.role == "assistant":
                err_msg = getattr(msg, "error_message", None)
                if err_msg:
                    self._state.error_message = err_msg

        elif event_type == "agent_end":
            self._state.streaming_message = None

        # 分发给监听器
        signal = (
            self._active_run.cancellation_token
            if self._active_run is not None
            else None
        )
        if signal is None:
            raise RuntimeError("Agent listener invoked outside active run")

        for listener in list(self._listeners):
            result = listener(event, signal)
            if result is not None:
                await result

    # ------------------------------------------------------------------
    # 内部：解析 stream_fn
    # ------------------------------------------------------------------

    def _resolve_stream_fn(self) -> StreamFn:
        """解析 stream_fn：优先使用构造时传入的，否则使用全局默认。"""
        if self._stream_fn is not None:
            return self._stream_fn
        return get_default_stream_fn()
