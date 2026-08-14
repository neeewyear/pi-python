"""远程会话。

管理远程会话的生命周期、状态和操作。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

from .transcript import (
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

# ============================================================================
# Types
# ============================================================================

RemoteSessionOperation: TypeAlias = Literal[
    "open", "create", "submit", "abort", "setModel", "setThinking", "reconnect"
]
"""远程会话操作类型。"""

RemoteSessionLifecycle: TypeAlias = dict[str, Any]
"""远程会话生命周期状态。"""


class RemoteSessionState:
    """远程会话状态。"""

    def __init__(
        self,
        lifecycle: RemoteSessionLifecycle,
        snapshot: dict[str, Any] | None = None,
        transcript: list[dict[str, Any]] | None = None,
    ) -> None:
        self.lifecycle = lifecycle
        self.snapshot = snapshot
        self.transcript = transcript or []


class CreateRemoteSessionOptions:
    """创建远程会话选项。"""

    def __init__(
        self,
        cwd: str,
        model: Any | None = None,
        thinking_level: Any | None = None,
    ) -> None:
        self.cwd = cwd
        self.model = model
        self.thinking_level = thinking_level


class RemoteSessionOptions:
    """远程会话选项。"""

    def __init__(
        self,
        on_listener_error: Callable[[Exception], None] | None = None,
    ) -> None:
        self.on_listener_error = on_listener_error


class _RemoteSessionDisposedError(Exception):
    """远程会话已释放错误。"""

    def __init__(self) -> None:
        super().__init__("Remote session is disposed")
        self.name = "RemoteSessionDisposedError"


async def _settle_remote_session_disposal(
    cleanup: list[Awaitable[None]],
) -> None:
    """处理远程会话清理结果。"""
    results = await asyncio.gather(*cleanup, return_exceptions=True)
    errors = [
        r
        for r in results
        if isinstance(r, BaseException)
        and not isinstance(r, _RemoteSessionDisposedError)
    ]
    if len(errors) == 1:
        raise errors[0]
    if len(errors) > 1:
        raise Exception("Failed to dispose remote session")


# ============================================================================
# RemoteSession
# ============================================================================


class RemoteSession:
    """远程会话。

    管理与远程 agent 服务的连接和会话生命周期。
    """

    def __init__(
        self,
        client: Any,
        options: RemoteSessionOptions | None = None,
    ) -> None:
        self._client = client
        self._on_listener_error = options.on_listener_error if options else None
        self._lifecycle: RemoteSessionLifecycle = {"status": "unbound"}
        self._handle: Any = None
        self._transcript: TranscriptState | None = None
        self._unsubscribe_snapshot: Callable[[], None] | None = None
        self._unsubscribe_events: Callable[[], None] | None = None
        self._listeners: set[Callable[[RemoteSessionState], None]] = set()
        self._pending_attachment_operations: set[Awaitable[None]] = set()
        self._active_operation_states: set[RemoteSessionLifecycle] = set()
        self._dispose_promise: asyncio.Task[None] | None = None
        self._resolve_dispose_signal: Callable[[], None] = lambda: None
        self._dispose_event = asyncio.Event()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def id(self) -> str | None:
        """获取会话 ID。"""
        return self._handle.id if self._handle else None

    @property
    def state(self) -> RemoteSessionState:
        """获取当前状态。"""
        return RemoteSessionState(
            lifecycle=self._lifecycle,
            snapshot=self._transcript.snapshot if self._transcript else None,
            transcript=self._transcript
            and list(select_transcript(self._transcript))
            or [],
        )

    @property
    def snapshot(self) -> dict[str, Any] | None:
        """获取会话快照。"""
        return self._transcript.snapshot if self._transcript else None

    @property
    def phase(self) -> Any | None:
        """获取会话阶段。"""
        snap = self.snapshot
        return snap.get("phase") if snap else None

    @property
    def operation(self) -> RemoteSessionOperation | None:
        """获取当前操作。"""
        if self._lifecycle.get("status") == "busy":
            return self._lifecycle.get("operation")  # type: ignore[return-value]
        return None

    @property
    def models(self) -> list[Any]:
        """获取模型列表。"""
        snap = self._client.snapshot if hasattr(self._client, "snapshot") else None
        return snap.get("models", []) if snap else []

    @property
    def sessions(self) -> list[Any]:
        """获取会话列表。"""
        snap = self._client.snapshot if hasattr(self._client, "snapshot") else None
        return snap.get("sessions", []) if snap else []

    @property
    def connection_state(self) -> Any:
        """获取连接状态。"""
        return (
            self._client.connection_state
            if hasattr(self._client, "connection_state")
            else None
        )

    @property
    def disposed(self) -> bool:
        """是否已释放。"""
        return self._lifecycle.get("status") == "disposed"

    # ------------------------------------------------------------------
    # Public Methods
    # ------------------------------------------------------------------

    def subscribe(
        self, listener: Callable[[RemoteSessionState], None]
    ) -> Callable[[], None]:
        """订阅状态变更。"""
        self._assert_not_disposed()
        self._listeners.add(listener)
        self._call_listener(listener, self.state)
        return lambda: self._listeners.discard(listener)

    def on_connection_state_change(
        self, listener: Callable[[Any], None]
    ) -> Callable[[], None]:
        """订阅连接状态变更。"""
        self._assert_not_disposed()
        if hasattr(self._client, "on_connection_state_change"):
            return self._client.on_connection_state_change(listener)  # type: ignore[no-any-return]
        return lambda: None

    @classmethod
    async def open(
        cls,
        client: Any,
        session_id: str,
        options: RemoteSessionOptions | None = None,
    ) -> RemoteSession:
        """打开现有会话。"""
        session = cls(client, options)
        try:
            await session._open(session_id)
            return session
        except Exception:
            await session.dispose()
            raise

    async def _open(self, session_id: str) -> None:
        """打开会话（内部）。"""
        if (
            self._handle
            and self._handle.id == session_id
            and self._lifecycle.get("status") == "ready"
        ):
            return
        await self._replace(
            "open",
            lambda: self._client.acquire_session(session_id, {"mode": "exclusive"}),
        )

    @classmethod
    async def create(
        cls,
        client: Any,
        create_options: CreateRemoteSessionOptions,
        options: RemoteSessionOptions | None = None,
    ) -> RemoteSession:
        """创建新会话。"""
        session = cls(client, options)
        try:
            await session._create(create_options)
            return session
        except Exception:
            await session.dispose()
            raise

    async def _create(self, options: CreateRemoteSessionOptions) -> None:
        """创建会话（内部）。"""
        await self._replace("create", lambda: self._client.create_session(options))

    async def submit(self, text: str) -> None:
        """提交输入。"""
        normalized = text.strip()
        if not normalized:
            return
        self._assert_available()
        handle = self._require_handle()
        phase = self.phase
        if phase not in ("idle", "turn"):
            raise RuntimeError(
                f"Session cannot accept input during {phase or 'unknown'} phase"
            )
        if phase == "idle":
            await self._run_operation("submit", lambda: handle.prompt(normalized))
        else:
            await self._run_operation("submit", lambda: handle.steer(normalized))

    async def abort(self) -> None:
        """中止操作。"""
        preempting_submit = (
            self._lifecycle.get("status") == "busy"
            and self._lifecycle.get("operation") == "submit"
        )
        if preempting_submit:
            self._assert_not_disposed()
        else:
            self._assert_available()
        handle = self._require_handle()
        if self.phase == "idle" and not preempting_submit:
            return
        await self._run_operation(
            "abort", lambda: handle.abort(), preempt=preempting_submit
        )

    async def set_model(self, model: Any) -> None:
        """设置模型。"""
        await self._run_idle_operation(
            "setModel", "change model", lambda: self._require_handle().set_model(model)
        )

    async def set_thinking(self, thinking_level: Any) -> None:
        """设置思考级别。"""
        await self._run_idle_operation(
            "setThinking",
            "change thinking level",
            lambda: self._require_handle().set_thinking(thinking_level),
        )

    async def reconnect(self) -> None:
        """重新连接。"""
        self._assert_available()
        session_id = self._require_handle().id

        async def _do_reconnect() -> None:
            await self._track_attachment_operation(self._do_reconnect_inner(session_id))

        await self._run_operation("reconnect", _do_reconnect)

    async def _do_reconnect_inner(self, session_id: str) -> None:
        """重新连接内部逻辑。"""
        await self._client.reconnect()
        handle = await self._client.acquire_session(session_id, {"mode": "exclusive"})
        await self._assert_not_disposed_after_await(handle)
        self._bind(handle)

    async def dispose(self) -> None:
        """释放会话。"""
        if self._dispose_promise is not None:
            await self._dispose_promise
            return
        handle = self._handle
        self._lifecycle = {"status": "disposed"}
        self._dispose_event.set()
        self._clear_subscriptions()
        self._handle = None
        self._transcript = None
        cleanup: list[Awaitable[None]] = list(self._pending_attachment_operations)
        if handle:
            cleanup.append(handle.dispose())
        self._dispose_promise = asyncio.ensure_future(
            _settle_remote_session_disposal(cleanup)
        )
        self._notify()
        self._listeners.clear()
        await self._dispose_promise

    # ------------------------------------------------------------------
    # Internal Methods
    # ------------------------------------------------------------------

    async def _replace(
        self,
        operation: Literal["open", "create"],
        prepare: Callable[[], Awaitable[Any]],
    ) -> None:
        """替换会话。"""
        self._assert_available()
        if self._handle and self.phase != "idle":
            raise RuntimeError(
                f"Cannot {operation} a session while session is {self.phase or 'unavailable'}"
            )
        await self._run_operation(
            operation,
            lambda: self._track_attachment_operation(
                self._prepare_replacement(operation, prepare)
            ),
        )

    async def _track_attachment_operation(self, run: Awaitable[None]) -> None:
        """跟踪附件操作。"""
        pending = asyncio.ensure_future(run)
        self._pending_attachment_operations.add(pending)
        try:
            await pending
        finally:
            self._pending_attachment_operations.discard(pending)

    async def _prepare_replacement(
        self,
        operation: Literal["open", "create"],
        prepare: Callable[[], Awaitable[Any]],
    ) -> None:
        """准备替换会话。"""
        previous = self._handle
        next_handle = await prepare()
        await self._assert_not_disposed_after_await(next_handle)
        snapshot = next_handle.snapshot if hasattr(next_handle, "snapshot") else None
        if not snapshot:
            await self._detach(next_handle)
            raise RuntimeError(f"Session {next_handle.id} did not provide a snapshot")
        if (
            previous
            and previous.id != next_handle.id
            and getattr(previous, "attached", False)
            and self.phase != "idle"
        ):
            await self._detach(next_handle)
            raise RuntimeError(
                f"Cannot {operation} a session while session is {self.phase or 'unavailable'}"
            )
        if (
            previous
            and previous.id != next_handle.id
            and getattr(previous, "attached", False)
        ):
            try:
                await previous.detach()
            except Exception as error:
                try:
                    await self._detach(next_handle)
                except Exception:
                    raise error
                raise error
        await self._assert_not_disposed_after_await(next_handle)
        self._bind(next_handle, snapshot)

    async def _run_idle_operation(
        self,
        operation: Literal["setModel", "setThinking"],
        description: str,
        run: Callable[[], Awaitable[None]],
    ) -> None:
        """运行空闲操作。"""
        self._assert_available()
        self._require_handle()
        if self.phase != "idle":
            raise RuntimeError(
                f"Cannot {description} while session is {self.phase or 'unavailable'}"
            )
        await self._run_operation(operation, run)

    async def _run_operation(
        self,
        operation: RemoteSessionOperation,
        run: Callable[[], Awaitable[None]],
        preempt: bool = False,
    ) -> None:
        """运行操作。"""
        if preempt:
            self._assert_not_disposed()
        else:
            self._assert_available()
        previous = self._lifecycle
        busy: RemoteSessionLifecycle = {
            "status": "busy",
            "operation": operation,
        }
        self._lifecycle = busy
        self._active_operation_states.add(busy)
        self._notify()
        try:
            running = asyncio.ensure_future(run())
            dispose_waiter = asyncio.ensure_future(self._wait_for_dispose())
            done, _ = await asyncio.wait(
                [running, dispose_waiter],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if dispose_waiter in done:
                raise RuntimeError("Remote session is disposed")
            # 等待 running 完成（如果尚未完成）
            if not running.done():
                await running
        finally:
            self._active_operation_states.discard(busy)
            if not self.disposed and self._lifecycle == busy:
                if preempt and previous in self._active_operation_states:
                    self._lifecycle = previous
                elif self._handle:
                    self._lifecycle = {"status": "ready"}
                else:
                    self._lifecycle = {"status": "unbound"}
                self._notify()

    async def _wait_for_dispose(self) -> None:
        """等待释放信号。"""
        await self._dispose_event.wait()

    def _bind(
        self,
        handle: Any,
        known_snapshot: dict[str, Any] | None = None,
    ) -> None:
        """绑定会话。"""
        snapshot = known_snapshot or (
            handle.snapshot if hasattr(handle, "snapshot") else None
        )
        if not snapshot:
            raise RuntimeError(f"Session {handle.id} did not provide a snapshot")
        self._clear_subscriptions()
        self._handle = handle
        self._transcript = create_transcript_state(snapshot)

        if hasattr(handle, "subscribe"):
            self._unsubscribe_snapshot = handle.subscribe(self._on_snapshot_update)
        if hasattr(handle, "on_event"):
            self._unsubscribe_events = handle.on_event(self._handle_event)

    def _on_snapshot_update(self, next_snapshot: dict[str, Any]) -> None:
        """快照更新回调。"""
        if not self._transcript:
            return
        self._transcript = apply_transcript_snapshot(self._transcript, next_snapshot)
        self._notify()

    def _handle_event(self, event: dict[str, Any]) -> None:
        """事件处理。"""
        event_type = event.get("type")
        if event_type == "session_removed":
            self._clear_subscriptions()
            self._handle = None
            self._transcript = None
            if self._lifecycle.get("status") != "busy":
                self._lifecycle = {"status": "unbound"}
            self._notify()
            return
        if event_type != "session_progress" or not self._transcript:
            return
        self._transcript = apply_transcript_progress(
            self._transcript, event.get("progress", {})
        )
        self._notify()

    def _notify(self) -> None:
        """通知所有监听器。"""
        state = self.state
        for listener in list(self._listeners):
            self._call_listener(listener, state)

    def _call_listener(
        self,
        listener: Callable[[RemoteSessionState], None],
        state: RemoteSessionState,
    ) -> None:
        """调用监听器。"""
        try:
            listener(state)
        except Exception as error:
            self._report_listener_error(error)

    def _report_listener_error(self, error: object) -> None:
        """报告监听器错误。"""
        if not self._on_listener_error:
            return
        try:
            self._on_listener_error(
                error if isinstance(error, Exception) else Exception(str(error))
            )
        except Exception:
            pass

    def _clear_subscriptions(self) -> None:
        """清除订阅。"""
        if self._unsubscribe_snapshot:
            self._unsubscribe_snapshot()
        if self._unsubscribe_events:
            self._unsubscribe_events()
        self._unsubscribe_snapshot = None
        self._unsubscribe_events = None

    def _require_handle(self) -> Any:
        """获取会话句柄，不存在则抛出异常。"""
        if not self._handle:
            raise RuntimeError("No remote session is attached")
        return self._handle

    def _assert_available(self) -> None:
        """断言会话可用。"""
        self._assert_not_disposed()
        if self._lifecycle.get("status") == "busy":
            raise RuntimeError(
                f"Remote session is busy with {self._lifecycle.get('operation')}"
            )

    def _assert_not_disposed(self) -> None:
        """断言会话未释放。"""
        if self.disposed:
            raise RuntimeError("Remote session is disposed")

    async def _assert_not_disposed_after_await(self, handle: Any) -> None:
        """等待后断言会话未释放。"""
        if not self.disposed:
            return
        await self._detach(handle)
        raise _RemoteSessionDisposedError()

    async def _detach(self, handle: Any) -> None:
        """分离会话。"""
        await handle.dispose()


__all__ = [
    "CreateRemoteSessionOptions",
    "RemoteSession",
    "RemoteSessionLifecycle",
    "RemoteSessionOperation",
    "RemoteSessionOptions",
    "RemoteSessionState",
]
