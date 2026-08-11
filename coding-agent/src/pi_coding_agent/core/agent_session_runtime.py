"""Agent session runtime - owns the current AgentSession plus its cwd-bound services.

Corresponds to TS ``core/agent-session-runtime.ts``.
"""

from __future__ import annotations

import os
import shutil
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from .session_cwd import SessionCwdSource, assert_session_cwd_exists
from .tools.path_utils import resolve_path

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .agent_session import AgentSession
    from .agent_session_services import (
        AgentSessionRuntimeDiagnostic,
        AgentSessionServices,
        CreateAgentSessionFromServicesOptions,
        CreateAgentSessionServicesOptions,
    )
    from .extensions.types import (
        ProjectTrustContext,
        ReplacedSessionContext,
        SessionStartEvent,
    )
    from .session_manager import NewSessionOptions, SessionManager

# ---------------------------------------------------------------------------
# 运行时导入（模块将在后续阶段创建）
# ---------------------------------------------------------------------------

if TYPE_CHECKING:
    from .extensions.runner import (
        emit_session_shutdown_event,
    )
else:
    # 占位：extensions.runner 将在后续阶段创建
    async def emit_session_shutdown_event(
        _runner: Any, _event: dict[str, object]
    ) -> bool:
        return False


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class CreateAgentSessionRuntimeResult:
    """Result returned by runtime creation.

    The caller gets the created session, its cwd-bound services, and all
    diagnostics collected during setup.
    """

    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
        model_fallback_message: str | None = None,
    ) -> None:
        self.session = session
        self.services = services
        self.diagnostics = list(diagnostics) if diagnostics else []
        self.model_fallback_message = model_fallback_message


class CreateAgentSessionRuntimeFactory(Protocol):
    """Creates a full runtime for a target cwd and session manager.

    The factory closes over process-global fixed inputs, recreates cwd-bound
    services for the effective cwd, resolves session options against those
    services, and finally creates the AgentSession.
    """

    async def __call__(
        self,
        *,
        cwd: str,
        agent_dir: str,
        session_manager: SessionManager,
        session_start_event: SessionStartEvent | None = None,
        project_trust_context: ProjectTrustContext | None = None,
    ) -> CreateAgentSessionRuntimeResult: ...


# ---------------------------------------------------------------------------
# 异常
# ---------------------------------------------------------------------------


class SessionImportFileNotFoundError(FileNotFoundError):
    """Thrown when /import references a JSONL file path that does not exist."""

    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        super().__init__(f"File not found: {file_path}")
        self.name = "SessionImportFileNotFoundError"


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_user_message_text(content: str | list[dict[str, object]]) -> str:
    """Extract plain text from a user message content.

    Handles both plain string content and structured content arrays (like
    ``[{type: "text", text: "..."}]``).
    """
    if isinstance(content, str):
        return content

    parts: list[str] = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            text = part.get("text")
            if isinstance(text, str):
                parts.append(text)
    return "".join(parts)


# ---------------------------------------------------------------------------
# AgentSessionRuntime
# ---------------------------------------------------------------------------


class AgentSessionRuntime:
    """Owns the current AgentSession plus its cwd-bound services.

    Session replacement methods tear down the current runtime first, then create
    and apply the next runtime. If creation fails, the error is propagated to the
    caller. The caller is responsible for user-facing error handling.
    """

    def __init__(
        self,
        session: AgentSession,
        services: AgentSessionServices,
        create_runtime: CreateAgentSessionRuntimeFactory,
        diagnostics: list[AgentSessionRuntimeDiagnostic] | None = None,
        model_fallback_message: str | None = None,
    ) -> None:
        self._session = session
        self._services = services
        self._create_runtime = create_runtime
        self._diagnostics = list(diagnostics) if diagnostics else []
        self._model_fallback_message = model_fallback_message
        self._rebind_session: Callable[[AgentSession], Awaitable[None]] | None = None
        self._before_session_invalidate: Callable[[], None] | None = None

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------

    @property
    def services(self) -> AgentSessionServices:
        return self._services

    @property
    def session(self) -> AgentSession:
        return self._session

    @property
    def cwd(self) -> str:
        return self._services.cwd

    @property
    def diagnostics(self) -> list[AgentSessionRuntimeDiagnostic]:
        return self._diagnostics

    @property
    def model_fallback_message(self) -> str | None:
        return self._model_fallback_message

    # ------------------------------------------------------------------
    # 回调注册
    # ------------------------------------------------------------------

    def set_rebind_session(
        self,
        rebind_session: Callable[[AgentSession], Awaitable[None]] | None = None,
    ) -> None:
        """Set a callback that runs after the session is replaced.

        The callback receives the new session so the host can re-bind UI state.
        """
        self._rebind_session = rebind_session

    def set_before_session_invalidate(
        self,
        before_session_invalidate: Callable[[], None] | None = None,
    ) -> None:
        """Set a synchronous callback that runs after ``session_shutdown`` handlers
        finish but before the current session is invalidated.

        This is for host-owned UI teardown that must not yield to the event loop,
        such as detaching extension-provided TUI components before the old extension
        context becomes stale.
        """
        self._before_session_invalidate = before_session_invalidate

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    async def _emit_before_switch(
        self,
        reason: Literal["new", "resume"],
        target_session_file: str | None = None,
    ) -> dict[str, bool]:
        """Emit ``session_before_switch`` event and return cancellation status."""
        from .extensions.types import SessionBeforeSwitchEvent

        runner = self._session.extension_runner
        if runner is None or not runner.has_handlers("session_before_switch"):
            return {"cancelled": False}

        result = await runner.emit(
            SessionBeforeSwitchEvent(
                type="session_before_switch",
                reason=reason,
                target_session_file=target_session_file,
            )
        )
        return {"cancelled": bool(result and result.get("cancel") is True)}

    async def _emit_before_fork(
        self,
        entry_id: str,
        position: Literal["before", "at"],
    ) -> dict[str, bool]:
        """Emit ``session_before_fork`` event and return cancellation status."""
        from .extensions.types import SessionBeforeForkEvent

        runner = self._session.extension_runner
        if runner is None or not runner.has_handlers("session_before_fork"):
            return {"cancelled": False}

        result = await runner.emit(
            SessionBeforeForkEvent(
                type="session_before_fork",
                entry_id=entry_id,
                position=position,
            )
        )
        return {"cancelled": bool(result and result.get("cancel") is True)}

    async def _teardown_current(
        self,
        reason: Literal["quit", "reload", "new", "resume", "fork"],
        target_session_file: str | None = None,
    ) -> None:
        """Settle active response, emit shutdown, and dispose the current session."""
        from .extensions.types import SessionShutdownEvent

        await self._session.abort()
        runner = self._session.extension_runner
        if runner is not None:
            await emit_session_shutdown_event(
                runner,
                SessionShutdownEvent(
                    type="session_shutdown",
                    reason=reason,
                    target_session_file=target_session_file,
                ),
            )
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self._session.dispose()

    def _apply(self, result: CreateAgentSessionRuntimeResult) -> None:
        """Replace internal state with the result of a new runtime creation."""
        self._session = result.session
        self._services = result.services
        self._diagnostics = result.diagnostics
        self._model_fallback_message = result.model_fallback_message

    async def _finish_session_replacement(
        self,
        with_session: Callable[[ReplacedSessionContext], Awaitable[None]] | None = None,
    ) -> None:
        """Run post-replacement callbacks."""
        if self._rebind_session is not None:
            await self._rebind_session(self._session)
        if with_session is not None:
            await with_session(self._session.create_replaced_session_context())

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    async def switch_session(
        self,
        session_path: str,
        options: dict[str, object] | None = None,
    ) -> dict[str, bool]:
        """Switch to a different session file.

        Args:
            session_path: Path to the session file to resume.
            options: Optional dict with keys ``cwd_override`` (str),
                ``with_session`` (callable), ``project_trust_context_factory`` (callable).

        Returns:
            ``{"cancelled": True}`` when cancelled by ``session_before_switch``,
            otherwise ``{"cancelled": False}``.
        """
        options = options or {}
        cwd_override = cast("str | None", options.get("cwd_override"))
        with_session = cast(
            "Callable[[ReplacedSessionContext], Awaitable[None]] | None",
            options.get("with_session"),
        )
        project_trust_context_factory = cast(
            "Callable[[str], ProjectTrustContext] | None",
            options.get("project_trust_context_factory"),
        )

        before_result = await self._emit_before_switch("resume", session_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self._session.session_file
        session_manager = SessionManager.open(session_path, cwd_override)
        assert_session_cwd_exists(cast("SessionCwdSource", session_manager), self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self._services.agent_dir,
                session_manager=session_manager,
                session_start_event=cast(
                    "SessionStartEvent",
                    {
                        "type": "session_start",
                        "reason": "resume",
                        "previousSessionFile": previous_session_file,
                    },
                ),
                project_trust_context=project_trust_context_factory(
                    session_manager.get_cwd()
                )
                if project_trust_context_factory
                else None,
            ),
        )
        await self._finish_session_replacement(with_session)
        return {"cancelled": False}

    async def new_session(
        self,
        options: dict[str, object] | None = None,
    ) -> dict[str, bool]:
        """Create a new session in the current cwd.

        Args:
            options: Optional dict with keys ``parent_session`` (str),
                ``setup`` (callable), ``with_session`` (callable).

        Returns:
            ``{"cancelled": True}`` when cancelled by ``session_before_switch``,
            otherwise ``{"cancelled": False}``.
        """
        options = options or {}
        parent_session = cast("str | None", options.get("parent_session"))
        setup = cast(
            "Callable[[SessionManager], Awaitable[None]] | None",
            options.get("setup"),
        )
        with_session = cast(
            "Callable[[ReplacedSessionContext], Awaitable[None]] | None",
            options.get("with_session"),
        )

        before_result = await self._emit_before_switch("new")
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self._session.session_file
        session_dir = self._session.session_manager.get_session_dir()
        session_manager = (
            SessionManager.create(self.cwd, session_dir)
            if self._session.session_manager.is_persisted()
            else SessionManager.in_memory(self.cwd)
        )
        if parent_session is not None:
            session_manager.new_session(
                cast("NewSessionOptions", {"parentSession": parent_session})
            )

        await self._teardown_current("new", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=self.cwd,
                agent_dir=self._services.agent_dir,
                session_manager=session_manager,
                session_start_event=cast(
                    "SessionStartEvent",
                    {
                        "type": "session_start",
                        "reason": "new",
                        "previousSessionFile": previous_session_file,
                    },
                ),
            ),
        )
        if setup is not None:
            await setup(session_manager)
            self._session.agent.state.messages = (
                session_manager.build_session_context().messages
            )
        await self._finish_session_replacement(with_session)
        return {"cancelled": False}

    async def fork(
        self,
        entry_id: str,
        options: dict[str, object] | None = None,
    ) -> dict[str, bool | str | None]:
        """Fork the session at the given entry.

        Args:
            entry_id: The entry ID to fork at.
            options: Optional dict with keys ``position`` ("before" | "at"),
                ``with_session`` (callable).

        Returns:
            ``{"cancelled": True}`` when cancelled, otherwise
            ``{"cancelled": False, "selectedText": ...}``.
        """
        options = options or {}
        position = cast("Literal['before', 'at']", options.get("position", "before"))
        with_session = cast(
            "Callable[[ReplacedSessionContext], Awaitable[None]] | None",
            options.get("with_session"),
        )

        before_result = await self._emit_before_fork(entry_id, position)
        if before_result["cancelled"]:
            return {"cancelled": True}

        target_leaf_id: str | None
        selected_text: str | None = None

        selected_entry = self._session.session_manager.get_entry(entry_id)
        if selected_entry is None:
            raise ValueError("Invalid entry ID for forking")

        if position == "at":
            target_leaf_id = selected_entry.id
        else:
            if (
                getattr(selected_entry, "type", None) != "message"
                or getattr(getattr(selected_entry, "message", None), "role", None)
                != "user"
            ):
                raise ValueError("Invalid entry ID for forking")
            target_leaf_id = getattr(selected_entry, "parent_id", None)
            selected_text = _extract_user_message_text(
                getattr(getattr(selected_entry, "message", None), "content", ""),
            )

        previous_session_file = self._session.session_file
        if self._session.session_manager.is_persisted():
            current_session_file = self._session.session_file
            if current_session_file is None:
                raise ValueError("Persisted session is missing a session file")
            session_dir = self._session.session_manager.get_session_dir()
            if target_leaf_id is None:
                session_manager = SessionManager.create(self.cwd, session_dir)
                session_manager.new_session(
                    cast("NewSessionOptions", {"parentSession": current_session_file})
                )
                await self._teardown_current("fork", session_manager.get_session_file())
                self._apply(
                    await self._create_runtime(
                        cwd=self.cwd,
                        agent_dir=self._services.agent_dir,
                        session_manager=session_manager,
                        session_start_event=cast(
                            "SessionStartEvent",
                            {
                                "type": "session_start",
                                "reason": "fork",
                                "previousSessionFile": previous_session_file,
                            },
                        ),
                    ),
                )
                await self._finish_session_replacement(with_session)
                return {"cancelled": False, "selectedText": selected_text}

            if not os.path.exists(current_session_file):
                raise ValueError(
                    "This session has not been saved yet. "
                    "Wait for the first assistant response before cloning or forking it.",
                )
            session_manager = SessionManager.open(current_session_file)
            forked_session_path = session_manager.create_branched_session(
                target_leaf_id
            )
            if not forked_session_path:
                raise ValueError("Failed to create forked session")
            await self._teardown_current("fork", session_manager.get_session_file())
            self._apply(
                await self._create_runtime(
                    cwd=session_manager.get_cwd(),
                    agent_dir=self._services.agent_dir,
                    session_manager=session_manager,
                    session_start_event=cast(
                        "SessionStartEvent",
                        {
                            "type": "session_start",
                            "reason": "fork",
                            "previousSessionFile": previous_session_file,
                        },
                    ),
                ),
            )
            await self._finish_session_replacement(with_session)
            return {"cancelled": False, "selectedText": selected_text}

        # In-memory session path
        session_manager = self._session.session_manager
        if target_leaf_id is None:
            session_manager.new_session(
                cast("NewSessionOptions", {"parentSession": self._session.session_file})
            )
        else:
            session_manager.create_branched_session(target_leaf_id)
        await self._teardown_current("fork", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=self.cwd,
                agent_dir=self._services.agent_dir,
                session_manager=session_manager,
                session_start_event=cast(
                    "SessionStartEvent",
                    {
                        "type": "session_start",
                        "reason": "fork",
                        "previousSessionFile": previous_session_file,
                    },
                ),
            ),
        )
        await self._finish_session_replacement(with_session)
        return {"cancelled": False, "selectedText": selected_text}

    async def import_from_jsonl(
        self, input_path: str, cwd_override: str | None = None
    ) -> dict[str, bool]:
        """Import a session JSONL file and switch runtime state to the imported session.

        Args:
            input_path: Path to the JSONL file to import.
            cwd_override: Optional cwd override for the imported session.

        Returns:
            ``{"cancelled": True}`` when cancelled by ``session_before_switch``,
            otherwise ``{"cancelled": False}``.

        Raises:
            SessionImportFileNotFoundError: When the input path does not exist.
            MissingSessionCwdError: When the imported session cwd cannot be resolved
                and no override is provided.
        """
        resolved_path = resolve_path(input_path)
        if not os.path.exists(resolved_path):
            raise SessionImportFileNotFoundError(resolved_path)

        session_dir = self._session.session_manager.get_session_dir()
        if not os.path.exists(session_dir):
            os.makedirs(session_dir, exist_ok=True)

        destination_path = os.path.join(session_dir, os.path.basename(resolved_path))
        before_result = await self._emit_before_switch("resume", destination_path)
        if before_result["cancelled"]:
            return before_result

        previous_session_file = self._session.session_file
        if os.path.realpath(destination_path) != resolved_path:
            shutil.copy2(resolved_path, destination_path)

        session_manager = SessionManager.open(destination_path, cwd_override)
        assert_session_cwd_exists(cast("SessionCwdSource", session_manager), self.cwd)
        await self._teardown_current("resume", session_manager.get_session_file())
        self._apply(
            await self._create_runtime(
                cwd=session_manager.get_cwd(),
                agent_dir=self._services.agent_dir,
                session_manager=session_manager,
                session_start_event=cast(
                    "SessionStartEvent",
                    {
                        "type": "session_start",
                        "reason": "resume",
                        "previousSessionFile": previous_session_file,
                    },
                ),
            ),
        )
        await self._finish_session_replacement()
        return {"cancelled": False}

    async def dispose(self) -> None:
        """Shut down the current session and dispose resources."""
        from .extensions.types import SessionShutdownEvent

        runner = self._session.extension_runner
        if runner is not None:
            await emit_session_shutdown_event(
                runner,
                SessionShutdownEvent(
                    type="session_shutdown",
                    reason="quit",
                ),
            )
        if self._before_session_invalidate is not None:
            self._before_session_invalidate()
        self._session.dispose()


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


async def create_agent_session_runtime(
    create_runtime: CreateAgentSessionRuntimeFactory,
    options: dict[str, object],
) -> AgentSessionRuntime:
    """Create the initial runtime from a runtime factory and initial session target.

    The same factory is stored on the returned ``AgentSessionRuntime`` and reused
    for later ``/new``, ``/resume``, ``/fork``, and import flows.
    """
    assert_session_cwd_exists(
        cast("SessionCwdSource", options["session_manager"]),
        cast("str", options["cwd"]),
    )
    result = await create_runtime(
        cwd=cast("str", options["cwd"]),
        agent_dir=cast("str", options["agent_dir"]),
        session_manager=cast("SessionManager", options["session_manager"]),
        session_start_event=cast(
            "SessionStartEvent | None", options.get("session_start_event")
        ),
    )
    return AgentSessionRuntime(
        session=result.session,
        services=result.services,
        create_runtime=create_runtime,
        diagnostics=result.diagnostics,
        model_fallback_message=result.model_fallback_message,
    )


# ---------------------------------------------------------------------------
# 再导出
# ---------------------------------------------------------------------------

from .agent_session_services import (
    AgentSessionRuntimeDiagnostic,
    AgentSessionServices,
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)

__all__ = [
    "AgentSessionRuntime",
    "AgentSessionRuntimeDiagnostic",
    "AgentSessionServices",
    "CreateAgentSessionFromServicesOptions",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "CreateAgentSessionServicesOptions",
    "SessionImportFileNotFoundError",
    "create_agent_session_from_services",
    "create_agent_session_runtime",
    "create_agent_session_services",
]
