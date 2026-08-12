"""AgentSession - Core abstraction for agent lifecycle and session management.

This class is shared between all run modes (interactive, print, rpc).
It encapsulates:
- Agent state access
- Event subscription with automatic session persistence
- Model and thinking level management
- Compaction (manual and auto)
- Bash execution
- Session switching and branching
- Extension system integration
- Auto-retry logic

Modes use this class and add their own I/O layer on top.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, TypeAlias, cast

from pi_agent.agent import Agent
from pi_agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopTurnUpdate,
    AgentMessage,
    AgentState,
    AgentTool,
    AssistantMessage,
    BashExecutionMessage,
    CustomMessage,
    PrepareNextTurnContext,
    ThinkingLevel,
)
from pi_ai.auth.types import AuthResult
from pi_ai.compat import reset_api_providers
from pi_ai.models import (
    clamp_thinking_level,
    get_supported_thinking_levels,
    models_are_equal,
)
from pi_ai.session_resources import cleanup_session_resources
from pi_ai.types import (
    ImageContent,
    Model,
    ProviderHeaders,
    TextContent,
    Usage,
)
from pi_ai.utils.overflow import is_context_overflow, is_recoverable_length
from pi_ai.utils.retry import RetryCallbacks, is_retryable_assistant_error
from pi_ai.utils.text import content_text

from .auth_guidance import (
    format_no_api_key_found_message,
    format_no_model_selected_message,
)
from .bash_executor import BashResult, execute_bash_with_operations
from .compaction.branch_summarization import (
    collect_entries_for_branch_summary,
    generate_branch_summary,
)
from .compaction.compaction import (
    CompactionResult,
    calculate_context_tokens,
    compact,
    estimate_context_tokens,
    estimate_tokens,
    prepare_compaction,
    should_compact,
)
from .defaults import DEFAULT_THINKING_LEVEL
from .export_html import (  # type: ignore
    export_session_to_html,
)
from .export_html.tool_renderer import (  # type: ignore
    create_tool_html_renderer,
)
from .extensions.runner import ExtensionRunner, emit_session_shutdown_event
from .extensions.types import (
    AgentEndEvent,
    AgentStartEvent,
    CompactOptions,
    ContextUsage,
    ExtensionCommandContextActions,
    ExtensionError,
    ExtensionMode,
    ExtensionUIContext,
    InputSource,
    MessageEndEvent,
    MessageStartEvent,
    MessageUpdateEvent,
    RegisteredTool,
    ReplacedSessionContext,
    SessionBeforeCompactResult,
    SessionBeforeTreeResult,
    SessionShutdownEvent,
    SessionStartEvent,
    ToolDefinition,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    ToolExecutionUpdateEvent,
    ToolInfo,
    TreePreparation,
    TurnEndEvent,
    TurnStartEvent,
)
from .extensions.wrapper import wrap_registered_tools
from .model_registry import ModelRegistry
from .prompt_templates import PromptTemplate, expand_prompt_template
from .resource_loader import ResourceExtensionPaths
from .session_manager import (
    CURRENT_SESSION_VERSION,
    BranchSummaryEntry,
    CompactionEntry,
    SessionEntry,
    SessionHeader,
    SessionManager,
)
from .settings_manager import SettingsManager
from .slash_commands import SlashCommandInfo
from .source_info import SourceInfo, create_synthetic_source_info
from .system_prompt import BuildSystemPromptOptions, build_system_prompt
from .tools import create_all_tool_definitions
from .tools.bash import BashOperations
from .tools.tool_definition_wrapper import create_tool_definition_from_agent_tool
from .usage_totals import add_usage_to_totals, create_usage_totals

# Type aliases for extension system
ShutdownHandler: TypeAlias = Callable[[], None]
"""Handler for session shutdown."""
ExtensionErrorListener: TypeAlias = Callable[["ExtensionError"], None]
"""Listener for extension errors."""

# ============================================================================
# Skill Block Parsing
# ============================================================================


@dataclass
class ParsedSkillBlock:
    """Parsed skill block from a user message."""

    name: str
    location: str
    content: str
    user_message: str | None = None


def parse_skill_block(text: str) -> ParsedSkillBlock | None:
    """Parse a skill block from message text.

    Returns None if the text doesn't contain a skill block.
    """
    match = re.match(
        r'^<skill name="([^"]+)" location="([^"]+)">\n([\s\S]*?)\n</skill>(?:\n\n([\s\S]+))?$',
        text,
    )
    if not match:
        return None
    return ParsedSkillBlock(
        name=match.group(1),
        location=match.group(2),
        content=match.group(3),
        user_message=match.group(4).strip() if match.group(4) else None,
    )


# ============================================================================
# Agent Session Event Types
# ============================================================================


@dataclass
class AgentSessionAgentEndEvent:
    """Agent end event with retry information."""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage] = field(default_factory=list)
    will_retry: bool = False


@dataclass
class AgentSessionAgentSettledEvent:
    """Agent settled event."""

    type: Literal["agent_settled"] = "agent_settled"


@dataclass
class AgentSessionQueueUpdateEvent:
    """Queue update event."""

    type: Literal["queue_update"] = "queue_update"
    steering: list[str] = field(default_factory=list)
    follow_up: list[str] = field(default_factory=list)


@dataclass
class AgentSessionCompactionStartEvent:
    """Compaction start event."""

    type: Literal["compaction_start"] = "compaction_start"
    reason: Literal["manual", "threshold", "overflow"] = "manual"


@dataclass
class AgentSessionEntryAppendedEvent:
    """Entry appended event."""

    type: Literal["entry_appended"] = "entry_appended"
    entry: SessionEntry | None = None


@dataclass
class AgentSessionInfoChangedEvent:
    """Session info changed event."""

    type: Literal["session_info_changed"] = "session_info_changed"
    name: str | None = None


@dataclass
class AgentSessionThinkingLevelChangedEvent:
    """Thinking level changed event."""

    type: Literal["thinking_level_changed"] = "thinking_level_changed"
    level: ThinkingLevel = "off"


@dataclass
class AgentSessionCompactionEndEvent:
    """Compaction end event."""

    type: Literal["compaction_end"] = "compaction_end"
    reason: Literal["manual", "threshold", "overflow"] = "manual"
    result: CompactionResult | None = None
    aborted: bool = False
    will_retry: bool = False
    error_message: str | None = None


@dataclass
class AgentSessionAutoRetryStartEvent:
    """Auto-retry start event."""

    type: Literal["auto_retry_start"] = "auto_retry_start"
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""


@dataclass
class AgentSessionAutoRetryEndEvent:
    """Auto-retry end event."""

    type: Literal["auto_retry_end"] = "auto_retry_end"
    success: bool = False
    attempt: int = 0
    final_error: str | None = None


@dataclass
class AgentSessionSummarizationRetryScheduledEvent:
    """Summarization retry scheduled event."""

    type: Literal["summarization_retry_scheduled"] = "summarization_retry_scheduled"
    attempt: int = 0
    max_attempts: int = 0
    delay_ms: int = 0
    error_message: str = ""


@dataclass
class AgentSessionSummarizationBranchRetryStartEvent:
    """Summarization retry attempt start (branch summary)."""

    type: Literal["summarization_retry_attempt_start"] = (
        "summarization_retry_attempt_start"
    )
    source: Literal["branchSummary"] = "branchSummary"


@dataclass
class AgentSessionSummarizationCompactionRetryStartEvent:
    """Summarization retry attempt start (compaction)."""

    type: Literal["summarization_retry_attempt_start"] = (
        "summarization_retry_attempt_start"
    )
    source: Literal["compaction"] = "compaction"
    reason: Literal["manual", "threshold", "overflow"] = "manual"


@dataclass
class AgentSessionSummarizationRetryFinishedEvent:
    """Summarization retry finished event."""

    type: Literal["summarization_retry_finished"] = "summarization_retry_finished"


@dataclass
class AgentSessionBashExecutionUpdateEvent:
    """Bash execution update event."""

    type: Literal["bash_execution_update"] = "bash_execution_update"
    id: str | None = None
    delta: str = ""


AgentSessionEvent: TypeAlias = (
    AgentSessionAgentEndEvent
    | AgentSessionAgentSettledEvent
    | AgentSessionQueueUpdateEvent
    | AgentSessionCompactionStartEvent
    | AgentSessionEntryAppendedEvent
    | AgentSessionInfoChangedEvent
    | AgentSessionThinkingLevelChangedEvent
    | AgentSessionCompactionEndEvent
    | AgentSessionAutoRetryStartEvent
    | AgentSessionAutoRetryEndEvent
    | AgentSessionSummarizationRetryScheduledEvent
    | AgentSessionSummarizationBranchRetryStartEvent
    | AgentSessionSummarizationCompactionRetryStartEvent
    | AgentSessionSummarizationRetryFinishedEvent
    | AgentSessionBashExecutionUpdateEvent
)
"""Session-specific events that extend the core AgentEvent."""

AgentSessionEventListener: TypeAlias = Callable[[AgentSessionEvent], None]
"""Listener function for agent session events."""


# ============================================================================
# Main Types
# ============================================================================


def without_deleted_headers(
    headers: ProviderHeaders | None,
) -> dict[str, str] | None:
    """Filter out headers with null values."""
    if not headers:
        return None
    result: dict[str, str] = {}
    for key, value in headers.items():
        if value is not None:
            result[key] = value
    return result or None


@dataclass
class AgentSessionConfig:
    """Configuration for AgentSession."""

    agent: Agent
    session_manager: SessionManager
    settings_manager: SettingsManager
    cwd: str
    scoped_models: list[dict[str, Any]] | None = None
    """Models to cycle through with Ctrl+P (from --models flag)."""
    resource_loader: Any = None
    """Resource loader for extensions, skills, prompts, themes, context files, and system prompt."""
    custom_tools: list[ToolDefinition] | None = None
    """SDK custom tools registered outside extensions."""
    model_runtime: Any = None
    """Canonical model/auth runtime used by coding-agent internals."""
    initial_active_tool_names: list[str] | None = None
    """Initial active built-in tool names. Default: [read, bash, edit, write]."""
    allowed_tool_names: list[str] | None = None
    """Optional allowlist of tool names."""
    excluded_tool_names: list[str] | None = None
    """Optional denylist of tool names."""
    base_tools_override: dict[str, AgentTool] | None = None
    """Override base tools (useful for custom runtimes)."""
    extension_runner_ref: dict[str, Any] | None = None
    """Mutable ref used by Agent to access the current ExtensionRunner."""
    session_start_event: Any | None = None
    """Session start event metadata emitted when extensions bind to this runtime."""


@dataclass
class ExtensionBindings:
    """Extension bindings for AgentSession."""

    ui_context: ExtensionUIContext | None = None
    mode: ExtensionMode | None = None
    command_context_actions: ExtensionCommandContextActions | None = None
    abort_handler: Callable[[], None] | None = None
    shutdown_handler: ShutdownHandler | None = None
    on_error: ExtensionErrorListener | None = None


@dataclass
class PromptOptions:
    """Options for AgentSession.prompt()."""

    expand_prompt_templates: bool = True
    """Whether to expand file-based prompt templates (default: true)."""
    images: list[ImageContent] | None = None
    """Image attachments."""
    streaming_behavior: Literal["steer", "follow_up"] | None = None
    """When streaming, how to queue the message."""
    source: InputSource | None = None
    """Source of input for extension input event handlers. Defaults to 'interactive'."""
    preflight_result: Callable[[bool], None] | None = None
    """Internal hook used by RPC mode to observe prompt preflight acceptance or rejection."""


@dataclass
class ModelCycleResult:
    """Result from cycle_model()."""

    model: Model | None = None
    thinking_level: ThinkingLevel = "off"
    is_scoped: bool = False
    """Whether cycling through scoped models (--models flag) or all available."""


@dataclass
class SessionStats:
    """Session statistics for /session command."""

    session_file: str | None = None
    session_id: str = ""
    user_messages: int = 0
    assistant_messages: int = 0
    tool_calls: int = 0
    tool_results: int = 0
    total_messages: int = 0
    tokens: dict[str, int] = field(
        default_factory=lambda: {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total": 0,
        }
    )
    cost: float = 0.0
    context_usage: ContextUsage | None = None


@dataclass
class ToolDefinitionEntry:
    """Internal tool definition entry with source info."""

    definition: ToolDefinition
    source_info: SourceInfo


def estimate_messages_tokens(messages: list[AgentMessage]) -> int:
    """Estimate token count for a list of messages."""
    tokens = 0
    for message in messages:
        tokens += estimate_tokens(message)
    return tokens


def get_latest_compaction_entry(
    entries: list[SessionEntry],
) -> CompactionEntry[Any] | None:
    """Get the latest compaction entry from a list of session entries."""
    for entry in reversed(entries):
        if isinstance(entry, CompactionEntry):
            return entry
    return None


# ============================================================================
# Constants
# ============================================================================

THINKING_LEVELS: list[ThinkingLevel] = ["off", "minimal", "low", "medium", "high"]
"""Standard thinking levels."""

# ============================================================================
# AgentSession Class
# ============================================================================


class AgentSession:
    """Core abstraction for agent lifecycle and session management."""

    agent: Agent
    session_manager: SessionManager
    settings_manager: SettingsManager

    def __init__(self, config: AgentSessionConfig) -> None:
        self.agent = config.agent
        self.session_manager = config.session_manager
        self.settings_manager = config.settings_manager
        self._scoped_models: list[dict[str, Any]] = config.scoped_models or []

        # Event subscription state
        self._unsubscribe_agent: Callable[[], None] | None = None
        self._event_listeners: list[AgentSessionEventListener] = []
        self._is_agent_run_active: bool = False
        self._idle_wait_event: asyncio.Event = asyncio.Event()
        self._idle_wait_event.set()

        # Tracks pending steering messages for UI display
        self._steering_messages: list[str] = []
        # Tracks pending follow-up messages for UI display
        self._follow_up_messages: list[str] = []
        # Messages queued to be included with the next user prompt as context ("asides")
        self._pending_next_turn_messages: list[CustomMessage] = []

        # Compaction state
        self._compaction_abort_event: asyncio.Event | None = None
        self._auto_compaction_abort_event: asyncio.Event | None = None
        self._overflow_recovery_attempted: bool = False

        # Branch summarization state
        self._branch_summary_abort_event: asyncio.Event | None = None

        # Retry state
        self._retry_abort_event: asyncio.Event | None = None
        self._retry_attempt: int = 0

        # Bash execution state
        self._bash_abort_events: list[asyncio.Event] = []
        self._pending_bash_messages: list[BashExecutionMessage] = []

        # Extension system
        self._extension_runner: ExtensionRunner | None = None
        self._turn_index: int = 0

        self._resource_loader: Any = config.resource_loader
        self._custom_tools: list[ToolDefinition] = config.custom_tools or []
        self._base_tool_definitions: dict[str, ToolDefinition] = {}
        self._cwd: str = config.cwd
        self._extension_runner_ref: dict[str, Any] | None = config.extension_runner_ref
        self._initial_active_tool_names: list[str] | None = (
            config.initial_active_tool_names
        )
        self._allowed_tool_names: set[str] | None = (
            set(config.allowed_tool_names) if config.allowed_tool_names else None
        )
        self._excluded_tool_names: set[str] | None = (
            set(config.excluded_tool_names) if config.excluded_tool_names else None
        )
        self._base_tools_override: dict[str, AgentTool] | None = (
            config.base_tools_override
        )
        self._session_start_event: SessionStartEvent = (
            SessionStartEvent(**config.session_start_event)
            if isinstance(config.session_start_event, dict)
            else config.session_start_event
            if config.session_start_event
            else SessionStartEvent(reason="startup")
        )
        self._extension_ui_context: ExtensionUIContext | None = None
        self._extension_mode: ExtensionMode = "print"
        self._extension_command_context_actions: (
            ExtensionCommandContextActions | None
        ) = None
        self._extension_abort_handler: Callable[[], None] | None = None
        self._extension_shutdown_handler: ShutdownHandler | None = None
        self._extension_error_listener: ExtensionErrorListener | None = None
        self._extension_error_unsubscriber: Callable[[], None] | None = None

        self._model_runtime: Any = config.model_runtime

        # Tool registry for extension get_tools/set_tools
        self._tool_registry: dict[str, AgentTool] = {}
        self._tool_definitions: dict[str, ToolDefinitionEntry] = {}
        self._tool_prompt_snippets: dict[str, str] = {}
        self._tool_prompt_guidelines: dict[str, list[str]] = {}

        # Base system prompt (without extension appends)
        self._base_system_prompt: str = ""
        self._base_system_prompt_options: BuildSystemPromptOptions | None = None
        self._system_prompt_override: str | None = None

        # Always subscribe to agent events for internal handling
        self._unsubscribe_agent = self.agent.subscribe(
            lambda event, _signal=None: self._handle_agent_event(event)  # type: ignore[misc]
        )
        self._install_agent_tool_hooks()
        self._install_agent_next_turn_refresh()

        self._build_runtime(
            active_tool_names=self._initial_active_tool_names,
            include_all_extension_tools=True,
        )

    @property
    def model_runtime(self) -> Any:
        return self._model_runtime

    async def _get_required_request_auth(self, model: Model) -> dict[str, Any]:
        """Get required auth for a model request."""
        result: AuthResult | None = None
        try:
            result = await self._model_runtime.get_auth(model)
        except Exception as error:
            cause = getattr(error, "__cause__", None)
            if (
                cause
                and isinstance(cause, Exception)
                and "authHeader requires a resolved API key" in str(cause)
            ):
                raise ValueError(
                    format_no_api_key_found_message(model.provider)
                ) from error
            raise

        if result and (result["auth"].get("api_key") or result["auth"].get("headers")):
            request_model = model
            if result["auth"].get("base_url"):
                # Create a model-like object with base_url
                request_model = cast(
                    Model,
                    {**cast(Any, model), "base_url": result["auth"]["base_url"]},
                )
            return {
                "model": request_model,
                "api_key": result["auth"]["api_key"],
                "headers": without_deleted_headers(result["auth"].get("headers")),
                "env": result.get("env"),
            }

        is_oauth = self._model_runtime.is_using_oauth(model.provider)
        if is_oauth:
            raise ValueError(
                f'Authentication failed for "{model.provider}". '
                f"Credentials may have expired or network is unavailable. "
                f"Run '/login {model.provider}' to re-authenticate."
            )
        raise ValueError(format_no_api_key_found_message(model.provider))

    async def _get_summarization_request_auth(self, model: Model) -> dict[str, Any]:
        """Get auth for summarization request (graceful failure)."""
        try:
            result = await self._model_runtime.get_auth(model)
            if not result:
                return {"model": model}
            request_model = model
            if result["auth"].get("base_url"):
                request_model = cast(
                    Model,
                    {**cast(Any, model), "base_url": result["auth"]["base_url"]},
                )
            return {
                "model": request_model,
                "api_key": result["auth"].get("api_key"),
                "headers": without_deleted_headers(result["auth"].get("headers")),
                "env": result.get("env"),
            }
        except Exception:
            return {"model": model}

    def _install_agent_tool_hooks(self) -> None:
        """Install tool hooks once on the Agent instance."""
        agent = self.agent

        async def before_tool_call(tool_call: Any, args: dict[str, Any]) -> Any:
            runner = self._extension_runner
            if not runner or not runner.has_handlers("tool_call"):
                return None
            try:
                return await runner.emit_tool_call(
                    {  # type: ignore[arg-type]
                        "type": "tool_call",
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "input": args,
                    }
                )
            except Exception as err:
                raise RuntimeError(
                    f"Extension failed, blocking execution: {err}"
                ) from err

        async def after_tool_call(
            tool_call: Any, args: dict[str, Any], result: Any, is_error: bool
        ) -> Any:
            runner = self._extension_runner
            if not runner:
                return None
            hook_result = (
                await runner.emit_tool_result(
                    {  # type: ignore[arg-type]
                        "type": "tool_result",
                        "tool_name": tool_call.name,
                        "tool_call_id": tool_call.id,
                        "input": args,
                        "content": result.content,
                        "details": result.details,
                        "is_error": is_error,
                        "usage": result.usage,
                    }
                )
                if runner.has_handlers("tool_result")
                else None
            )
            content = (
                hook_result.content
                if hook_result and hook_result.content
                else (result.content or [])
            )
            from ..utils.tool_result_images import (  # type: ignore[import-untyped]
                normalize_tool_result_images,
            )

            normalized_content = await normalize_tool_result_images(
                content,
                auto_resize_images=self.settings_manager.get_image_auto_resize(),
            )
            if not hook_result and normalized_content is content:
                return None
            return {
                "content": normalized_content,
                "details": hook_result.details if hook_result else None,
                "is_error": hook_result.is_error
                if hook_result is not None
                else is_error,
                "usage": hook_result.usage if hook_result else None,
            }

        agent.before_tool_call = before_tool_call  # type: ignore[assignment]
        agent.after_tool_call = after_tool_call  # type: ignore[assignment]

    def _install_agent_next_turn_refresh(self) -> None:
        """Install next-turn refresh hook to refresh system prompt and tools."""
        agent = self.agent
        previous_prepare = getattr(agent, "prepare_next_turn_with_context", None)
        if previous_prepare is None:
            prev_no_ctx = getattr(agent, "prepare_next_turn", None)
            if prev_no_ctx is not None:

                async def with_ctx(
                    turn: PrepareNextTurnContext, signal: asyncio.Event | None = None
                ) -> Any:
                    return await prev_no_ctx(signal) if prev_no_ctx else None

                previous_prepare = with_ctx

        async def prepare_next_turn_with_context(
            turn: PrepareNextTurnContext, signal: asyncio.Event | None = None
        ) -> Any:
            previous_snapshot = (
                await previous_prepare(turn, signal) if previous_prepare else None
            )
            # turn.context 是 AgentContext 对象，需转为 dict 才能用 ** 展开
            previous_context = (
                previous_snapshot.get("context")
                if previous_snapshot
                and isinstance(previous_snapshot, dict)
                and "context" in previous_snapshot
                else (
                    turn.context.model_dump()
                    if hasattr(turn.context, "model_dump")
                    else dict(turn.context)
                )
            )
            next_context = AgentContext(
                **{
                    **previous_context,
                    "system_prompt": self._system_prompt_override
                    or self._base_system_prompt,
                    "tools": list(self.agent.state.tools),
                }
            )
            return AgentLoopTurnUpdate(
                context=next_context,
                model=self.agent.state.model,
                thinking_level=self.agent.state.thinking_level,
            )

        agent.prepare_next_turn_with_context = prepare_next_turn_with_context

    # =========================================================================
    # Event Subscription
    # =========================================================================

    def _emit(self, event: AgentSessionEvent) -> None:
        """Emit an event to all listeners."""
        for listener in self._event_listeners:
            listener(event)

    def _emit_queue_update(self) -> None:
        """Emit queue update event."""
        self._emit(
            AgentSessionQueueUpdateEvent(
                steering=list(self._steering_messages),
                follow_up=list(self._follow_up_messages),
            )
        )

    def _resolve_idle_wait_if_idle(self) -> None:
        """Resolve idle wait if agent is idle."""
        if self._is_agent_run_active:
            return
        self._idle_wait_event.set()

    async def _emit_agent_settled(self) -> None:
        """Emit agent settled event."""
        self._is_agent_run_active = False
        try:
            if self._extension_runner:
                await self._extension_runner.emit({"type": "agent_settled"})  # type: ignore[arg-type]
            self._emit(AgentSessionAgentSettledEvent())
        finally:
            self._resolve_idle_wait_if_idle()

    # Track last assistant message for auto-compaction check
    _last_assistant_message: AssistantMessage | None = None

    async def _handle_agent_event(self, event: AgentEvent) -> None:
        """Internal handler for agent events."""
        # 调试：确保 event 是 Pydantic model 而非 dict
        if isinstance(event, dict):
            raise TypeError(
                f"_handle_agent_event 收到 dict 而非 AgentEvent (type={event.get('type', '?')})"
            )
        # When a user message starts, check if it's from either queue and remove it
        if event.type == "message_start" and event.message.role == "user":
            self._overflow_recovery_attempted = False
            message_text = content_text(event.message.content, "")  # type: ignore[arg-type]
            if message_text:
                steering_index = next(
                    (
                        i
                        for i, m in enumerate(self._steering_messages)
                        if m == message_text
                    ),
                    -1,
                )
                if steering_index != -1:
                    self._steering_messages.pop(steering_index)
                    self._emit_queue_update()
                else:
                    follow_up_index = next(
                        (
                            i
                            for i, m in enumerate(self._follow_up_messages)
                            if m == message_text
                        ),
                        -1,
                    )
                    if follow_up_index != -1:
                        self._follow_up_messages.pop(follow_up_index)
                        self._emit_queue_update()

        # Emit to extensions first
        if self._extension_runner:
            await self._emit_extension_event(event)

        # Notify all listeners
        if event.type == "agent_end":
            self._emit(
                AgentSessionAgentEndEvent(
                    messages=event.messages,
                    will_retry=self._will_retry_after_agent_end(event),
                )
            )
        else:
            # For other event types, emit as-is using the AgentSessionEvent type
            self._emit(event)  # type: ignore[arg-type]

        # Handle session persistence
        if event.type == "message_end":
            if event.message.role == "custom":
                self.session_manager.append_custom_message_entry(
                    event.message.custom_type,
                    event.message.content,
                    event.message.display,
                    event.message.details,
                )
            elif event.message.role in ("user", "assistant", "toolResult"):
                self.session_manager.append_message(event.message)

            if event.message.role == "assistant":
                self._last_assistant_message = event.message
                assistant_msg = event.message
                if assistant_msg.stop_reason not in ("error", "length"):
                    self._overflow_recovery_attempted = False
                if assistant_msg.stop_reason != "error" and self._retry_attempt > 0:
                    self._emit(
                        AgentSessionAutoRetryEndEvent(
                            success=True,
                            attempt=self._retry_attempt,
                        )
                    )
                    self._retry_attempt = 0

    def _will_retry_after_agent_end(self, event: AgentEvent) -> bool:
        """Check if agent will retry after agent_end."""
        if event.type != "agent_end":
            return False
        settings = self.settings_manager.get_retry_settings()
        if not (settings.enabled or False) or self._retry_attempt >= (
            settings.max_retries or 0
        ):
            return False
        for i in range(len(event.messages) - 1, -1, -1):
            message = event.messages[i]
            if message.role == "assistant":
                return self._is_retryable_error(message)
        return False

    def _find_last_assistant_message(self) -> AssistantMessage | None:
        """Find the last assistant message in agent state."""
        messages = self.agent.state.messages
        for i in range(len(messages) - 1, -1, -1):
            msg = messages[i]
            if msg.role == "assistant":
                return msg
        return None

    def _replace_message_in_place(
        self, target: AgentMessage, replacement: AgentMessage
    ) -> None:
        """Replace a message in place in agent state."""
        if target is replacement:
            return
        target_dict = target.__dict__
        target_dict.clear()
        target_dict.update(replacement.__dict__)

    async def _emit_extension_event(self, event: AgentEvent) -> None:
        """Emit extension events based on agent events."""
        runner = self._extension_runner
        if not runner:
            return

        if event.type == "agent_start":
            self._turn_index = 0
            await runner.emit(AgentStartEvent(type="agent_start"))
        elif event.type == "agent_end":
            await runner.emit(AgentEndEvent(type="agent_end", messages=event.messages))
        elif event.type == "turn_start":
            await runner.emit(
                TurnStartEvent(
                    type="turn_start",
                    turn_index=self._turn_index,
                    timestamp=int(time.time() * 1000),
                )
            )
        elif event.type == "turn_end":
            await runner.emit(
                TurnEndEvent(
                    type="turn_end",
                    turn_index=self._turn_index,
                    message=event.message,
                    tool_results=event.tool_results,
                )
            )
            self._turn_index += 1
        elif event.type == "message_start":
            await runner.emit(
                MessageStartEvent(
                    type="message_start",
                    message=event.message,
                )
            )
        elif event.type == "message_update":
            await runner.emit(
                MessageUpdateEvent(
                    type="message_update",
                    message=event.message,
                    assistant_message_event=event.assistant_message_event,
                )
            )
        elif event.type == "message_end":
            extension_event = MessageEndEvent(
                type="message_end",
                message=event.message,
            )
            replacement = await runner.emit_message_end(extension_event)
            if replacement:
                normalized = replacement
                if (
                    replacement.role in ("user", "assistant", "toolResult", "custom")
                    and replacement.content is None
                ):
                    normalized = replacement.__class__(
                        **{**replacement.__dict__, "content": []}
                    )
                self._replace_message_in_place(event.message, normalized)
        elif event.type == "tool_execution_start":
            await runner.emit(
                ToolExecutionStartEvent(
                    type="tool_execution_start",
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    args=event.args,
                )
            )
        elif event.type == "tool_execution_update":
            await runner.emit(
                ToolExecutionUpdateEvent(
                    type="tool_execution_update",
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    args=event.args,
                    partial_result=event.partial_result,
                )
            )
        elif event.type == "tool_execution_end":
            await runner.emit(
                ToolExecutionEndEvent(
                    type="tool_execution_end",
                    tool_call_id=event.tool_call_id,
                    tool_name=event.tool_name,
                    result=event.result,
                    is_error=event.is_error,
                )
            )

    def subscribe(self, listener: AgentSessionEventListener) -> Callable[[], None]:
        """Subscribe to agent events.

        Session persistence is handled internally.
        Returns unsubscribe function for this listener.
        """
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def _disconnect_from_agent(self) -> None:
        """Disconnect from agent events during disposal."""
        if self._unsubscribe_agent:
            self._unsubscribe_agent()
            self._unsubscribe_agent = None

    def dispose(self) -> None:
        """Remove all listeners and disconnect from agent."""
        try:
            self.abort_retry()
            self.abort_compaction()
            self.abort_branch_summary()
            self.abort_bash()
            self.agent.abort()
        except Exception:
            pass

        if self._extension_runner:
            self._extension_runner.invalidate(
                "This extension ctx is stale after session replacement or reload."
            )
        self._disconnect_from_agent()
        self._event_listeners = []
        cleanup_session_resources(self.session_id)

    # =========================================================================
    # Read-only State Access
    # =========================================================================

    @property
    def state(self) -> AgentState:
        """Full agent state."""
        return self.agent.state  # type: ignore[return-value]

    @property
    def model(self) -> Model | None:
        """Current model (may be None if not yet selected)."""
        return self.agent.state.model

    @property
    def thinking_level(self) -> ThinkingLevel:
        """Current thinking level."""
        return self.agent.state.thinking_level

    @property
    def is_streaming(self) -> bool:
        """Whether the session is currently processing an agent run."""
        return self._is_agent_run_active

    @property
    def is_idle(self) -> bool:
        """Whether the session has no active agent run."""
        return not self._is_agent_run_active

    @property
    def system_prompt(self) -> str:
        """Current effective system prompt."""
        return self.agent.state.system_prompt

    @property
    def retry_attempt(self) -> int:
        """Current retry attempt (0 if not retrying)."""
        return self._retry_attempt

    def get_active_tool_names(self) -> list[str]:
        """Get the names of currently active tools."""
        return [t.name for t in self.agent.state.tools]

    def get_all_tools(self) -> list[ToolInfo]:
        """Get all configured tools with metadata."""
        result: list[ToolInfo] = []
        for entry in self._tool_definitions.values():
            result.append(
                ToolInfo(
                    name=entry.definition.name,
                    description=entry.definition.description,
                    parameters=entry.definition.parameters,
                    prompt_guidelines=entry.definition.prompt_guidelines,
                    source_info=entry.source_info,
                )
            )
        return result

    def get_tool_definition(self, name: str) -> ToolDefinition | None:
        """Get tool definition by name."""
        entry = self._tool_definitions.get(name)
        return entry.definition if entry else None

    def set_active_tools_by_name(self, tool_names: list[str]) -> None:
        """Set active tools by name."""
        tools: list[AgentTool] = []
        valid_tool_names: list[str] = []
        for name in tool_names:
            tool = self._tool_registry.get(name)
            if tool:
                tools.append(tool)
                valid_tool_names.append(name)
        self.agent.state.tools = tools
        self._base_system_prompt = self._rebuild_system_prompt(valid_tool_names)
        self.agent.state.system_prompt = (
            self._system_prompt_override or self._base_system_prompt
        )

    @property
    def is_compacting(self) -> bool:
        """Whether compaction or branch summarization is currently running."""
        return (
            self._auto_compaction_abort_event is not None
            or self._compaction_abort_event is not None
            or self._branch_summary_abort_event is not None
        )

    @property
    def messages(self) -> list[AgentMessage]:
        """All messages including custom types."""
        return self.agent.state.messages

    @property
    def steering_mode(self) -> Literal["all", "one-at-a-time"]:
        """Current steering mode."""
        return self.agent.steering_mode

    @property
    def follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        """Current follow-up mode."""
        return self.agent.follow_up_mode

    @property
    def session_file(self) -> str | None:
        """Current session file path, or None if sessions are disabled."""
        return self.session_manager.get_session_file()

    @property
    def session_id(self) -> str:
        """Current session ID."""
        return self.session_manager.get_session_id()

    @property
    def session_name(self) -> str | None:
        """Current session display name, if set."""
        return self.session_manager.get_session_name()

    @property
    def scoped_models(self) -> Sequence[dict[str, Any]]:
        """Scoped models for cycling (from --models flag)."""
        return self._scoped_models

    def set_scoped_models(self, scoped_models: list[dict[str, Any]]) -> None:
        """Update scoped models for cycling."""
        self._scoped_models = scoped_models

    @property
    def prompt_templates(self) -> Sequence[PromptTemplate]:
        """File-based prompt templates."""
        if self._resource_loader:
            prompts = self._resource_loader.get_prompts()
            return prompts["prompts"] if prompts else []
        return []

    def _normalize_prompt_snippet(self, text: str | None) -> str | None:
        """Normalize a prompt snippet to a single line."""
        if not text:
            return None
        one_line = text.replace("\r\n", " ").replace("\n", " ").replace("\r", " ")
        one_line = " ".join(one_line.split())
        return one_line if len(one_line) > 0 else None

    def _normalize_prompt_guidelines(self, guidelines: list[str] | None) -> list[str]:
        """Normalize prompt guidelines (deduplicate)."""
        if not guidelines:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for guideline in guidelines:
            normalized = guideline.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                result.append(normalized)
        return result

    def _rebuild_system_prompt(self, tool_names: list[str]) -> str:
        """Rebuild the system prompt with current tool set."""
        valid_tool_names = [name for name in tool_names if name in self._tool_registry]
        tool_snippets: dict[str, str] = {}
        prompt_guidelines: list[str] = []
        for name in valid_tool_names:
            snippet = self._tool_prompt_snippets.get(name)
            if snippet:
                tool_snippets[name] = snippet
            guidelines = self._tool_prompt_guidelines.get(name)
            if guidelines:
                prompt_guidelines.extend(guidelines)

        loader_system_prompt = ""
        loader_append_system_prompt: list[str] = []
        loaded_skills: list[Any] = []
        loaded_context_files: list[Any] = []

        if self._resource_loader:
            loader_system_prompt = self._resource_loader.get_system_prompt()
            loader_append_system_prompt = (
                self._resource_loader.get_append_system_prompt()
            )
            loaded_skills = self._resource_loader.get_skills()["skills"]
            loaded_context_files = self._resource_loader.get_agents_files()[
                "agents_files"
            ]

        append_system_prompt = (
            "\n\n".join(loader_append_system_prompt)
            if loader_append_system_prompt
            else None
        )

        self._base_system_prompt_options = BuildSystemPromptOptions(
            cwd=self._cwd,
            skills=loaded_skills,
            context_files=loaded_context_files,
            custom_prompt=loader_system_prompt,
            append_system_prompt=append_system_prompt,
            selected_tools=valid_tool_names,
            tool_snippets=tool_snippets,
            prompt_guidelines=prompt_guidelines,
        )
        return build_system_prompt(self._base_system_prompt_options)

    # =========================================================================
    # Prompting
    # =========================================================================

    async def _run_agent_prompt(
        self, messages: AgentMessage | list[AgentMessage]
    ) -> None:
        """Run the agent prompt."""
        self._is_agent_run_active = True
        try:
            await self.agent.prompt(messages)
            while await self._handle_post_agent_run():
                await self.agent.continue_()
        finally:
            self._system_prompt_override = None
            self._flush_pending_bash_messages()
            await self._emit_agent_settled()

    async def _handle_post_agent_run(self) -> bool:
        """Handle post-agent-run logic. Returns True if agent should continue."""
        msg = self._last_assistant_message
        self._last_assistant_message = None
        if not msg:
            return False

        if self._is_retryable_error(msg) and await self._prepare_retry(msg):
            return True

        if msg.stop_reason == "error" and self._retry_attempt > 0:
            self._emit(
                AgentSessionAutoRetryEndEvent(
                    success=False,
                    attempt=self._retry_attempt,
                    final_error=msg.error_message,
                )
            )
            self._retry_attempt = 0

        if await self._check_compaction(msg):
            return True

        return self.agent.has_queued_messages()

    async def prompt(self, text: str, options: PromptOptions | None = None) -> None:
        """Send a prompt to the agent."""
        opts = options or PromptOptions()
        expand_prompt_templates = opts.expand_prompt_templates
        preflight_result = opts.preflight_result
        messages: list[AgentMessage] | None = None

        try:
            # Handle extension commands first
            if expand_prompt_templates and text.startswith("/"):
                handled = await self._try_execute_extension_command(text)
                if handled:
                    if preflight_result:
                        preflight_result(True)
                    return

            if self._compaction_abort_event is not None:
                raise RuntimeError(
                    "Cannot submit a prompt while compaction is in progress."
                )

            # Emit input event for extension interception
            current_text = text
            current_images = opts.images
            if self._extension_runner and self._extension_runner.has_handlers("input"):
                input_result = await self._extension_runner.emit_input(
                    current_text,
                    current_images,
                    opts.source or "interactive",
                    opts.streaming_behavior if self.is_streaming else None,
                )
                if input_result.action == "handled":
                    if preflight_result:
                        preflight_result(True)
                    return
                if input_result.action == "transform":
                    current_text = input_result.text
                    current_images = input_result.images or current_images

            # Expand skill commands and prompt templates
            expanded_text = current_text
            if expand_prompt_templates:
                expanded_text = self._expand_skill_command(expanded_text)
                expanded_text = expand_prompt_template(
                    expanded_text, list(self.prompt_templates)
                )

            # If streaming, queue
            if self.is_streaming:
                if not opts.streaming_behavior:
                    raise RuntimeError(
                        "Agent is already processing. Specify streaming_behavior ('steer' or 'follow_up') to queue the message."
                    )
                if opts.streaming_behavior == "follow_up":
                    await self._queue_follow_up(expanded_text, current_images)
                else:
                    await self._queue_steer(expanded_text, current_images)
                if preflight_result:
                    preflight_result(True)
                return

            # Flush pending bash messages
            self._flush_pending_bash_messages()

            # Validate model
            if not self.model:
                raise ValueError(format_no_model_selected_message())

            has_configured_auth = (
                self._model_runtime.has_configured_auth(self.model.provider)
                or (await self._model_runtime.check_auth(self.model.provider))
                is not None
            )
            if not has_configured_auth:
                is_oauth = self._model_runtime.is_using_oauth(self.model.provider)
                if is_oauth:
                    raise ValueError(
                        f'Authentication failed for "{self.model.provider}". '
                        f"Credentials may have expired or network is unavailable. "
                        f"Run '/login {self.model.provider}' to re-authenticate."
                    )
                raise ValueError(format_no_api_key_found_message(self.model.provider))

            # Check if we need to compact before sending
            last_assistant = self._find_last_assistant_message()
            if last_assistant:
                await self._check_compaction(last_assistant, False)

            # Build messages array
            messages = []
            user_content: list[TextContent | ImageContent] = [
                {"type": "text", "text": expanded_text}  # type: ignore[list-item]
            ]
            if current_images:
                user_content.extend(current_images)
            messages.append(
                {  # type: ignore[arg-type]
                    "role": "user",
                    "content": user_content,
                    "timestamp": int(time.time() * 1000),
                }
            )

            # Inject pending next-turn messages
            for msg in self._pending_next_turn_messages:
                messages.append(msg)
            self._pending_next_turn_messages = []

            # Emit before_agent_start extension event
            if self._extension_runner and self._base_system_prompt_options is not None:
                result = await self._extension_runner.emit_before_agent_start(
                    expanded_text,
                    current_images,
                    self._base_system_prompt,
                    self._base_system_prompt_options,
                )
                if result and result.messages:  # type: ignore[attr-defined]
                    for msg in result.messages:  # type: ignore[attr-defined]
                        messages.append(
                            {  # type: ignore[arg-type]
                                "role": "custom",
                                "custom_type": msg.custom_type,
                                "content": msg.content or [],
                                "display": msg.display,
                                "details": msg.details,
                                "timestamp": int(time.time() * 1000),
                            }
                        )
                if result and result.system_prompt is not None:  # type: ignore[attr-defined]
                    self._system_prompt_override = result.system_prompt  # type: ignore[attr-defined]
                    self.agent.state.system_prompt = result.system_prompt  # type: ignore[attr-defined]
                else:
                    self._system_prompt_override = None
                    self.agent.state.system_prompt = self._base_system_prompt
        except Exception:
            if preflight_result:
                preflight_result(False)
            raise

        if messages is not None:
            if preflight_result:
                preflight_result(True)
            await self._run_agent_prompt(messages)

    async def _try_execute_extension_command(self, text: str) -> bool:
        """Try to execute an extension command. Returns True if command was found and executed."""
        space_index = text.find(" ")
        command_name = text[1:space_index] if space_index != -1 else text[1:]
        args = text[space_index + 1 :] if space_index != -1 else ""

        if not self._extension_runner:
            return False
        command = self._extension_runner.get_command(command_name)
        if not command:
            return False

        ctx = self._extension_runner.create_command_context()
        try:
            await command.handler(args, ctx)  # type: ignore[arg-type]
            return True
        except Exception as err:
            if self._extension_runner:
                self._extension_runner.emit_error(
                    ExtensionError(
                        extension_path=f"command:{command_name}",
                        event="command",
                        error=str(err),
                    )
                )
            return True

    def _expand_skill_command(self, text: str) -> str:
        """Expand skill commands (/skill:name args) to their full content."""
        if not text.startswith("/skill:"):
            return text

        space_index = text.find(" ")
        skill_name = text[7:space_index] if space_index != -1 else text[7:]
        args = text[space_index + 1 :].strip() if space_index != -1 else ""

        if not self._resource_loader:
            return text
        skills = self._resource_loader.get_skills()["skills"]
        skill = next((s for s in skills if s.name == skill_name), None)
        if not skill:
            return text

        try:
            from pathlib import Path

            content = Path(skill.file_path).read_text(encoding="utf-8")
            from ..utils.frontmatter import (  # type: ignore[import-untyped]
                strip_frontmatter,
            )

            body = strip_frontmatter(content).strip()
            skill_block = (
                f'<skill name="{skill.name}" location="{skill.file_path}">\n'
                f"References are relative to {skill.base_dir}.\n\n"
                f"{body}\n"
                f"</skill>"
            )
            return f"{skill_block}\n\n{args}" if args else skill_block
        except Exception as err:
            if self._extension_runner:
                self._extension_runner.emit_error(
                    ExtensionError(
                        extension_path=skill.file_path,
                        event="skill_expansion",
                        error=str(err),
                    )
                )
            return text

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> None:
        """Queue a steering message while the agent is running."""
        if text.startswith("/"):
            self._throw_if_extension_command(text)
        expanded_text = self._expand_skill_command(text)
        expanded_text = expand_prompt_template(
            expanded_text, list(self.prompt_templates)
        )
        await self._queue_steer(expanded_text, images)

    async def follow_up(
        self, text: str, images: list[ImageContent] | None = None
    ) -> None:
        """Queue a follow-up message to be processed after the agent finishes."""
        if text.startswith("/"):
            self._throw_if_extension_command(text)
        expanded_text = self._expand_skill_command(text)
        expanded_text = expand_prompt_template(
            expanded_text, list(self.prompt_templates)
        )
        await self._queue_follow_up(expanded_text, images)

    async def _queue_steer(
        self, text: str, images: list[ImageContent] | None = None
    ) -> None:
        """Internal: Queue a steering message."""
        self._steering_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [{"type": "text", "text": text}]  # type: ignore[list-item]
        if images:
            content.extend(images)
        self.agent.steer(
            {  # type: ignore[arg-type]
                "role": "user",
                "content": content,
                "timestamp": int(time.time() * 1000),
            }
        )

    async def _queue_follow_up(
        self, text: str, images: list[ImageContent] | None = None
    ) -> None:
        """Internal: Queue a follow-up message."""
        self._follow_up_messages.append(text)
        self._emit_queue_update()
        content: list[TextContent | ImageContent] = [{"type": "text", "text": text}]  # type: ignore[list-item]
        if images:
            content.extend(images)
        self.agent.follow_up(
            {  # type: ignore[arg-type]
                "role": "user",
                "content": content,
                "timestamp": int(time.time() * 1000),
            }
        )

    def _throw_if_extension_command(self, text: str) -> None:
        """Throw an error if the text is an extension command."""
        space_index = text.find(" ")
        command_name = text[1:space_index] if space_index != -1 else text[1:]
        if self._extension_runner:
            command = self._extension_runner.get_command(command_name)
            if command:
                raise RuntimeError(
                    f'Extension command "/{command_name}" cannot be queued. '
                    f"Use prompt() or execute the command when not streaming."
                )

    async def send_custom_message(
        self,
        message: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send a custom message to the session."""
        app_message = {
            "role": "custom",
            "custom_type": message.get("custom_type", ""),
            "content": message.get("content") or [],
            "display": message.get("display"),
            "details": message.get("details"),
            "timestamp": int(time.time() * 1000),
        }
        deliver_as = options.get("deliver_as") if options else None
        trigger_turn = options.get("trigger_turn", False) if options else False

        if deliver_as == "next_turn":
            self._pending_next_turn_messages.append(app_message)  # type: ignore[arg-type]
        elif self.is_streaming:
            if deliver_as == "follow_up":
                self.agent.follow_up(app_message)  # type: ignore[arg-type]
            else:
                self.agent.steer(app_message)  # type: ignore[arg-type]
        elif trigger_turn:
            await self._run_agent_prompt(app_message)  # type: ignore[arg-type]
        else:
            self.agent.state.messages.append(app_message)  # type: ignore[arg-type]
            self.session_manager.append_custom_message_entry(
                message.get("custom_type", ""),
                message.get("content") or [],
                message.get("display"),  # type: ignore[arg-type]
                message.get("details"),
            )
            self._emit({"type": "message_start", "message": app_message})  # type: ignore[arg-type]
            self._emit({"type": "message_end", "message": app_message})  # type: ignore[arg-type]

    async def send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
        options: dict[str, Any] | None = None,
    ) -> None:
        """Send a user message to the agent."""
        text: str
        images: list[ImageContent] | None = None

        if isinstance(content, str):
            text = content
        else:
            text_parts: list[str] = []
            images = []
            for part in content:
                if part.get("type") == "text":  # type: ignore[union-attr]
                    text_parts.append(part.get("text", ""))  # type: ignore[union-attr]
                else:
                    images.append(part)  # type: ignore[arg-type]
            text = "\n".join(text_parts)
            if not images:
                images = None

        deliver_as = options.get("deliver_as") if options else None
        await self.prompt(
            text,
            PromptOptions(
                expand_prompt_templates=False,
                streaming_behavior=deliver_as,
                images=images,
                source="extension",
            ),
        )

    def clear_queue(self) -> dict[str, list[str]]:
        """Clear all queued messages and return them."""
        steering = list(self._steering_messages)
        follow_up = list(self._follow_up_messages)
        self._steering_messages = []
        self._follow_up_messages = []
        self.agent.clear_all_queues()
        self._emit_queue_update()
        return {"steering": steering, "follow_up": follow_up}

    @property
    def pending_message_count(self) -> int:
        """Number of pending messages."""
        return len(self._steering_messages) + len(self._follow_up_messages)

    def get_steering_messages(self) -> Sequence[str]:
        """Get pending steering messages (read-only)."""
        return self._steering_messages

    def get_follow_up_messages(self) -> Sequence[str]:
        """Get pending follow-up messages (read-only)."""
        return self._follow_up_messages

    @property
    def resource_loader(self) -> Any:
        return self._resource_loader

    async def abort(self) -> None:
        """Abort current operation and wait for agent to become idle."""
        self.abort_retry()
        self.agent.abort()
        await self.wait_for_idle()

    async def wait_for_idle(self) -> None:
        """Wait for the agent to become idle."""
        if self.is_idle:
            return
        await self._idle_wait_event.wait()

    # =========================================================================
    # Model Management
    # =========================================================================

    async def _emit_model_select(
        self,
        next_model: Model,
        previous_model: Model | None,
        source: Literal["set", "cycle", "restore"],
    ) -> None:
        """Emit model select event."""
        if models_are_equal(previous_model, next_model):
            return
        if self._extension_runner:
            await self._extension_runner.emit(
                {  # type: ignore[arg-type]
                    "type": "model_select",
                    "model": next_model,
                    "previous_model": previous_model,
                    "source": source,
                }
            )

    async def set_model(self, model: Model) -> None:
        """Set model directly."""
        if not (await self._model_runtime.check_auth(model.provider)):
            raise ValueError(f"No API key for {model.provider}/{model.model_id}")

        previous_model = self.model
        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = model
        self.session_manager.append_model_change(model.provider, model.model_id)
        self.settings_manager.set_default_provider(model.provider)
        self.settings_manager.set_default_model(model.model_id)

        # Re-clamp thinking level for new model's capabilities
        self.set_thinking_level(thinking_level)

        await self._emit_model_select(model, previous_model, "set")

    async def cycle_model(
        self, direction: Literal["forward", "backward"] = "forward"
    ) -> ModelCycleResult | None:
        """Cycle to next/previous model."""
        if self._scoped_models:
            return await self._cycle_scoped_model(direction)
        return await self._cycle_available_model(direction)

    async def _cycle_scoped_model(
        self, direction: Literal["forward", "backward"]
    ) -> ModelCycleResult | None:
        """Cycle through scoped models."""
        available_ids = {
            f"{model.provider}\0{model.model_id}"
            for model in self._model_runtime.get_available_snapshot()
        }
        scoped_models = [
            sm
            for sm in self._scoped_models
            if f"{sm['model'].provider}\0{sm['model'].model_id}" in available_ids
        ]
        if len(scoped_models) <= 1:
            return None

        current_model = self.model
        current_index = next(
            (
                i
                for i, sm in enumerate(scoped_models)
                if models_are_equal(sm["model"], current_model)
            ),
            -1,
        )
        if current_index == -1:
            current_index = 0
        length = len(scoped_models)
        next_index = (
            (current_index + 1) % length
            if direction == "forward"
            else (current_index - 1 + length) % length
        )
        next_scoped = scoped_models[next_index]
        thinking_level = self._get_thinking_level_for_model_switch(
            next_scoped.get("thinking_level")
        )

        self.agent.state.model = next_scoped["model"]
        self.session_manager.append_model_change(
            next_scoped["model"].provider, next_scoped["model"].model_id
        )
        self.settings_manager.set_default_provider(next_scoped["model"].provider)
        self.settings_manager.set_default_model(next_scoped["model"].model_id)
        self.set_thinking_level(thinking_level)

        await self._emit_model_select(next_scoped["model"], current_model, "cycle")
        return ModelCycleResult(
            model=next_scoped["model"],
            thinking_level=self.thinking_level,
            is_scoped=True,
        )

    async def _cycle_available_model(
        self, direction: Literal["forward", "backward"]
    ) -> ModelCycleResult | None:
        """Cycle through all available models."""
        available_models = self._model_runtime.get_available_snapshot()
        if len(available_models) <= 1:
            return None

        current_model = self.model
        current_index = next(
            (
                i
                for i, m in enumerate(available_models)
                if models_are_equal(m, current_model)
            ),
            -1,
        )
        if current_index == -1:
            current_index = 0
        length = len(available_models)
        next_index = (
            (current_index + 1) % length
            if direction == "forward"
            else (current_index - 1 + length) % length
        )
        next_model = available_models[next_index]

        thinking_level = self._get_thinking_level_for_model_switch()
        self.agent.state.model = next_model
        self.session_manager.append_model_change(
            next_model.provider, next_model.model_id
        )
        self.settings_manager.set_default_provider(next_model.provider)
        self.settings_manager.set_default_model(next_model.model_id)
        self.set_thinking_level(thinking_level)

        await self._emit_model_select(next_model, current_model, "cycle")
        return ModelCycleResult(
            model=next_model,
            thinking_level=self.thinking_level,
            is_scoped=False,
        )

    # =========================================================================
    # Thinking Level Management
    # =========================================================================

    def set_thinking_level(self, level: ThinkingLevel) -> None:
        """Set thinking level, clamped to model capabilities."""
        available_levels = self.get_available_thinking_levels()
        effective_level = (
            level
            if level in available_levels
            else self._clamp_thinking_level(level, available_levels)
        )

        previous_level = self.agent.state.thinking_level
        is_changing = effective_level != previous_level

        self.agent.state.thinking_level = effective_level

        if is_changing:
            self.session_manager.append_thinking_level_change(effective_level)
            if self.supports_thinking() or effective_level != "off":
                self.settings_manager.set_default_thinking_level(effective_level)
            self._emit(AgentSessionThinkingLevelChangedEvent(level=effective_level))
            if self._extension_runner:
                self._extension_runner.emit(  # type: ignore[unused-coroutine]
                    {  # type: ignore[arg-type]
                        "type": "thinking_level_select",
                        "level": effective_level,
                        "previous_level": previous_level,
                    }
                )

    def cycle_thinking_level(self) -> ThinkingLevel | None:
        """Cycle to next thinking level."""
        if not self.supports_thinking():
            return None
        levels = self.get_available_thinking_levels()
        current_index = levels.index(self.thinking_level)
        next_index = (current_index + 1) % len(levels)
        next_level = levels[next_index]
        self.set_thinking_level(next_level)
        return next_level

    def get_available_thinking_levels(self) -> list[ThinkingLevel]:
        """Get available thinking levels for current model."""
        if not self.model:
            return THINKING_LEVELS
        return get_supported_thinking_levels(self.model)

    def supports_thinking(self) -> bool:
        """Check if current model supports thinking/reasoning."""
        if self.model is None:
            return False
        return bool(getattr(self.model, "reasoning", False))

    def _get_thinking_level_for_model_switch(
        self, explicit_level: ThinkingLevel | None = None
    ) -> ThinkingLevel:
        """Get the thinking level to use when switching models."""
        if explicit_level is not None:
            return explicit_level
        if not self.supports_thinking():
            return (
                self.settings_manager.get_default_thinking_level()
                or DEFAULT_THINKING_LEVEL
            )
        return self.thinking_level

    def _clamp_thinking_level(
        self, level: ThinkingLevel, available_levels: list[ThinkingLevel]
    ) -> ThinkingLevel:
        """Clamp thinking level to available levels."""
        if self.model:
            return clamp_thinking_level(self.model, level)
        return "off"

    # =========================================================================
    # Queue Mode Management
    # =========================================================================

    def _sync_queue_modes_from_settings(self) -> None:
        """Sync queue modes from settings."""
        self.agent.steering_mode = self.settings_manager.get_steering_mode()
        self.agent.follow_up_mode = self.settings_manager.get_follow_up_mode()

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        """Set steering message mode."""
        self.agent.steering_mode = mode
        self.settings_manager.set_steering_mode(mode)

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        """Set follow-up message mode."""
        self.agent.follow_up_mode = mode
        self.settings_manager.set_follow_up_mode(mode)

    # =========================================================================
    # Compaction
    # =========================================================================

    async def compact(self, custom_instructions: str | None = None) -> CompactionResult:
        """Manually compact the session context."""
        await self.abort()
        self._compaction_abort_event = asyncio.Event()
        self._emit(AgentSessionCompactionStartEvent(reason="manual"))

        try:
            if not self.model:
                raise ValueError(format_no_model_selected_message())

            auth = await self._get_summarization_request_auth(self.model)
            request_model = auth["model"]
            api_key = auth.get("api_key")
            headers = auth.get("headers")
            env = auth.get("env")

            path_entries = self.session_manager.get_branch()
            settings = self.settings_manager.get_compaction_settings()

            preparation = prepare_compaction(path_entries, settings)
            if not preparation:
                last_entry = path_entries[-1] if path_entries else None
                if last_entry and last_entry.type == "compaction":
                    raise RuntimeError("Already compacted")
                raise RuntimeError("Nothing to compact (session too small)")

            extension_compaction: CompactionResult | None = None
            from_extension = False

            if self._extension_runner and self._extension_runner.has_handlers(
                "session_before_compact"
            ):
                result = await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_before_compact",
                        "preparation": preparation,
                        "branch_entries": path_entries,
                        "custom_instructions": custom_instructions,
                        "reason": "manual",
                        "will_retry": False,
                        "signal": self._compaction_abort_event,
                    }
                )
                if isinstance(result, SessionBeforeCompactResult):
                    if result.cancel:
                        raise RuntimeError("Compaction cancelled")
                    if result.compaction:
                        extension_compaction = result.compaction
                        from_extension = True

            summary: str
            first_kept_entry_id: str
            tokens_before: int
            usage: Usage | None = None
            details: Any = None

            if extension_compaction:
                summary = extension_compaction.summary
                first_kept_entry_id = extension_compaction.first_kept_entry_id
                tokens_before = extension_compaction.tokens_before
                usage = extension_compaction.usage
                details = extension_compaction.details
            else:
                compact_result = await compact(
                    preparation,
                    request_model,
                    api_key=api_key,
                    headers=headers,
                    custom_instructions=custom_instructions,
                    signal=self._compaction_abort_event,
                    thinking_level=self.thinking_level,
                    stream_fn=self.agent.stream_function,  # type: ignore[attr-defined]
                    env=env,
                    retry=self.settings_manager.get_retry_settings(),  # type: ignore[arg-type]
                    callbacks=self._summarization_retry_callbacks(
                        {"source": "compaction", "reason": "manual"}
                    ),
                )
                summary = compact_result.summary
                first_kept_entry_id = compact_result.first_kept_entry_id
                tokens_before = compact_result.tokens_before
                usage = compact_result.usage
                details = compact_result.details

            if self._compaction_abort_event.is_set():
                raise RuntimeError("Compaction cancelled")

            self.session_manager.append_compaction(
                summary,
                first_kept_entry_id,
                tokens_before,
                details,
                from_extension,
                usage,
            )
            new_entries = self.session_manager.get_entries()
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = estimate_messages_tokens(session_context.messages)

            saved_compaction_entry = next(
                (
                    e
                    for e in new_entries
                    if e.type == "compaction" and e.summary == summary
                ),
                None,
            )

            if self._extension_runner and saved_compaction_entry:
                await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_compact",
                        "compaction_entry": saved_compaction_entry,
                        "from_extension": from_extension,
                        "reason": "manual",
                        "will_retry": False,
                    }
                )

            compaction_result = CompactionResult(
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                estimated_tokens_after=estimated_tokens_after,
                usage=usage,
                details=details,
            )
            self._emit(
                AgentSessionCompactionEndEvent(
                    reason="manual",
                    result=compaction_result,
                    aborted=False,
                    will_retry=False,
                )
            )
            return compaction_result
        except Exception as error:
            error_message = str(error)
            aborted = error_message == "Compaction cancelled"
            self._emit(
                AgentSessionCompactionEndEvent(
                    reason="manual",
                    result=None,
                    aborted=aborted,
                    will_retry=False,
                    error_message=None
                    if aborted
                    else f"Compaction failed: {error_message}",
                )
            )
            raise
        finally:
            self._compaction_abort_event = None

    def abort_compaction(self) -> None:
        """Cancel in-progress compaction."""
        if self._compaction_abort_event:
            self._compaction_abort_event.set()
        if self._auto_compaction_abort_event:
            self._auto_compaction_abort_event.set()

    def abort_branch_summary(self) -> None:
        """Cancel in-progress branch summarization."""
        if self._branch_summary_abort_event:
            self._branch_summary_abort_event.set()

    async def _check_compaction(
        self, assistant_message: AssistantMessage, skip_aborted_check: bool = True
    ) -> bool:
        """Check if compaction is needed and run it."""
        settings = self.settings_manager.get_compaction_settings()
        if not settings.enabled:
            return False

        if skip_aborted_check and assistant_message.stop_reason == "aborted":
            return False

        context_window = getattr(self.model, "context_window", 0) if self.model else 0

        same_model = (
            self.model is not None
            and assistant_message.provider == self.model.provider
            and assistant_message.model == self.model.model_id
        )

        compaction_entry = get_latest_compaction_entry(
            self.session_manager.get_branch()
        )
        assistant_is_from_before_compaction = False
        if compaction_entry is not None:
            ts = int(
                datetime.fromisoformat(compaction_entry.timestamp).timestamp() * 1000
            )
            assistant_is_from_before_compaction = assistant_message.timestamp <= ts
        if assistant_is_from_before_compaction:
            return False

        max_tokens = getattr(self.model, "max_tokens", 0) if self.model else 0
        recoverable_length = same_model and is_recoverable_length(
            assistant_message, max_tokens
        )
        if same_model and (
            is_context_overflow(assistant_message, context_window) or recoverable_length
        ):
            will_retry = assistant_message.stop_reason != "stop"

            if not will_retry:
                return await self._run_auto_compaction("overflow", False)

            if self._overflow_recovery_attempted:
                self._emit(
                    AgentSessionCompactionEndEvent(
                        reason="overflow",
                        result=None,
                        aborted=False,
                        will_retry=False,
                        error_message="Context overflow recovery failed after one compact-and-retry attempt.",
                    )
                )
                return False

            self._overflow_recovery_attempted = True
            messages = self.agent.state.messages
            if messages and messages[-1].role == "assistant":
                self.agent.state.messages = messages[:-1]
            return await self._run_auto_compaction("overflow", will_retry)

        # Case 2: Threshold
        context_tokens: int
        direct_context_tokens = (
            calculate_context_tokens(assistant_message.usage)
            if assistant_message.usage
            else 0
        )
        if assistant_message.stop_reason == "error" or direct_context_tokens == 0:
            estimate = estimate_context_tokens(self.agent.state.messages)
            if estimate.last_usage_index is None:
                return False
            usage_msg = self.agent.state.messages[estimate.last_usage_index]
            if (
                compaction_entry
                and usage_msg.role == "assistant"
                and usage_msg.timestamp
                <= int(
                    datetime.fromisoformat(compaction_entry.timestamp).timestamp()
                    * 1000
                )
            ):
                return False
            context_tokens = estimate.tokens
        else:
            context_tokens = direct_context_tokens

        if should_compact(context_tokens, context_window, settings):
            return await self._run_auto_compaction("threshold", False)
        return False

    async def _run_auto_compaction(
        self, reason: Literal["overflow", "threshold"], will_retry: bool
    ) -> bool:
        """Internal: Run auto-compaction with events."""
        settings = self.settings_manager.get_compaction_settings()
        started = False

        try:
            if not self.model:
                return False

            auth = await self._get_summarization_request_auth(self.model)
            request_model = auth["model"]
            api_key = auth.get("api_key")
            headers = auth.get("headers")
            env = auth.get("env")

            path_entries = self.session_manager.get_branch()

            preparation = prepare_compaction(path_entries, settings)
            if not preparation:
                return False

            self._emit(AgentSessionCompactionStartEvent(reason=reason))
            self._auto_compaction_abort_event = asyncio.Event()
            started = True

            extension_compaction: CompactionResult | None = None
            from_extension = False

            if self._extension_runner and self._extension_runner.has_handlers(
                "session_before_compact"
            ):
                extension_result = await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_before_compact",
                        "preparation": preparation,
                        "branch_entries": path_entries,
                        "custom_instructions": None,
                        "reason": reason,
                        "will_retry": will_retry,
                        "signal": self._auto_compaction_abort_event,
                    }
                )
                if isinstance(extension_result, SessionBeforeCompactResult):
                    if extension_result.cancel:
                        self._emit(
                            AgentSessionCompactionEndEvent(
                                reason=reason,
                                result=None,
                                aborted=True,
                                will_retry=False,
                            )
                        )
                        return False
                    if extension_result.compaction:
                        extension_compaction = extension_result.compaction
                        from_extension = True

            summary: str
            first_kept_entry_id: str
            tokens_before: int
            usage: Usage | None = None
            details: Any = None

            if extension_compaction:
                summary = extension_compaction.summary
                first_kept_entry_id = extension_compaction.first_kept_entry_id
                tokens_before = extension_compaction.tokens_before
                usage = extension_compaction.usage
                details = extension_compaction.details
            else:
                compact_result = await compact(
                    preparation,
                    request_model,
                    api_key=api_key,
                    headers=headers,
                    custom_instructions=None,
                    signal=self._auto_compaction_abort_event,
                    thinking_level=self.thinking_level,
                    stream_fn=self.agent.stream_function,  # type: ignore[attr-defined]
                    env=env,
                    retry=self.settings_manager.get_retry_settings(),  # type: ignore[arg-type]
                    callbacks=self._summarization_retry_callbacks(
                        {"source": "compaction", "reason": reason}
                    ),
                )
                summary = compact_result.summary
                first_kept_entry_id = compact_result.first_kept_entry_id
                tokens_before = compact_result.tokens_before
                usage = compact_result.usage
                details = compact_result.details

            if self._auto_compaction_abort_event.is_set():
                self._emit(
                    AgentSessionCompactionEndEvent(
                        reason=reason,
                        result=None,
                        aborted=True,
                        will_retry=False,
                    )
                )
                return False

            self.session_manager.append_compaction(
                summary,
                first_kept_entry_id,
                tokens_before,
                details,
                from_extension,
                usage,
            )
            new_entries = self.session_manager.get_entries()
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages
            estimated_tokens_after = estimate_messages_tokens(session_context.messages)

            saved_compaction_entry = next(
                (
                    e
                    for e in new_entries
                    if e.type == "compaction" and e.summary == summary
                ),
                None,
            )

            if self._extension_runner and saved_compaction_entry:
                await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_compact",
                        "compaction_entry": saved_compaction_entry,
                        "from_extension": from_extension,
                        "reason": reason,
                        "will_retry": will_retry,
                    }
                )

            result = CompactionResult(
                summary=summary,
                first_kept_entry_id=first_kept_entry_id,
                tokens_before=tokens_before,
                estimated_tokens_after=estimated_tokens_after,
                usage=usage,
                details=details,
            )
            self._emit(
                AgentSessionCompactionEndEvent(
                    reason=reason,
                    result=result,
                    aborted=False,
                    will_retry=will_retry,
                )
            )

            if will_retry:
                messages = self.agent.state.messages
                last_msg = messages[-1] if messages else None
                if (
                    last_msg
                    and last_msg.role == "assistant"
                    and last_msg.stop_reason in ("error", "length")
                ):
                    self.agent.state.messages = messages[:-1]
                return True

            return self.agent.has_queued_messages()
        except Exception as error:
            error_message = str(error)
            if started:
                prefix = (
                    "Context overflow recovery failed"
                    if reason == "overflow"
                    else "Auto-compaction failed"
                )
                self._emit(
                    AgentSessionCompactionEndEvent(
                        reason=reason,
                        result=None,
                        aborted=False,
                        will_retry=False,
                        error_message=f"{prefix}: {error_message}",
                    )
                )
            return False
        finally:
            self._auto_compaction_abort_event = None

    def set_auto_compaction_enabled(self, enabled: bool) -> None:
        """Toggle auto-compaction setting."""
        self.settings_manager.set_compaction_enabled(enabled)

    @property
    def auto_compaction_enabled(self) -> bool:
        """Whether auto-compaction is enabled."""
        return self.settings_manager.get_compaction_enabled()

    async def bind_extensions(self, bindings: ExtensionBindings) -> None:
        """Bind extensions to the session."""
        if bindings.ui_context is not None:
            self._extension_ui_context = bindings.ui_context
        if bindings.mode is not None:
            self._extension_mode = bindings.mode
        if bindings.command_context_actions is not None:
            self._extension_command_context_actions = bindings.command_context_actions
        if bindings.abort_handler is not None:
            self._extension_abort_handler = bindings.abort_handler
        if bindings.shutdown_handler is not None:
            self._extension_shutdown_handler = bindings.shutdown_handler
        if bindings.on_error is not None:
            self._extension_error_listener = bindings.on_error

        if self._extension_runner:
            self._apply_extension_bindings(self._extension_runner)
            await self._extension_runner.emit(self._session_start_event)
            reason = (
                "reload" if self._session_start_event.reason == "reload" else "startup"
            )
            await self._extend_resources_from_extensions(reason)  # type: ignore[arg-type]

    async def _extend_resources_from_extensions(
        self, reason: Literal["startup", "reload"]
    ) -> None:
        """Extend resources from extensions."""
        if not self._extension_runner or not self._extension_runner.has_handlers(
            "resources_discover"
        ):
            return

        discovered = await self._extension_runner.emit_resources_discover(
            self._cwd, reason
        )
        skill_paths = discovered.get("skill_paths", [])
        prompt_paths = discovered.get("prompt_paths", [])
        theme_paths = discovered.get("theme_paths", [])

        if not skill_paths and not prompt_paths and not theme_paths:
            return

        from os.path import dirname

        def build_extension_resource_paths(
            entries: list[dict[str, str]],
        ) -> list[dict[str, Any]]:
            result: list[dict[str, Any]] = []
            for entry in entries:
                source = self._get_extension_source_label(entry["extension_path"])
                base_dir = (
                    None
                    if entry["extension_path"].startswith("<")
                    else dirname(entry["extension_path"])
                )
                result.append(
                    {
                        "path": entry["path"],
                        "metadata": {
                            "source": source,
                            "scope": "temporary",
                            "origin": "top-level",
                            "base_dir": base_dir,
                        },
                    }
                )
            return result

        extension_paths: ResourceExtensionPaths = {  # type: ignore[assignment]
            "skill_paths": build_extension_resource_paths(skill_paths),
            "prompt_paths": build_extension_resource_paths(prompt_paths),
            "theme_paths": build_extension_resource_paths(theme_paths),
        }

        if self._resource_loader:
            self._resource_loader.extend_resources(extension_paths)
            self._base_system_prompt = self._rebuild_system_prompt(
                self.get_active_tool_names()
            )
            self.agent.state.system_prompt = self._base_system_prompt

    def _get_extension_source_label(self, extension_path: str) -> str:
        """Get a human-readable label for an extension source."""
        if extension_path.startswith("<"):
            return f"extension:{extension_path.replace('<', '').replace('>', '')}"
        base = os.path.basename(extension_path)
        name = re.sub(r"\.(ts|js)$", "", base)
        return f"extension:{name}"

    def _apply_extension_bindings(self, runner: ExtensionRunner) -> None:
        """Apply extension bindings to a runner."""
        runner.set_ui_context(self._extension_ui_context, self._extension_mode)
        if self._extension_command_context_actions:
            runner.bind_command_context(self._extension_command_context_actions)

        if self._extension_error_unsubscriber:
            self._extension_error_unsubscriber()
        self._extension_error_unsubscriber = (
            runner.on_error(self._extension_error_listener)
            if self._extension_error_listener
            else None
        )

    def _refresh_current_model_from_registry(self) -> None:
        """Refresh the current model from the registry."""
        current_model = self.model
        if not current_model:
            return
        refreshed_model = self._model_runtime.get_model(
            current_model.provider, current_model.model_id
        )
        if not refreshed_model or refreshed_model is current_model:
            return
        self.agent.state.model = refreshed_model

    def _bind_extension_core(self, runner: ExtensionRunner) -> None:
        """Bind core extension API to the runner."""

        def get_commands() -> list[SlashCommandInfo]:
            extension_commands: list[SlashCommandInfo] = [
                SlashCommandInfo(
                    name=cmd.invocation_name,
                    description=cmd.description,
                    source="extension",
                    source_info=cmd.source_info,
                )
                for cmd in runner.get_registered_commands()
            ]

            templates: list[SlashCommandInfo] = [
                SlashCommandInfo(
                    name=t.name,
                    description=t.description,
                    source="prompt",
                    source_info=t.source_info,  # type: ignore[arg-type]
                )
                for t in self.prompt_templates
            ]

            skills_list: list[SlashCommandInfo] = []
            if self._resource_loader:
                skills_list = [
                    SlashCommandInfo(
                        name=f"skill:{s.name}",
                        description=s.description,
                        source="skill",
                        source_info=s.source_info,
                    )
                    for s in self._resource_loader.get_skills()["skills"]
                ]

            return extension_commands + templates + skills_list

        def _send_message_wrapper(
            message: dict[str, Any], options: dict[str, Any] | None = None
        ) -> None:
            asyncio.ensure_future(self.send_custom_message(message, options))

        def _send_user_message_wrapper(
            content: str | list[TextContent | ImageContent],
            options: dict[str, Any] | None = None,
        ) -> None:
            asyncio.ensure_future(self.send_user_message(content, options))

        runner.bind_core(
            {  # type: ignore[arg-type]
                "send_message": _send_message_wrapper,
                "send_user_message": _send_user_message_wrapper,
                "append_entry": lambda custom_type, data=None: self._append_entry(
                    custom_type, data
                ),
                "set_session_name": lambda name: self.set_session_name(name),
                "get_session_name": lambda: self.session_manager.get_session_name(),
                "set_label": lambda entry_id, label: (
                    self.session_manager.append_label_change(entry_id, label)
                ),
                "get_active_tools": lambda: self.get_active_tool_names(),
                "get_all_tools": lambda: self.get_all_tools(),
                "set_active_tools": lambda tool_names: self.set_active_tools_by_name(
                    tool_names
                ),
                "refresh_tools": lambda: self._refresh_tool_registry(),
                "get_commands": get_commands,
                "set_model": lambda model: self._set_model_wrapper(model),
                "get_thinking_level": lambda: self.thinking_level,
                "set_thinking_level": lambda level: self.set_thinking_level(level),
            },
            {  # type: ignore[arg-type]
                "get_model": lambda: self.model,
                "get_scoped_models": lambda: self._scoped_models,
                "is_idle": lambda: self.is_idle,
                "is_project_trusted": lambda: (
                    self.settings_manager.is_project_trusted()
                ),
                "get_signal": lambda: self.agent.signal,
                "abort": lambda: self._abort_wrapper(),
                "has_pending_messages": lambda: self.pending_message_count > 0,
                "shutdown": lambda: (
                    self._extension_shutdown_handler()
                    if self._extension_shutdown_handler
                    else None
                ),
                "get_context_usage": lambda: self.get_context_usage(),
                "compact": lambda options=None: self._compact_wrapper(options),  # type: ignore[misc]
                "get_system_prompt": lambda: self.system_prompt,
                "get_system_prompt_options": lambda: self._base_system_prompt_options,
            },
            {
                "register_provider": lambda name, config: self._register_provider(
                    name, config
                ),
                "register_native_provider": lambda provider: (
                    self._register_native_provider(provider)
                ),
                "unregister_provider": lambda name: self._unregister_provider(name),
            },
        )

    def _append_entry(self, custom_type: str, data: Any = None) -> None:
        """Append a custom entry and emit event."""
        entry_id = self.session_manager.append_custom_entry(custom_type, data)
        entry = self.session_manager.get_entry(entry_id)
        if entry:
            self._emit(AgentSessionEntryAppendedEvent(entry=entry))

    async def _set_model_wrapper(self, model: Model) -> bool:
        """Wrapper for extension set_model."""
        if not self._model_runtime.has_configured_auth(model.provider):
            return False
        await self.set_model(model)
        return True

    def _abort_wrapper(self) -> None:
        """Wrapper for extension abort."""
        if self._extension_abort_handler:
            self._extension_abort_handler()
            return
        asyncio.ensure_future(self.abort())

    def _compact_wrapper(self, options: CompactOptions | None = None) -> None:
        """Wrapper for extension compact."""

        async def _do_compact() -> None:
            try:
                result = await self.compact(
                    options.custom_instructions if options else None
                )
                if options and options.on_complete:
                    options.on_complete(result)
            except Exception as error:
                if options and options.on_error:
                    options.on_error(error)

        asyncio.ensure_future(_do_compact())

    def _register_provider(self, name: str, config: Any) -> None:
        """Register a provider."""
        self._model_runtime.register_provider(name, config)
        self._refresh_current_model_from_registry()

    def _register_native_provider(self, provider: Any) -> None:
        """Register a native provider."""
        self._model_runtime.register_native_provider(provider)
        self._refresh_current_model_from_registry()

    def _unregister_provider(self, name: str) -> None:
        """Unregister a provider."""
        self._model_runtime.unregister_provider(name)
        self._refresh_current_model_from_registry()

    def _refresh_tool_registry(
        self,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Refresh the tool registry."""
        opts = options or {}
        previous_registry_names = set(self._tool_registry.keys())
        previous_active_tool_names = self.get_active_tool_names()
        allowed_tool_names = self._allowed_tool_names
        excluded_tool_names = self._excluded_tool_names

        def is_allowed_tool(name: str) -> bool:
            return (not allowed_tool_names or name in allowed_tool_names) and (
                not excluded_tool_names or name not in excluded_tool_names
            )

        registered_tools = (
            self._extension_runner.get_all_registered_tools()
            if self._extension_runner
            else []
        )
        all_custom_tools: list[dict[str, Any]] = list(registered_tools)  # type: ignore[arg-type]
        all_custom_tools.extend(
            {
                "definition": t,
                "source_info": create_synthetic_source_info(
                    f"<sdk:{t.name}>", {"source": "sdk"}
                ),
            }
            for t in self._custom_tools
            if is_allowed_tool(t.name)
        )

        definition_registry: dict[str, ToolDefinitionEntry] = {
            name: ToolDefinitionEntry(
                definition=definition,
                source_info=create_synthetic_source_info(
                    f"<builtin:{name}>", {"source": "builtin"}
                ),
            )
            for name, definition in self._base_tool_definitions.items()
            if is_allowed_tool(name)
        }
        for tool in all_custom_tools:
            definition_registry[tool["definition"].name] = ToolDefinitionEntry(
                definition=tool["definition"],
                source_info=tool["source_info"],
            )

        self._tool_definitions = definition_registry
        self._tool_prompt_snippets = {
            entry.definition.name: snippet
            for entry in definition_registry.values()
            if (
                snippet := self._normalize_prompt_snippet(
                    entry.definition.prompt_snippet
                )
            )
        }
        self._tool_prompt_guidelines = {
            entry.definition.name: guidelines
            for entry in definition_registry.values()
            if (
                guidelines := self._normalize_prompt_guidelines(
                    entry.definition.prompt_guidelines
                )
            )
        }

        if self._extension_runner:
            wrapped_extension_tools = wrap_registered_tools(
                [
                    RegisteredTool(
                        definition=(
                            t["definition"].model_dump()
                            if isinstance(t, dict)
                            and hasattr(t.get("definition"), "model_dump")
                            else t["definition"]
                            if isinstance(t, dict)
                            else t.definition
                        ),
                        source_info=(
                            t["source_info"] if isinstance(t, dict) else t.source_info
                        ),
                    )
                    if isinstance(t, dict)
                    else t
                    for t in all_custom_tools
                ],
                self._extension_runner,
            )
            wrapped_built_in_tools = wrap_registered_tools(
                [
                    RegisteredTool(
                        definition=d.model_dump() if hasattr(d, "model_dump") else d,
                        source_info=create_synthetic_source_info(
                            f"<builtin:{d.name}>", {"source": "builtin"}
                        ),
                    )
                    for d in self._base_tool_definitions.values()
                    if is_allowed_tool(d.name)
                ],
                self._extension_runner,
            )

            tool_registry: dict[str, AgentTool] = {
                tool.name: tool for tool in wrapped_built_in_tools
            }
            for tool in wrapped_extension_tools:  # type: ignore[assignment]
                tool_registry[tool.name] = tool  # type: ignore[attr-defined, assignment]
            self._tool_registry = tool_registry
        else:
            self._tool_registry = {}

        next_active_tool_names = list(
            opts.get("active_tool_names", list(previous_active_tool_names))
        )
        next_active_tool_names = [
            n for n in next_active_tool_names if is_allowed_tool(n)
        ]

        if allowed_tool_names:
            for tool_name in self._tool_registry:
                if tool_name in allowed_tool_names:
                    next_active_tool_names.append(tool_name)
        elif opts.get("include_all_extension_tools"):
            if self._extension_runner:
                for tool in wrap_registered_tools(  # type: ignore[assignment]
                    all_custom_tools,  # type: ignore[arg-type]
                    self._extension_runner,
                ):
                    next_active_tool_names.append(tool.name)  # type: ignore[attr-defined]
        elif not opts.get("active_tool_names"):
            for tool_name in self._tool_registry:
                if tool_name not in previous_registry_names:
                    next_active_tool_names.append(tool_name)

        self.set_active_tools_by_name(list(dict.fromkeys(next_active_tool_names)))

    def _build_runtime(self, **options: Any) -> None:
        """Build the runtime environment."""
        if self._base_tools_override:
            base_tool_definitions = {
                name: create_tool_definition_from_agent_tool(tool)
                for name, tool in self._base_tools_override.items()
            }
        else:
            base_tool_definitions = create_all_tool_definitions(self._cwd)  # type: ignore[assignment]

        self._base_tool_definitions = base_tool_definitions  # type: ignore[assignment]

        if self._resource_loader:
            extensions_result = self._resource_loader.get_extensions()
            if options.get("flag_values"):
                for name, value in options["flag_values"].items():
                    extensions_result.runtime.flag_values[name] = value

            self._extension_runner = ExtensionRunner(
                extensions_result.extensions,
                extensions_result.runtime,
                self._cwd,
                self.session_manager,
                ModelRegistry(self._model_runtime),
            )
            if self._extension_runner_ref is not None:
                self._extension_runner_ref["current"] = self._extension_runner
            self._bind_extension_core(self._extension_runner)
            self._apply_extension_bindings(self._extension_runner)

        default_active_tool_names = (
            list(self._base_tools_override.keys())
            if self._base_tools_override
            else ["read", "bash", "edit", "write"]
        )
        base_active_tool_names = options.get(
            "active_tool_names", default_active_tool_names
        )
        self._refresh_tool_registry(
            {
                "active_tool_names": base_active_tool_names,
                "include_all_extension_tools": options.get(
                    "include_all_extension_tools", False
                ),
            }
        )

    async def reload(self, options: dict[str, Any] | None = None) -> None:
        """Reload extensions, settings, and resources."""
        opts = options or {}
        before_session_start = opts.get("before_session_start")
        if self._extension_runner:
            previous_flag_values = self._extension_runner.get_flag_values()
            await emit_session_shutdown_event(
                self._extension_runner,
                SessionShutdownEvent(reason="reload"),
            )
        else:
            previous_flag_values = {}
        self.settings_manager.reload()
        self._sync_queue_modes_from_settings()
        reset_api_providers()
        if self._resource_loader:
            await self._resource_loader.reload()
        self._build_runtime(
            active_tool_names=self.get_active_tool_names(),
            flag_values=previous_flag_values,
            include_all_extension_tools=True,
        )

        has_bindings = (
            self._extension_ui_context is not None
            or self._extension_command_context_actions is not None
            or self._extension_shutdown_handler is not None
            or self._extension_error_listener is not None
        )
        if has_bindings and self._extension_runner:
            if before_session_start:
                await before_session_start()
            await self._extension_runner.emit(
                {"type": "session_start", "reason": "reload"}  # type: ignore[arg-type]
            )
            await self._extend_resources_from_extensions("reload")

    # =========================================================================
    # Auto-Retry
    # =========================================================================

    def _is_retryable_error(self, message: AssistantMessage) -> bool:
        """Check if an error is retryable."""
        context_window = getattr(self.model, "context_window", 0) if self.model else 0
        if is_context_overflow(message, context_window):
            return False
        return is_retryable_assistant_error(message)

    def _summarization_retry_callbacks(
        self,
        source: dict[str, Any],
    ) -> RetryCallbacks:
        """Create retry callbacks for summarization."""

        def on_retry_scheduled(
            attempt: int, max_attempts: int, delay_ms: int, error_message: str
        ) -> None:
            self._emit(
                AgentSessionSummarizationRetryScheduledEvent(
                    attempt=attempt,
                    max_attempts=max_attempts,
                    delay_ms=delay_ms,
                    error_message=error_message,
                )
            )

        def on_retry_attempt_start() -> None:
            if source.get("source") == "branchSummary":
                self._emit(AgentSessionSummarizationBranchRetryStartEvent())
            else:
                self._emit(
                    AgentSessionSummarizationCompactionRetryStartEvent(
                        source="compaction",
                        reason=source.get("reason", "manual"),
                    )
                )

        def on_retry_finished() -> None:
            self._emit(AgentSessionSummarizationRetryFinishedEvent())

        return RetryCallbacks(
            on_retry_scheduled=on_retry_scheduled,  # type: ignore[arg-type]
            on_retry_attempt_start=on_retry_attempt_start,  # type: ignore[arg-type]
            on_retry_finished=on_retry_finished,  # type: ignore[arg-type]
        )

    async def _prepare_retry(self, message: AssistantMessage) -> bool:
        """Prepare a retryable error for continuation with exponential backoff."""
        settings = self.settings_manager.get_retry_settings()
        if not (settings.enabled or False):
            return False

        self._retry_attempt += 1

        if self._retry_attempt > (settings.max_retries or 0):
            self._retry_attempt -= 1
            return False

        delay_ms = (settings.base_delay_ms or 1000) * (2 ** (self._retry_attempt - 1))

        self._emit(
            AgentSessionAutoRetryStartEvent(
                attempt=self._retry_attempt,
                max_attempts=settings.max_retries or 0,
                delay_ms=delay_ms,
                error_message=message.error_message or "Unknown error",
            )
        )

        # Remove error message from agent state
        messages = self.agent.state.messages
        if messages and messages[-1].role == "assistant":
            self.agent.state.messages = messages[:-1]

        # Wait with exponential backoff (abortable)
        self._retry_abort_event = asyncio.Event()
        try:
            await asyncio.sleep(delay_ms / 1000.0)
            if self._retry_abort_event.is_set():
                raise asyncio.CancelledError()
        except (asyncio.CancelledError, Exception):
            attempt = self._retry_attempt
            self._retry_attempt = 0
            self._emit(
                AgentSessionAutoRetryEndEvent(
                    success=False,
                    attempt=attempt,
                    final_error="Retry cancelled",
                )
            )
            return False
        finally:
            self._retry_abort_event = None

        return True

    def abort_retry(self) -> None:
        """Cancel in-progress retry."""
        if self._retry_abort_event:
            self._retry_abort_event.set()

    @property
    def is_retrying(self) -> bool:
        """Whether auto-retry is currently in progress."""
        return self._retry_abort_event is not None

    @property
    def auto_retry_enabled(self) -> bool:
        """Whether auto-retry is enabled."""
        return self.settings_manager.get_retry_enabled()

    def set_auto_retry_enabled(self, enabled: bool) -> None:
        """Toggle auto-retry setting."""
        self.settings_manager.set_retry_enabled(enabled)

    # =========================================================================
    # Bash Execution
    # =========================================================================

    async def execute_bash(
        self,
        command: str,
        on_chunk: Callable[[str], None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> BashResult:
        """Execute a bash command."""
        opts = options or {}
        abort_event = asyncio.Event()
        self._bash_abort_events.append(abort_event)

        resolved_command = command

        try:
            result = await execute_bash_with_operations(
                resolved_command,
                self.session_manager.get_cwd(),
                opts.get("operations", BashOperations()),
                {  # type: ignore[arg-type]
                    "on_chunk": lambda delta: (
                        on_chunk(delta) if on_chunk else None,
                        self._emit(  # type: ignore[func-returns-value]
                            AgentSessionBashExecutionUpdateEvent(
                                id=opts.get("id"),
                                delta=delta,
                            )
                        ),
                    ),
                    "signal": abort_event,
                },
            )

            self.record_bash_result(command, result, opts)
            return result
        finally:
            self._bash_abort_events.remove(abort_event)

    def record_bash_result(
        self,
        command: str,
        result: BashResult,
        options: dict[str, Any] | None = None,
    ) -> None:
        """Record a bash execution result in session history."""
        opts = options or {}
        bash_message = BashExecutionMessage(
            role="bashExecution",
            command=command,
            output=result.output,
            exit_code=result.exit_code,
            cancelled=result.cancelled,
            truncated=result.truncated,
            full_output_path=result.full_output_path,
            timestamp=int(time.time() * 1000),
            exclude_from_context=bool(opts.get("exclude_from_context")),
        )

        if self.is_streaming:
            self._pending_bash_messages.append(bash_message)
        else:
            self.agent.state.messages.append(bash_message)
            self.session_manager.append_message(bash_message)

    def abort_bash(self) -> None:
        """Cancel running bash command."""
        for event in list(self._bash_abort_events):
            event.set()

    @property
    def is_bash_running(self) -> bool:
        """Whether a bash command is currently running."""
        return len(self._bash_abort_events) > 0

    @property
    def has_pending_bash_messages(self) -> bool:
        """Whether there are pending bash messages."""
        return len(self._pending_bash_messages) > 0

    def _flush_pending_bash_messages(self) -> None:
        """Flush pending bash messages to agent state and session."""
        if not self._pending_bash_messages:
            return
        for bash_message in self._pending_bash_messages:
            self.agent.state.messages.append(bash_message)
            self.session_manager.append_message(bash_message)
        self._pending_bash_messages = []

    # =========================================================================
    # Session Management
    # =========================================================================

    def set_session_name(self, name: str) -> None:
        """Set a display name for the current session."""
        self.session_manager.append_session_info(name)
        event = AgentSessionInfoChangedEvent(
            name=self.session_manager.get_session_name()
        )
        self._emit(event)
        if self._extension_runner:
            self._extension_runner.emit(event)  # type: ignore[unused-coroutine, arg-type]

    # =========================================================================
    # Tree Navigation
    # =========================================================================

    async def navigate_tree(
        self,
        target_id: str,
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Navigate to a different node in the session tree."""
        opts = options or {}
        if self.is_streaming:
            raise RuntimeError(
                "Wait for the current response to finish before navigating the session tree."
            )

        old_leaf_id = self.session_manager.get_leaf_id()

        if target_id == old_leaf_id:
            return {"cancelled": False}

        if opts.get("summarize") and not self.model:
            raise ValueError("No model available for summarization")

        target_entry = self.session_manager.get_entry(target_id)
        if not target_entry:
            raise ValueError(f"Entry {target_id} not found")

        entries_result = collect_entries_for_branch_summary(
            self.session_manager,
            old_leaf_id,
            target_id,
        )
        entries_to_summarize = entries_result.entries
        common_ancestor_id = entries_result.common_ancestor_id

        custom_instructions = opts.get("custom_instructions")
        replace_instructions = opts.get("replace_instructions")
        label = opts.get("label")

        self._branch_summary_abort_event = asyncio.Event()

        try:
            extension_summary: dict[str, Any] | None = None
            from_extension = False

            if self._extension_runner and self._extension_runner.has_handlers(
                "session_before_tree"
            ):
                result = await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_before_tree",
                        "preparation": TreePreparation(
                            target_id=target_id,
                            old_leaf_id=old_leaf_id,
                            common_ancestor_id=common_ancestor_id,
                            entries_to_summarize=entries_to_summarize,
                            user_wants_summary=opts.get("summarize", False),
                            custom_instructions=custom_instructions,
                            replace_instructions=replace_instructions,
                            label=label,
                        ),
                        "signal": self._branch_summary_abort_event,
                    }
                )
                if isinstance(result, SessionBeforeTreeResult):
                    if result.cancel:
                        return {"cancelled": True}
                    if result.summary and opts.get("summarize"):
                        extension_summary = result.summary
                        from_extension = True
                    if result.custom_instructions is not None:
                        custom_instructions = result.custom_instructions
                    if result.replace_instructions is not None:
                        replace_instructions = result.replace_instructions
                    if result.label is not None:
                        label = result.label

            summary_text: str | None = None
            summary_details: Any = None
            summary_usage: Usage | None = None
            if opts.get("summarize") and entries_to_summarize and not extension_summary:
                model = self.model
                if model is None:
                    raise ValueError("No model available for summarization")
                auth = await self._get_summarization_request_auth(model)
                branch_summary_settings = (
                    self.settings_manager.get_branch_summary_settings()
                )
                branch_result = await generate_branch_summary(  # type: ignore[call-arg]
                    entries_to_summarize,
                    model=auth["model"],
                    api_key=auth.get("api_key"),
                    headers=auth.get("headers"),
                    env=auth.get("env"),
                    signal=self._branch_summary_abort_event,
                    custom_instructions=custom_instructions,
                    replace_instructions=replace_instructions,
                    reserve_tokens=branch_summary_settings.reserve_tokens or 0,
                    stream_fn=self.agent.stream_function,  # type: ignore[attr-defined]
                    retry=self.settings_manager.get_retry_settings(),
                    callbacks=self._summarization_retry_callbacks(
                        {"source": "branchSummary"}
                    ),
                )
                if branch_result.aborted:
                    return {"cancelled": True, "aborted": True}
                if branch_result.error:
                    raise RuntimeError(branch_result.error)
                summary_text = branch_result.summary
                summary_usage = branch_result.usage
                summary_details = {
                    "read_files": branch_result.read_files or [],
                    "modified_files": branch_result.modified_files or [],
                }
            elif extension_summary:
                summary_text = extension_summary.get("summary")
                summary_details = extension_summary.get("details")
                summary_usage = extension_summary.get("usage")

            # Determine new leaf position
            new_leaf_id: str | None
            editor_text: str | None = None

            if target_entry.type == "message" and target_entry.message.role == "user":
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.message.content, "")  # type: ignore[arg-type]
            elif target_entry.type == "custom_message":
                new_leaf_id = target_entry.parent_id
                editor_text = content_text(target_entry.content, "")  # type: ignore[arg-type]
            else:
                new_leaf_id = target_id

            # Switch leaf
            summary_entry: BranchSummaryEntry[Any] | None = None
            if summary_text:
                summary_id = self.session_manager.branch_with_summary(
                    new_leaf_id,
                    summary_text,
                    summary_details,
                    from_extension,
                    summary_usage,
                )
                summary_entry = self.session_manager.get_entry(summary_id)  # type: ignore[assignment]
                if label:
                    self.session_manager.append_label_change(summary_id, label)
            elif new_leaf_id is None:
                self.session_manager.reset_leaf()
            else:
                self.session_manager.branch(new_leaf_id)

            if label and not summary_text:
                self.session_manager.append_label_change(target_id, label)

            # Update agent state
            session_context = self.session_manager.build_session_context()
            self.agent.state.messages = session_context.messages

            if self._extension_runner:
                await self._extension_runner.emit(
                    {  # type: ignore[arg-type]
                        "type": "session_tree",
                        "new_leaf_id": self.session_manager.get_leaf_id(),
                        "old_leaf_id": old_leaf_id,
                        "summary_entry": summary_entry,
                        "from_extension": from_extension if summary_text else None,
                    }
                )

            return {
                "editor_text": editor_text,
                "cancelled": False,
                "summary_entry": summary_entry,
            }
        finally:
            self._branch_summary_abort_event = None

    def get_user_messages_for_forking(self) -> list[dict[str, str]]:
        """Get all user messages from session for fork selector."""
        entries = self.session_manager.get_entries()
        result: list[dict[str, str]] = []
        for entry in entries:
            if entry.type != "message":
                continue
            if entry.message.role != "user":
                continue
            text = content_text(entry.message.content, "")  # type: ignore[arg-type]
            if text:
                result.append({"entry_id": entry.id, "text": text})
        return result

    def get_session_stats(self) -> SessionStats:
        """Get session statistics."""
        user_messages = 0
        assistant_messages = 0
        tool_results = 0
        total_messages = 0
        tool_calls = 0
        usage_totals = create_usage_totals()

        for entry in self.session_manager.get_entries():
            if entry.type in ("branch_summary", "compaction") and entry.usage:
                add_usage_to_totals(usage_totals, entry.usage)
            if entry.type != "message":
                continue
            total_messages += 1
            message = entry.message
            if message.role == "user":
                user_messages += 1
            elif message.role == "toolResult":
                tool_results += 1
                if message.usage:
                    add_usage_to_totals(usage_totals, message.usage)
            elif message.role == "assistant":
                assistant_messages += 1
                if hasattr(message, "content") and isinstance(message.content, list):
                    tool_calls += sum(
                        1
                        for c in message.content
                        if getattr(c, "type", None) == "toolCall"
                    )
                if message.usage:
                    add_usage_to_totals(usage_totals, message.usage)

        total_tokens = (
            usage_totals.input
            + usage_totals.output
            + usage_totals.cache_read
            + usage_totals.cache_write
        )

        return SessionStats(
            session_file=self.session_file,
            session_id=self.session_id,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            tool_calls=tool_calls,
            tool_results=tool_results,
            total_messages=total_messages,
            tokens={
                "input": usage_totals.input,
                "output": usage_totals.output,
                "cache_read": usage_totals.cache_read,
                "cache_write": usage_totals.cache_write,
                "total": total_tokens,
            },
            cost=usage_totals.cost,
            context_usage=self.get_context_usage(),
        )

    def get_context_usage(self) -> ContextUsage | None:
        """Get context usage information."""
        model = self.model
        if not model:
            return None
        context_window = getattr(model, "context_window", 0)
        if context_window <= 0:
            return None

        branch_entries = self.session_manager.get_branch()
        latest_compaction = get_latest_compaction_entry(branch_entries)

        if latest_compaction:
            compaction_index = branch_entries.index(latest_compaction)
            has_post_compaction_usage = False
            for i in range(len(branch_entries) - 1, compaction_index, -1):
                entry = branch_entries[i]
                if entry.type == "message" and entry.message.role == "assistant":
                    assistant = entry.message
                    if assistant.stop_reason not in ("aborted", "error"):
                        context_tokens = calculate_context_tokens(assistant.usage)  # type: ignore[arg-type]
                        if context_tokens > 0:
                            has_post_compaction_usage = True
                            break
            if not has_post_compaction_usage:
                return ContextUsage(
                    tokens=None, context_window=context_window, percent=None
                )

        estimate = estimate_context_tokens(self.messages)
        percent = int((estimate.tokens / context_window) * 100)
        return ContextUsage(
            tokens=estimate.tokens,
            context_window=context_window,
            percent=percent,
        )

    async def export_to_html(self, output_path: str | None = None) -> str:
        """Export session to HTML."""
        configured_theme_name = self.settings_manager.get_theme()
        theme_name_result: str | None = None
        if configured_theme_name:
            try:
                from ..modes.interactive.theme.theme import (  # type: ignore[import-untyped]
                    get_theme_by_name,
                )

                if get_theme_by_name(configured_theme_name):
                    theme_name_result = configured_theme_name
            except ImportError:
                pass

        from ..modes.interactive.theme.theme import theme as theme_module

        tool_renderer = create_tool_html_renderer(
            get_tool_definition=lambda name: self.get_tool_definition(name),
            theme=theme_module,
            cwd=self.session_manager.get_cwd(),
        )

        return await export_session_to_html(  # type: ignore[no-any-return,attr-defined]
            self.session_manager,
            self.state,
            {
                "output_path": output_path,
                "theme_name": theme_name_result,
                "tool_renderer": tool_renderer,
            },
        )

    def export_to_jsonl(self, output_path: str | None = None) -> str:
        """Export the current session branch to a JSONL file."""
        from ..utils.paths import resolve_path  # type: ignore[import-untyped]

        file_path = resolve_path(
            output_path
            or f"session-{datetime.now().isoformat().replace(':', '-').replace('.', '-')}.jsonl",
            os.getcwd(),
        )
        dir_path = os.path.dirname(file_path)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)

        header = SessionHeader(
            type="session",
            version=CURRENT_SESSION_VERSION,
            id=self.session_manager.get_session_id(),
            timestamp=datetime.now().isoformat(),
            cwd=self.session_manager.get_cwd(),
        )

        branch_entries = self.session_manager.get_branch()
        lines: list[str] = [header.model_dump_json()]

        prev_id: str | None = None
        for entry in branch_entries:
            linear = entry.model_dump()
            linear["parent_id"] = prev_id
            lines.append(json.dumps(linear, default=str))
            prev_id = entry.id

        with open(file_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return file_path  # type: ignore[no-any-return]

    # =========================================================================
    # Utilities
    # =========================================================================

    def get_last_assistant_text(self) -> str | None:
        """Get text content of last assistant message."""
        last_assistant = None
        for m in reversed(self.messages):
            if m.role != "assistant":
                continue
            if m.stop_reason == "aborted" and len(m.content) == 0:
                continue
            last_assistant = m
            break

        if not last_assistant:
            return None

        text = ""
        for content in last_assistant.content:
            if content.type == "text":
                text += content.text
        return text.strip() or None

    # =========================================================================
    # Extension System
    # =========================================================================

    def create_replaced_session_context(self) -> ReplacedSessionContext:
        """Create a replaced session context."""
        if not self._extension_runner:
            raise RuntimeError("No extension runner available")
        ctx = self._extension_runner.create_command_context()
        ctx.send_message = lambda message, options=None: asyncio.ensure_future(  # type: ignore[attr-defined]
            self.send_custom_message(message, options)
        )
        ctx.send_user_message = lambda content, options=None: asyncio.ensure_future(  # type: ignore[attr-defined]
            self.send_user_message(content, options)
        )
        return ctx  # type: ignore[return-value]

    def has_extension_handlers(self, event_type: str) -> bool:
        """Check if extensions have handlers for a specific event type."""
        return (
            self._extension_runner is not None
            and self._extension_runner.has_handlers(event_type)
        )

    @property
    def extension_runner(self) -> ExtensionRunner | None:
        """Get the extension runner."""
        return self._extension_runner
