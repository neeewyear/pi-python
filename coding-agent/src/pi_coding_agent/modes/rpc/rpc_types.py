"""RPC protocol types for headless operation.

Commands are sent as JSON lines on stdin.
Responses and events are emitted as JSON lines on stdout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


# ============================================================================
# RPC Error
# ============================================================================


class RpcError(Exception):
    """RPC protocol error."""


# ============================================================================
# RPC Commands (stdin)
# ============================================================================

RpcCommand = (
    dict[Literal["type"], Literal["prompt"]]
    | dict[Literal["type"], Literal["steer"]]
    | dict[Literal["type"], Literal["follow_up"]]
    | dict[Literal["type"], Literal["abort"]]
    | dict[Literal["type"], Literal["new_session"]]
    | dict[Literal["type"], Literal["get_state"]]
    | dict[Literal["type"], Literal["set_model"]]
    | dict[Literal["type"], Literal["cycle_model"]]
    | dict[Literal["type"], Literal["get_available_models"]]
    | dict[Literal["type"], Literal["set_thinking_level"]]
    | dict[Literal["type"], Literal["cycle_thinking_level"]]
    | dict[Literal["type"], Literal["get_available_thinking_levels"]]
    | dict[Literal["type"], Literal["set_steering_mode"]]
    | dict[Literal["type"], Literal["set_follow_up_mode"]]
    | dict[Literal["type"], Literal["compact"]]
    | dict[Literal["type"], Literal["set_auto_compaction"]]
    | dict[Literal["type"], Literal["set_auto_retry"]]
    | dict[Literal["type"], Literal["abort_retry"]]
    | dict[Literal["type"], Literal["bash"]]
    | dict[Literal["type"], Literal["abort_bash"]]
    | dict[Literal["type"], Literal["get_session_stats"]]
    | dict[Literal["type"], Literal["export_html"]]
    | dict[Literal["type"], Literal["switch_session"]]
    | dict[Literal["type"], Literal["fork"]]
    | dict[Literal["type"], Literal["clone"]]
    | dict[Literal["type"], Literal["get_fork_messages"]]
    | dict[Literal["type"], Literal["get_entries"]]
    | dict[Literal["type"], Literal["get_tree"]]
    | dict[Literal["type"], Literal["get_last_assistant_text"]]
    | dict[Literal["type"], Literal["set_session_name"]]
    | dict[Literal["type"], Literal["get_messages"]]
    | dict[Literal["type"], Literal["get_commands"]]
)
"""RPC command union type."""


# ============================================================================
# RPC Slash Command (for get_commands response)
# ============================================================================


@dataclass
class RpcSlashCommand:
    """A command available for invocation via prompt."""

    name: str
    """Command name (without leading slash)."""
    description: str | None = None
    """Human-readable description."""
    source: str = ""
    """What kind of command this is (extension, prompt, skill)."""
    source_info: dict[str, Any] = field(default_factory=dict)
    """Source metadata for the owning resource."""


# ============================================================================
# RPC State
# ============================================================================


@dataclass
class RpcSessionState:
    """Current session state."""

    model: dict[str, Any] | None = None
    thinking_level: str = "off"
    is_streaming: bool = False
    is_compacting: bool = False
    steering_mode: str = "all"
    follow_up_mode: str = "all"
    session_file: str | None = None
    session_id: str = ""
    session_name: str | None = None
    auto_compaction_enabled: bool = False
    message_count: int = 0
    pending_message_count: int = 0


# ============================================================================
# RPC Responses (stdout)
# ============================================================================


RpcResponse = dict[str, Any]
"""RPC response type (success or error)."""


# ============================================================================
# Extension UI Events (stdout)
# ============================================================================


RpcExtensionUIRequest = dict[str, Any]
"""Extension UI request emitted to stdout."""


# ============================================================================
# Extension UI Commands (stdin)
# ============================================================================


RpcExtensionUIResponse = dict[str, Any]
"""Response to an extension UI request."""


# ============================================================================
# Helper type for extracting command types
# ============================================================================

RpcCommandType = str
"""RPC command type string."""