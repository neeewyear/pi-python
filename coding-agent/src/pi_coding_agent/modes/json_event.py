"""JSON event serialization for the JSON and RPC stdout protocols.

Strips cumulative assistant snapshots from streaming wire events.
``message_start`` provides the initial message, deltas build it, and
``message_end`` provides the final authoritative message.
"""

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any

from pi_coding_agent.core.agent_session import AgentSessionEvent

if TYPE_CHECKING:
    from pi_agent.types import MessageUpdateEvent

JsonAgentSessionEvent = AgentSessionEvent
"""Session event shape emitted by the JSON and RPC stdout protocols.

In the TypeScript original this is a mapped type that strips ``partial``
from ``message_update`` events. At the Python level we handle this at
serialization time instead.
"""


def _strip_partial_from_event(event: dict[str, Any]) -> dict[str, Any]:
    """Remove ``partial`` field from assistant_message_event if present."""
    if event.get("type") != "message_update":
        return event
    assistant_event = event.get("assistant_message_event")
    if isinstance(assistant_event, dict) and "partial" in assistant_event:
        event = dict(event)
        cleaned = {k: v for k, v in assistant_event.items() if k != "partial"}
        event["assistant_message_event"] = cleaned
    return event


def serialize_agent_session_event(event: AgentSessionEvent) -> dict[str, Any]:
    """Serialize an agent session event to a JSON-compatible dict.

    For ``message_update`` events, the ``partial`` field is stripped from
    ``assistant_message_event`` to keep stream size linear.
    """
    if isinstance(event, dict):
        raw = event
    elif dataclasses.is_dataclass(event):
        raw = dataclasses.asdict(event)  # type: ignore[assignment]
    else:
        raw = {}  # type: ignore[assignment]
    return _strip_partial_from_event(raw)


def serialize_agent_event(event: MessageUpdateEvent) -> dict[str, Any]:
    """Serialize an agent event to a JSON-compatible dict.

    Dedicated overload for ``MessageUpdateEvent``; strips ``partial``.
    """
    if hasattr(event, "model_dump"):
        raw = event.model_dump()  # type: ignore[union-attr]
    elif dataclasses.is_dataclass(event):
        raw = dataclasses.asdict(event)  # type: ignore[assignment]
    else:
        raw = {}  # type: ignore[assignment]
    return _strip_partial_from_event(raw)
