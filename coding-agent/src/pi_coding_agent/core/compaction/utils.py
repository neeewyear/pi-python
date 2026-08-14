"""Shared utilities for compaction and branch summarization.

Contains:
- ``FileOperations``: file operation accumulator
- ``serialize_conversation``: serialize LLM messages to plain text
- File list computation and formatting
"""

from __future__ import annotations

import json

from pi_agent.types import AgentMessage, AssistantMessage, Message
from pi_ai.utils.text import content_text
from pydantic import BaseModel, ConfigDict

# ---------------------------------------------------------------------------
# File Operation Tracking
# ---------------------------------------------------------------------------


class FileOperations(BaseModel):
    """File operation accumulator."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    read: set[str] = set()
    written: set[str] = set()
    edited: set[str] = set()


def create_file_ops() -> FileOperations:
    """Create an empty file operation accumulator."""
    return FileOperations()


def extract_file_ops_from_message(
    message: AgentMessage, file_ops: FileOperations
) -> None:
    """Extract file paths from tool calls in an assistant message."""
    if message.role != "assistant":
        return
    if not isinstance(message, AssistantMessage):
        return

    for block in message.content:
        if block.type != "toolCall":
            continue
        args = block.args
        path = args.get("path") if isinstance(args, dict) else None
        if not isinstance(path, str):
            continue

        if block.name == "read":
            file_ops.read.add(path)
        elif block.name == "write":
            file_ops.written.add(path)
        elif block.name == "edit":
            file_ops.edited.add(path)


def compute_file_lists(file_ops: FileOperations) -> tuple[list[str], list[str]]:
    """Compute sorted read-only and modified file lists."""
    modified: set[str] = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """Format file operations as XML tags for summary."""
    sections: list[str] = []
    if read_files:
        sections.append(f"<read-files>\n{chr(10).join(read_files)}\n</read-files>")
    if modified_files:
        sections.append(
            f"<modified-files>\n{chr(10).join(modified_files)}\n</modified-files>"
        )
    if not sections:
        return ""
    return f"\n\n{chr(10).join(sections)}"


# ---------------------------------------------------------------------------
# Message Serialization
# ---------------------------------------------------------------------------

TOOL_RESULT_MAX_CHARS = 2000
"""Maximum characters for a tool result in serialized summaries."""


def _safe_json_stringify(value: object) -> str:
    try:
        return json.dumps(value) if value is not None else "undefined"
    except (TypeError, ValueError):
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    """Truncate text to a maximum character length for summarization."""
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def serialize_conversation(messages: list[Message]) -> str:
    """Serialize LLM messages to text for summarization.

    This prevents the model from treating it as a conversation to continue.
    """
    parts: list[str] = []

    for msg in messages:
        if msg.role == "user":
            text = content_text(msg.content, "")  # type: ignore[arg-type]
            if text:
                parts.append(f"[User]: {text}")

        elif msg.role == "assistant":
            thinking_parts: list[str] = []
            tool_calls: list[str] = []

            # thinking blocks are a separate field on AssistantMessage
            if msg.thinking:
                for tb in msg.thinking:
                    thinking_parts.append(tb.text)

            for block in msg.content:
                if block.type == "toolCall":
                    args_str = ", ".join(
                        f"{k}={_safe_json_stringify(v)}" for k, v in block.args.items()
                    )
                    tool_calls.append(f"{block.name}({args_str})")

            if thinking_parts:
                parts.append(f"[Assistant thinking]: {chr(10).join(thinking_parts)}")
            if any(b.type == "text" for b in msg.content):
                parts.append(f"[Assistant]: {content_text(msg.content)}")  # type: ignore[arg-type]
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")

        elif msg.role == "toolResult":
            text = content_text(msg.content, "")  # type: ignore[arg-type]
            if text:
                parts.append(
                    f"[Tool result]: {_truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)}"
                )

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Summarization System Prompt
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a context summarization assistant. Your task is to read a conversation "
    "between a user and an AI assistant, then produce a structured summary following "
    "the exact format specified.\n\n"
    "Do NOT continue the conversation. Do NOT respond to any questions in the "
    "conversation. ONLY output the structured summary."
)

__all__ = [
    "SUMMARIZATION_SYSTEM_PROMPT",
    "TOOL_RESULT_MAX_CHARS",
    "FileOperations",
    "compute_file_lists",
    "create_file_ops",
    "extract_file_ops_from_message",
    "format_file_operations",
    "serialize_conversation",
]
