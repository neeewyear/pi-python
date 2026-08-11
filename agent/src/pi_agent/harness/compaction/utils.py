"""压缩工具函数（对应 ``harness/compaction/utils.ts``）。

包含：
- ``FileOperations``：文件操作累加器
- ``serializeConversation``：LLM 消息序列化为纯文本
- 文件列表计算与格式化
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ...types import AgentMessage, AssistantMessage, Message

# ---------------------------------------------------------------------------
# FileOperations
# ---------------------------------------------------------------------------


class FileOperations(BaseModel):
    """文件操作累加器（对应 TS ``FileOperations``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    read: set[str] = set()
    written: set[str] = set()
    edited: set[str] = set()


def create_file_ops() -> FileOperations:
    """创建空文件操作累加器（对应 TS ``createFileOps``）。"""
    return FileOperations()


def extract_file_ops_from_message(
    message: AgentMessage, file_ops: FileOperations
) -> None:
    """从 assistant 消息的 toolCall 中提取文件路径（对应 TS ``extractFileOpsFromMessage``）。"""
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
    """计算 sorted read-only 和 modified 文件列表（对应 TS ``computeFileLists``）。"""
    modified: set[str] = file_ops.edited | file_ops.written
    read_only = sorted(f for f in file_ops.read if f not in modified)
    modified_files = sorted(modified)
    return read_only, modified_files


def format_file_operations(read_files: list[str], modified_files: list[str]) -> str:
    """格式化为 XML 标签（对应 TS ``formatFileOperations``）。"""
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
# 序列化
# ---------------------------------------------------------------------------

TOOL_RESULT_MAX_CHARS = 2000


def _safe_json_stringify(value: object) -> str:
    try:
        return json.dumps(value) if value is not None else "undefined"
    except (TypeError, ValueError):
        return "[unserializable]"


def _truncate_for_summary(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    truncated_chars = len(text) - max_chars
    return f"{text[:max_chars]}\n\n[... {truncated_chars} more characters truncated]"


def _content_text(content: Sequence[object]) -> str:
    """提取 content 列表中的纯文本。"""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        elif hasattr(block, "type") and block.type == "text":
            parts.append(block.text)  # type: ignore[attr-defined]
    return "".join(parts)


def serialize_conversation(messages: list[Message]) -> str:
    """序列化 LLM 消息为纯文本（对应 TS ``serializeConversation``）。"""
    parts: list[str] = []

    for msg in messages:
        if msg.role == "user":
            text = _content_text(msg.content)
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
                parts.append(f"[Assistant]: {_content_text(msg.content)}")
            if tool_calls:
                parts.append(f"[Assistant tool calls]: {'; '.join(tool_calls)}")

        elif msg.role == "toolResult":
            text = _content_text(msg.content)
            if text:
                parts.append(
                    f"[Tool result]: {_truncate_for_summary(text, TOOL_RESULT_MAX_CHARS)}"
                )

    return "\n\n".join(parts)
