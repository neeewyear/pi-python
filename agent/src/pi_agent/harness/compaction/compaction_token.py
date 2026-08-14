"""Token 估算与切点查找。

包含：
- ``estimate_tokens`` / ``estimate_context_tokens`` / ``calculate_context_tokens``
- ``find_cut_point`` / ``find_turn_start_index``
- ``get_last_assistant_usage``
- 相关数据模型：``ContextUsageEstimate`` / ``CutPointResult`` / ``CompactionPreparation``
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ...types import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from pi_session.types import Entry
from .utils import FileOperations

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _safe_json_stringify(value: object) -> str:
    try:
        return json.dumps(value) if value is not None else "undefined"
    except (TypeError, ValueError):
        return "[unserializable]"


ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(content: Sequence[object]) -> int:
    """估算文本+图片内容块的字符数。"""
    chars = 0
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                chars += len(block["text"])
            elif block.get("type") == "image":
                chars += ESTIMATED_IMAGE_CHARS
        elif hasattr(block, "type"):
            if block.type == "text":
                chars += len(block.text)  # type: ignore[attr-defined]
            elif block.type == "image":
                chars += ESTIMATED_IMAGE_CHARS
    return chars


# ---------------------------------------------------------------------------
# Token 估算
# ---------------------------------------------------------------------------


def estimate_tokens(message: AgentMessage) -> int:
    """估算单条消息的 token 数。

    使用保守的字符/4 启发式。
    """
    chars = 0

    if isinstance(message, UserMessage):
        chars = _estimate_text_and_image_content_chars(message.content)
        return (chars + 3) // 4

    if isinstance(message, AssistantMessage):
        for block in message.content:
            if block.type == "text":
                chars += len(block.text)
            elif block.type == "toolCall":
                chars += len(block.name) + len(_safe_json_stringify(block.args))
        # thinking blocks are separate on AssistantMessage
        if message.thinking:
            for tb in message.thinking:
                chars += len(tb.text)
        return (chars + 3) // 4

    if isinstance(message, (CustomMessage, ToolResultMessage)):
        content = message.content
        if isinstance(content, list):
            chars = _estimate_text_and_image_content_chars(content)
        elif isinstance(content, str):
            chars = len(content)
        return (chars + 3) // 4

    if isinstance(message, BashExecutionMessage):
        chars = len(message.command) + len(message.output)
        return (chars + 3) // 4

    if isinstance(message, (BranchSummaryMessage, CompactionSummaryMessage)):
        chars = len(message.summary)
        return (chars + 3) // 4

    return 0


def calculate_context_tokens(usage: Usage) -> int:
    """从 provider usage 计算总上下文 token 数。"""
    return (
        usage.total_tokens
        or usage.input + usage.output + usage.cache_read + usage.cache_write
    )


def _get_assistant_usage(message: AgentMessage) -> Usage | None:
    """获取 assistant 消息的有效 usage。"""
    if not isinstance(message, AssistantMessage):
        return None
    if message.stop_reason in ("aborted", "error"):
        return None
    if message.usage is None:
        return None
    if calculate_context_tokens(message.usage) <= 0:
        return None
    return message.usage


def get_last_assistant_usage(entries: list[Entry]) -> Usage | None:
    """返回最后一条有效 assistant 的 usage。"""
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if entry.type == "message":
            usage = _get_assistant_usage(entry.message)
            if usage is not None:
                return usage
    return None


# ---------------------------------------------------------------------------
# ContextUsageEstimate
# ---------------------------------------------------------------------------


class ContextUsageEstimate(BaseModel):
    """上下文用量估算。"""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None


def _get_last_assistant_usage_info(
    messages: list[AgentMessage],
) -> tuple[Usage, int] | None:
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage is not None:
            return usage, i
    return None


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    """估算上下文 token 数。

    优先使用 provider usage，否则退化为字符启发式。
    """
    usage_info = _get_last_assistant_usage_info(messages)

    if usage_info is None:
        estimated = sum(estimate_tokens(m) for m in messages)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

    usage_tokens = calculate_context_tokens(usage_info[0])
    trailing_tokens = sum(
        estimate_tokens(messages[i]) for i in range(usage_info[1] + 1, len(messages))
    )

    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=usage_info[1],
    )


# ---------------------------------------------------------------------------
# 切点查找
# ---------------------------------------------------------------------------


def _find_valid_cut_points(
    entries: list[Entry], start_index: int, end_index: int
) -> list[int]:
    """查找有效切点。"""    
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if entry.type == "message":
            role = entry.message.role
            if role in (
                "bashExecution",
                "custom",
                "branchSummary",
                "compactionSummary",
                "user",
                "assistant",
            ):
                cut_points.append(i)
        elif entry.type == "branch_summary":
            cut_points.append(i)
    return cut_points


def find_turn_start_index(
    entries: list[Entry], entry_index: int, start_index: int
) -> int:
    """查找包含指定条目的回合起始索引。"""
    for i in range(entry_index, start_index - 1, -1):
        entry = entries[i]
        if entry.type == "branch_summary":
            return i
        if entry.type == "message":
            if entry.message.role in ("user", "bashExecution"):
                return i
    return -1


class CutPointResult(BaseModel):
    """切点结果。"""

    first_kept_entry_index: int
    turn_start_index: int
    is_split_turn: bool


def find_cut_point(
    entries: list[Entry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
) -> CutPointResult:
    """查找压缩切点。"""
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(
            first_kept_entry_index=start_index,
            turn_start_index=-1,
            is_split_turn=False,
        )

    accumulated_tokens = 0
    cut_index = cut_points[0]

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        if entry.type != "message":
            continue
        message_tokens = estimate_tokens(entry.message)
        accumulated_tokens += message_tokens
        if accumulated_tokens >= keep_recent_tokens:
            for c in range(len(cut_points)):
                if cut_points[c] >= i:
                    cut_index = cut_points[c]
                    break
            break

    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        if prev_entry.type in ("compaction", "message"):
            break
        cut_index -= 1

    cut_entry = entries[cut_index]
    is_user_message = cut_entry.type == "message" and cut_entry.message.role == "user"
    turn_start = (
        -1
        if is_user_message
        else find_turn_start_index(entries, cut_index, start_index)
    )

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start,
        is_split_turn=(not is_user_message and turn_start != -1),
    )


# ---------------------------------------------------------------------------
# CompactionPreparation
# ---------------------------------------------------------------------------


class CompactionPreparation(BaseModel):
    """压缩准备数据。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages_to_summarize: list[AgentMessage]
    turn_prefix_messages: list[AgentMessage]
    retained_tail: list[AgentMessage]
    is_split_turn: bool
    tokens_before: int
    previous_summary: str | None = None
    file_ops: FileOperations = FileOperations()
    settings: object = None  # CompactionSettings，避免循环导入
