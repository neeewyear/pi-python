"""Token 估算工具。

提供 ``estimate_context_tokens``、``estimate_message_tokens``、
``estimate_text_tokens`` 等估算函数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from ..types import (
    AssistantMessage,
    Context,
    ImageContent,
    Message,
    TextContent,
    Tool,
    ToolResultMessage,
    Usage,
    UserMessage,
)


@dataclass
class ContextUsageEstimate:
    """上下文用量估算。"""

    tokens: int = 0
    usage_tokens: int = 0
    trailing_tokens: int = 0
    last_usage_index: int | None = None


CHARS_PER_TOKEN = 4
ESTIMATED_IMAGE_CHARS = 4800


def calculate_context_tokens(usage: Usage) -> int:
    """计算上下文 token 数。"""
    return usage.total_tokens or (
        usage.input + usage.output + usage.cache_read + usage.cache_write
    )


def _safe_json_stringify(value: object) -> str:
    """安全地 JSON 序列化。"""
    try:
        return json.dumps(value) if value is not None else "undefined"
    except (TypeError, ValueError, OverflowError):
        return "[unserializable]"


def _estimate_text_and_image_content_chars(
    content: str | list[TextContent | ImageContent],
) -> int:
    """估算文本和图片内容的字符数。"""
    if isinstance(content, str):
        return len(content)
    chars = 0
    for block in content:
        if block.type == "text":
            chars += len(block.text)
        else:
            chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_text_tokens(text: str) -> int:
    """估算文本 token 数。"""
    return max(1, (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)


def estimate_text_and_image_content_tokens(
    content: str | list[TextContent | ImageContent],
) -> int:
    """估算文本和图片内容 token 数。"""
    return max(
        1,
        (_estimate_text_and_image_content_chars(content) + CHARS_PER_TOKEN - 1)
        // CHARS_PER_TOKEN,
    )


def estimate_message_tokens(message: Message) -> int:
    """估算单条消息的 token 数。"""
    if isinstance(message, UserMessage):
        return estimate_text_and_image_content_tokens(message.content)

    if isinstance(message, ToolResultMessage):
        return estimate_text_and_image_content_tokens(message.content)

    if isinstance(message, AssistantMessage):
        chars = 0
        for block in message.content:
            if block.type == "text":
                chars += len(block.text)
            elif block.type == "toolCall":
                chars += len(block.name) + len(_safe_json_stringify(block.args))
            # toolResult 和 image 在 assistant 消息中不贡献 token
        return max(1, (chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN)

    return 1


def _get_last_assistant_usage_info(
    messages: list[Message],
) -> tuple[Usage, int] | None:
    """获取最后一个有效的 assistant 用量信息。"""
    latest_prefix_timestamp = float("-inf")
    usage_info: tuple[Usage, int] | None = None

    for i, msg in enumerate(messages):
        if isinstance(msg, AssistantMessage):
            assistant = msg
            usage_applies_to_prefix = assistant.timestamp >= latest_prefix_timestamp
            if (
                usage_applies_to_prefix
                and assistant.stop_reason not in ("aborted", "error")
                and assistant.usage is not None
                and calculate_context_tokens(assistant.usage) > 0
            ):
                usage_info = (assistant.usage, i)
        latest_prefix_timestamp = max(
            latest_prefix_timestamp, getattr(msg, "timestamp", 0)
        )

    return usage_info


def _estimate_messages(messages: list[Message]) -> ContextUsageEstimate:
    """估算消息列表的 token 数。"""
    usage_info = _get_last_assistant_usage_info(messages)
    if usage_info is not None:
        usage, index = usage_info
        usage_tokens = calculate_context_tokens(usage)
        trailing_tokens = 0
        for i in range(index + 1, len(messages)):
            trailing_tokens += estimate_message_tokens(messages[i])
        return ContextUsageEstimate(
            tokens=usage_tokens + trailing_tokens,
            usage_tokens=usage_tokens,
            trailing_tokens=trailing_tokens,
            last_usage_index=index,
        )

    tokens = 0
    for msg in messages:
        tokens += estimate_message_tokens(msg)
    return ContextUsageEstimate(
        tokens=tokens, trailing_tokens=tokens, last_usage_index=None
    )


def _estimate_tools_tokens(tools: list[Tool] | None) -> int:
    """估算工具定义 token 数。"""
    if not tools:
        return 0
    return estimate_text_tokens(_safe_json_stringify([t.model_dump() for t in tools]))


def estimate_context_tokens(
    context: Context | list[Message],
) -> ContextUsageEstimate:
    """估算上下文的 token 数。  

    可接受 ``Context`` 对象或 ``Message`` 列表。
    """
    if isinstance(context, list):
        return _estimate_messages(context)

    estimate = _estimate_messages(context.messages)
    if estimate.last_usage_index is not None:
        added_names: set[str] = set()
        for msg in context.messages[estimate.last_usage_index + 1 :]:
            if isinstance(msg, ToolResultMessage) and msg.added_tool_names:
                added_names.update(msg.added_tool_names)
        added_tool_tokens = _estimate_tools_tokens(
            [t for t in (context.tools or []) if t.name in added_names]
        )
        return ContextUsageEstimate(
            tokens=estimate.tokens + added_tool_tokens,
            usage_tokens=estimate.usage_tokens,
            trailing_tokens=estimate.trailing_tokens + added_tool_tokens,
            last_usage_index=estimate.last_usage_index,
        )

    prefix_tokens = (
        estimate_text_tokens(context.system_prompt) if context.system_prompt else 0
    ) + _estimate_tools_tokens(context.tools)
    return ContextUsageEstimate(
        tokens=estimate.tokens + prefix_tokens,
        usage_tokens=estimate.usage_tokens,
        trailing_tokens=estimate.trailing_tokens + prefix_tokens,
        last_usage_index=estimate.last_usage_index,
    )
