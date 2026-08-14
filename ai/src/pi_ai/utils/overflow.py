"""上下文溢出检测。

提供 ``is_context_overflow``、``is_recoverable_length``、``get_overflow_patterns``。
"""

from __future__ import annotations

import re

from ..types import AssistantMessage

# ---------------------------------------------------------------------------
# 溢出模式
# ---------------------------------------------------------------------------

OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"prompt is too long", re.IGNORECASE),
    re.compile(r"request_too_large", re.IGNORECASE),
    re.compile(r"input is too long for requested model", re.IGNORECASE),
    re.compile(r"exceeds the context window", re.IGNORECASE),
    re.compile(
        r"exceeds (?:the )?(?:model'?s )?maximum context length(?: of [\d,]+ tokens?|\s*\([\d,]+\))?",
        re.IGNORECASE,
    ),
    re.compile(r"input token count.*exceeds the maximum", re.IGNORECASE),
    re.compile(r"maximum prompt length is \d+", re.IGNORECASE),
    re.compile(r"reduce the length of the messages", re.IGNORECASE),
    re.compile(r"maximum context length is \d+ tokens", re.IGNORECASE),
    re.compile(r"exceeds (?:the )?maximum allowed input length of [\d,]+ tokens?", re.IGNORECASE),
    re.compile(r"input \(\d+ tokens\) is longer than the model'?s context length \(\d+ tokens\)", re.IGNORECASE),
    re.compile(r"exceeds the limit of \d+", re.IGNORECASE),
    re.compile(r"exceeds the available context size", re.IGNORECASE),
    re.compile(r"greater than the context length", re.IGNORECASE),
    re.compile(r"context window exceeds limit", re.IGNORECASE),
    re.compile(r"exceeded model token limit", re.IGNORECASE),
    re.compile(r"too large for model with \d+ maximum context length", re.IGNORECASE),
    re.compile(r"prompt has [\d,]+ tokens?, but the configured context size is [\d,]+ tokens?", re.IGNORECASE),
    re.compile(r"model_context_window_exceeded", re.IGNORECASE),
    re.compile(r"prompt too long; exceeded (?:max )?context length", re.IGNORECASE),
    re.compile(r"range of input length should be", re.IGNORECASE),
    re.compile(r"context[_ ]length[_ ]exceeded", re.IGNORECASE),
    re.compile(r"too many tokens", re.IGNORECASE),
    re.compile(r"token limit exceeded", re.IGNORECASE),
    re.compile(r"^4(?:00|13)\s*(?:status code)?\s*\(no body\)", re.IGNORECASE),
]

NON_OVERFLOW_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^(Throttling error|Service unavailable):", re.IGNORECASE),
    re.compile(r"rate limit", re.IGNORECASE),
    re.compile(r"too many requests", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def is_context_overflow(
    message: AssistantMessage, context_window: int | None = None
) -> bool:
    """检查 assistant 消息是否表示上下文溢出错误。

    处理三种情况：
    1. 基于错误消息的溢出（大多数 provider）
    2. 静默溢出（z.ai 风格）：成功但 usage 超过 context_window
    3. Length-stop 溢出（Xiaomi MiMo 风格）：stopReason "length" + output=0
    """
    # Case 1: 检查错误消息模式
    if message.stop_reason == "error" and message.error_message:
        is_non_overflow = any(p.search(message.error_message) for p in NON_OVERFLOW_PATTERNS)
        if not is_non_overflow and any(p.search(message.error_message) for p in OVERFLOW_PATTERNS):
            return True

    # Case 2: 静默溢出（z.ai 风格）
    if context_window is not None and message.stop_reason == "stop" and message.usage:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens > context_window:
            return True

    # Case 3: Length-stop 溢出（Xiaomi MiMo 风格）
    if context_window is not None and message.stop_reason == "length" and message.usage and message.usage.output == 0:
        input_tokens = message.usage.input + message.usage.cache_read
        if input_tokens >= context_window * 0.99:
            return True

    return False


def is_recoverable_length(message: AssistantMessage, desired_max_output: int) -> bool:
    """检查 length stop 是否因上下文压力或 provider 截断导致。"""
    return (
        message.stop_reason == "length"
        and desired_max_output > 0
        and message.usage is not None
        and message.usage.output < desired_max_output
    )


def get_overflow_patterns() -> list[re.Pattern[str]]:
    """获取溢出模式列表（用于测试）。"""    
    return list(OVERFLOW_PATTERNS)