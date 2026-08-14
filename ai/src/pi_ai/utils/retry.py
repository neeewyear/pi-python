"""重试逻辑。

提供 ``RetryPolicy``、``retry_assistant_call``、``is_retryable_assistant_error``。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol, cast

from ..types import AssistantMessage

# ---------------------------------------------------------------------------
# Provider 错误模式
# ---------------------------------------------------------------------------

NON_RETRYABLE_PROVIDER_LIMIT_PATTERN: re.Pattern[str] = re.compile(
    "|".join(
        [
            r"GoUsageLimitError",
            r"FreeUsageLimitError",
            r"Monthly usage limit reached",
            r"available balance",
            r"insufficient_quota",
            r"out of budget",
            r"quota exceeded",
            r"billing",
        ]
    ),
    re.IGNORECASE,
)

RETRYABLE_PROVIDER_ERROR_PATTERN: re.Pattern[str] = re.compile(
    "|".join(
        [
            r"overloaded",
            r"rate.?limit",
            r"too many requests",
            r"429",
            r"500",
            r"502",
            r"503",
            r"504",
            r"524",
            r"service.?unavailable",
            r"server.?error",
            r"internal.?error",
            r"provider.?returned.?error",
            r"network.?error",
            r"connection.?error",
            r"connection.?refused",
            r"connection.?lost",
            r"other side closed",
            r"fetch failed",
            r"getaddrinfo",
            r"ENOTFOUND",
            r"EAI_AGAIN",
            r"upstream.?connect",
            r"reset before headers",
            r"socket hang up",
            r"socket connection was closed",
            r"timed?\s?out",
            r"timeout",
            r"terminated",
            r"websocket.?closed",
            r"websocket.?error",
            r"ended without",
            r"stream ended before message_stop",
            r"stream ended before a terminal response event",
            r"http2 request did not get a response",
            r"retry delay",
            r"you can retry your request",
            r"try your request again",
            r"please retry your request",
            r"ResourceExhausted",
        ]
    ),
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class AbortSignal(Protocol):
    """abort 信号协议。"""

    @property
    def aborted(self) -> bool: ...


@dataclass
class RetryPolicy:
    """重试策略。"""

    enabled: bool = False
    max_retries: int = 0
    base_delay_ms: int = 1000


@dataclass
class RetryCallbacks:
    """重试回调。"""

    on_retry_scheduled: Callable[[int, int, int, str], Awaitable[None]] | None = None
    on_retry_attempt_start: Callable[[], Awaitable[None]] | None = None
    on_retry_finished: Callable[[bool, int, str | None], Awaitable[None]] | None = None


class RetrySleepAbortError(Exception):
    """重试休眠被中止。"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


async def _sleep(ms: int, signal: AbortSignal | None = None) -> None:
    """休眠。"""
    if signal is not None and signal.aborted:
        raise RetrySleepAbortError()
    try:
        await asyncio.sleep(ms / 1000)
    except asyncio.CancelledError:
        raise RetrySleepAbortError() from None


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


async def retry_assistant_call(
    produce: Callable[[], Awaitable[AssistantMessage]],
    policy: RetryPolicy | None,
    signal: AbortSignal | None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """带边界重试的 assistant 调用。

    行为：
    - 成功响应立即返回。abort 是终态的，不重试。
    - 不可重试的错误（配额/计费耗尽）立即返回。
    - 否则按指数退避重试最多 ``max_retries`` 次。
    """
    max_attempts = policy.max_retries if (policy and policy.enabled) else 0

    attempt = 0
    last_retry: dict[str, object] | None = None

    while True:
        response = await produce()

        # abort: 终态但不成功
        if response.stop_reason == "aborted":
            if last_retry is not None and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(
                    False, int(cast(int, last_retry["attempt"])), None
                )
            return response

        # 成功: 非 error 非 abort 的响应直接返回
        if response.stop_reason != "error":
            if last_retry is not None and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(
                    True, int(cast(int, last_retry["attempt"])), None
                )
            return response

        # 不可重试或预算耗尽
        if attempt >= max_attempts or not is_retryable_assistant_error(response):
            if last_retry is not None and callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(
                    False, int(cast(int, last_retry["attempt"])), response.error_message
                )
            return response

        attempt += 1
        last_retry = {
            "attempt": attempt,
            "error_message": response.error_message or "Unknown error",
        }
        delay_ms = policy.base_delay_ms * (2 ** (attempt - 1)) if policy else 1000
        if callbacks and callbacks.on_retry_scheduled:
            await callbacks.on_retry_scheduled(
                attempt, max_attempts, delay_ms, str(last_retry["error_message"])
            )

        try:
            await _sleep(delay_ms, signal)
        except RetrySleepAbortError:
            if callbacks and callbacks.on_retry_finished:
                await callbacks.on_retry_finished(
                    False, attempt, str(last_retry["error_message"])
                )
            return AssistantMessage(
                content=response.content,
                api=response.api,
                provider=response.provider,
                model=response.model,
                stop_reason="aborted",
                timestamp=response.timestamp,
            )

        if callbacks and callbacks.on_retry_attempt_start:
            await callbacks.on_retry_attempt_start()


def is_retryable_assistant_error(message: AssistantMessage) -> bool:
    """判断 assistant 错误是否可重试。"""
    if message.stop_reason != "error" or not message.error_message:
        return False
    error_message = message.error_message
    if NON_RETRYABLE_PROVIDER_LIMIT_PATTERN.search(error_message):
        return False
    return bool(RETRYABLE_PROVIDER_ERROR_PATTERN.search(error_message))
