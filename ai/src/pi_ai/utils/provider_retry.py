"""Provider 级别重试策略。

提供与 OpenAI / Anthropic SDK 相同的重试行为。
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, cast

DEFAULT_MAX_RETRY_DELAY_MS = 60_000


@dataclass
class ProviderRetryOptions:
    """Provider 重试选项。"""

    max_retries: int = 0
    max_retry_delay_ms: int | None = None
    signal: Any = None  # CancellationToken


class ProviderError(Exception):
    """Provider 错误。"""

    def __init__(
        self,
        message: str,
        status: int | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.headers = headers or {}


def _is_provider_error(error: object) -> bool:
    """检查是否为 provider 错误。"""
    return isinstance(error, ProviderError)


def _is_retryable_provider_error(error: ProviderError) -> bool:
    """判断 provider 错误是否可重试。"""
    should_retry = error.headers.get("x-should-retry")
    if should_retry == "true":
        return True
    if should_retry == "false":
        return False

    if error.status is None:
        return True
    return (
        error.status == 408
        or error.status == 409
        or error.status == 429
        or error.status >= 500
    )


def _validate_server_retry_delay_ms(
    delay_ms: float,
    max_retry_delay_ms: int | None,
    provider_error_message: str,
) -> float:
    """验证服务器请求的重试延迟时间。"""
    max_delay = max_retry_delay_ms or DEFAULT_MAX_RETRY_DELAY_MS
    if max_delay > 0 and delay_ms > max_delay:
        raise ProviderError(
            f"Server requested {delay_ms / 1000:.0f}s retry delay"
            f" (max: {max_delay / 1000:.0f}s). {provider_error_message}",
            status=429,
        )
    return delay_ms


def _get_retry_delay_ms(
    error: ProviderError,
    retry_index: int,
    max_retry_delay_ms: int | None,
) -> float:
    """计算重试延迟时间。"""
    retry_after_ms = error.headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            value = float(retry_after_ms)
            return _validate_server_retry_delay_ms(
                value, max_retry_delay_ms, error.args[0] if error.args else ""
            )
        except (ValueError, TypeError):
            pass

    retry_after = error.headers.get("retry-after")
    if retry_after is not None:
        try:
            seconds = float(retry_after)
            delay_ms = seconds * 1000
        except (ValueError, TypeError):
            try:
                parsed = time.mktime(
                    time.strptime(retry_after, "%a, %d %b %Y %H:%M:%S %Z")
                )
                delay_ms = (parsed - time.time()) * 1000
            except (ValueError, OSError):
                delay_ms = 0
        return _validate_server_retry_delay_ms(
            delay_ms, max_retry_delay_ms, error.args[0] if error.args else ""
        )

    exponential_delay = min(0.5 * 2**retry_index, 8) * 1000
    return float(exponential_delay * (1 - random.random() * 0.25))


def _create_abort_error() -> BaseException:
    """创建中止错误。"""
    return asyncio.CancelledError("Request aborted")


async def _abortable_sleep(ms: float, signal: Any = None) -> None:
    """可中止的休眠。"""
    if signal is not None and getattr(signal, "aborted", False):
        raise _create_abort_error()

    try:
        await asyncio.sleep(max(0, ms / 1000))
    except asyncio.CancelledError:
        raise _create_abort_error() from None


async def retry_provider_request(
    request: Callable[[], Awaitable[object]],
    options: ProviderRetryOptions | None = None,
) -> object:
    """Provider 请求重试。

    重现 OpenAI 和 Anthropic SDK 的重试行为，同时使其退避休眠可中断。
    """
    opts = options or ProviderRetryOptions()
    max_retries = opts.max_retries
    retries_remaining = max_retries

    while True:
        try:
            return await request()
        except Exception as error:
            if opts.signal is not None and getattr(opts.signal, "aborted", False):
                raise _create_abort_error() from error
            if retries_remaining <= 0 or not _is_provider_error(error):
                raise
            provider_error = cast(ProviderError, error)
            if not _is_retryable_provider_error(provider_error):
                raise

            retry_index = max_retries - retries_remaining
            retries_remaining -= 1
            delay_ms = _get_retry_delay_ms(
                provider_error, retry_index, opts.max_retry_delay_ms
            )
            await _abortable_sleep(delay_ms, opts.signal)
