"""OAuth 设备码轮询流程（对应 ``oauth/device-code.ts``）。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypedDict

CANCEL_MESSAGE = "Login cancelled"
TIMEOUT_MESSAGE = "Device flow timed out"
SLOW_DOWN_TIMEOUT_MESSAGE = (
    "Device flow timed out after one or more slow_down responses. "
    "This is often caused by clock drift in WSL or VM environments. "
    "Please sync or restart the VM clock and try again."
)
MINIMUM_INTERVAL_MS = 1000
# RFC 8628 section 3.2: if the authorization server omits `interval`, the client must use 5 seconds.
DEFAULT_POLL_INTERVAL_SECONDS = 5
# RFC 8628 section 3.5: `slow_down` means the polling interval must increase by 5 seconds.
SLOW_DOWN_INTERVAL_INCREMENT_MS = 5000


class OAuthDeviceCodePollResult(TypedDict, total=False):
    """设备码轮询结果。

    所有字段均为可选，运行时通过 ``.get()`` 安全访问。
    """

    status: Literal["incomplete", "complete", "failed", "slow_down", "pending"]
    value: Any
    message: str
    interval_seconds: int


class OAuthDeviceCodePollOptions(TypedDict, total=False):
    """设备码轮询选项。"""

    user_code: str
    verification_uri: str
    verification_uri_complete: str | None
    interval_seconds: int
    expires_in_seconds: int
    device_code: str
    signal: Any
    poll: Callable[[str, Any], Awaitable[OAuthDeviceCodePollResult]]


async def _abortable_sleep(
    ms: float,
    signal: Any,
    cancel_message: str = CANCEL_MESSAGE,
) -> None:
    """支持取消的可中断睡眠。"""
    try:
        await asyncio.sleep(ms / 1000)
    except asyncio.CancelledError:
        raise RuntimeError(cancel_message) from None


async def poll_oauth_device_code_flow(options: OAuthDeviceCodePollOptions) -> Any:
    """轮询 OAuth 设备码流程。

    对应 TS ``pollOAuthDeviceCodeFlow``。
    """
    device_code = options.get("device_code", "")
    signal = options.get("signal")
    interval_seconds = options.get("interval_seconds", DEFAULT_POLL_INTERVAL_SECONDS)
    expires_in_seconds = options.get("expires_in_seconds")
    poll_fn = options.get("poll")

    deadline = (
        time.time() + expires_in_seconds
        if expires_in_seconds is not None
        else float("inf")
    )
    interval_ms = max(MINIMUM_INTERVAL_MS, int(interval_seconds * 1000))

    slow_down_responses = 0

    while time.time() < deadline:
        if signal is not None and getattr(signal, "aborted", False):
            raise RuntimeError(CANCEL_MESSAGE)

        if poll_fn is not None:
            result = await poll_fn(device_code, signal)
        else:
            raise RuntimeError("No poll function provided")

        # Check for complete result
        if result.get("status") == "complete":
            return result.get("value")

        # Check for failure
        if result.get("status") == "failed":
            msg = result.get("message", "Unknown error")
            raise RuntimeError(msg)

        # Handle slow_down
        if result.get("status") == "slow_down":
            slow_down_responses += 1
            server_interval = result.get("interval_seconds")
            if (
                server_interval is not None
                and isinstance(server_interval, (int, float))
                and server_interval > 0
            ):
                interval_ms = max(
                    MINIMUM_INTERVAL_MS,
                    int(server_interval * 1000),
                )
            else:
                interval_ms = max(
                    MINIMUM_INTERVAL_MS,
                    interval_ms + SLOW_DOWN_INTERVAL_INCREMENT_MS,
                )

        remaining_ms = (deadline - time.time()) * 1000
        if remaining_ms <= 0:
            break

        await _abortable_sleep(
            min(interval_ms, remaining_ms),
            signal,
            CANCEL_MESSAGE,
        )

    raise RuntimeError(
        SLOW_DOWN_TIMEOUT_MESSAGE if slow_down_responses > 0 else TIMEOUT_MESSAGE
    )
