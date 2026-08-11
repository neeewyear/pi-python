"""取消信号工具（对应 TS ``utils/abort.ts``）。

提供 ``AbortError`` 自定义异常和 ``abort_with_timeout`` 超时包装函数。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar

_T = TypeVar("_T")


class AbortError(asyncio.TimeoutError):
    """操作被取消时抛出的自定义异常。

    对应 TS 端的 ``AbortError``。
    """

    def __init__(self, message: str = "The operation was aborted") -> None:
        super().__init__(message)


async def abort_with_timeout(coro: Awaitable[_T], timeout: float) -> _T:
    """使用超时包装协程。

    在指定超时时间内等待协程完成，超时则抛出 ``AbortError``。

    Args:
        coro: 要执行的协程。
        timeout: 超时时间（秒）。

    Returns:
        协程的返回值。

    Raises:
        AbortError: 超时时抛出。
    """
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except asyncio.TimeoutError as e:
        raise AbortError(f"Operation timed out after {timeout}s") from e
