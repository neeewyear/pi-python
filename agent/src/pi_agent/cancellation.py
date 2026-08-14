"""取消令牌。

在需要"非抛异常取消"的边界（工具执行、钩子回调、文件系统操作）使用显式
``CancellationToken``；纯协程取消场景使用 ``asyncio.CancelledError``。
"""

from __future__ import annotations

import asyncio


class CancellationToken:
    """可被显式取消的令牌，内部基于 ``asyncio.Event`` 实现。"""

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = asyncio.Event()

    @property
    def aborted(self) -> bool:
        """是否已被取消。"""
        return self._event.is_set()

    def cancel(self) -> None:
        """触发取消。后续 ``aborted`` 为 True。"""
        self._event.set()

    def throw_if_cancelled(self) -> None:
        """若已取消则抛出 ``asyncio.CancelledError``（协作式取消检查点）。"""
        if self.aborted:
            raise asyncio.CancelledError("operation cancelled")

    async def wait(self) -> None:
        """阻塞直到令牌被取消。"""
        await self._event.wait()
