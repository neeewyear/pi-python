"""中止信号工具（对应 ``utils/abort.ts`` + ``utils/abort-signals.ts``）。

提供 ``CancellationToken`` 类以及 ``operation_signal``、``race_with_abort_signal``、
``combine_abort_signals``。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


class CancellationToken:
    """取消令牌（对应 TS ``AbortSignal``）。

    基于 ``asyncio.Event`` 实现，替代 TS 的 ``AbortSignal``/``AbortController``。
    """

    __slots__ = ("_callbacks", "_event")

    def __init__(self) -> None:
        self._event = asyncio.Event()
        self._callbacks: list[Callable[[], None]] = []

    @property
    def aborted(self) -> bool:
        """是否已被取消。"""
        return self._event.is_set()

    def cancel(self, reason: object = None) -> None:
        """触发取消。"""
        if not self._event.is_set():
            self._event.set()
            for cb in self._callbacks:
                try:
                    cb()
                except Exception:
                    pass

    def add_callback(self, callback: Callable[[], None]) -> Callable[[], None]:
        """添加取消回调。返回移除函数。"""
        self._callbacks.append(callback)

        def _remove() -> None:
            if callback in self._callbacks:
                self._callbacks.remove(callback)

        return _remove

    def throw_if_cancelled(self) -> None:
        """若已取消则抛出 ``asyncio.CancelledError``。"""
        if self.aborted:
            raise asyncio.CancelledError("operation cancelled")

    async def wait(self) -> None:
        """阻塞直到令牌被取消。"""
        await self._event.wait()


def operation_signal(signal: CancellationToken | None = None) -> CancellationToken:
    """创建或返回操作信号（对应 TS ``operationSignal``）。"""
    return signal if signal is not None else CancellationToken()


async def race_with_abort_signal(
    operation: Awaitable[object], signal: CancellationToken
) -> object:
    """与取消信号赛跑（对应 TS ``raceWithAbortSignal``）。

    ``asyncio`` 的协程取消机制（``CancelledError``）已内建此语义，
    Python 侧只需在适当时机检查 ``signal.aborted``。
    """
    if signal.aborted:
        raise asyncio.CancelledError("The operation was aborted")
    return await operation


@dataclass
class CombinedAbortSignal:
    """组合取消信号（对应 TS ``CombinedAbortSignal``）。"""

    signal: CancellationToken | None = None
    cleanup: Callable[[], None] = lambda: None


def combine_abort_signals(
    signals: list[CancellationToken | None],
) -> CombinedAbortSignal:
    """组合多个取消信号（对应 TS ``combineAbortSignals``）。

    当任一信号取消时，合并的信号也取消。
    """
    active_signals = [s for s in signals if s is not None]
    if not active_signals:
        return CombinedAbortSignal()
    if len(active_signals) == 1:
        return CombinedAbortSignal(signal=active_signals[0])

    combined = CancellationToken()
    removers: list[Callable[[], None]] = []

    def _on_abort(source: CancellationToken) -> None:
        if not combined.aborted:
            combined.cancel()

    for sig in active_signals:
        if sig.aborted:
            combined.cancel()
            break

        def _on_abort_wrapper(s: CancellationToken = sig) -> None:
            _on_abort(s)

        remover = sig.add_callback(_on_abort_wrapper)
        removers.append(remover)

    def _cleanup() -> None:
        for r in removers:
            try:
                r()
            except Exception:
                pass

    return CombinedAbortSignal(signal=combined, cleanup=_cleanup)
