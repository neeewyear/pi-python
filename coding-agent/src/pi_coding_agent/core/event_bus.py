"""轻量事件总线（对应 TS ``core/event-bus.ts``）。

使用 ``asyncio`` 原语替代 Node.js ``EventEmitter``。所有 handler 支持 async，
emit 以同步方式触发，但内部会 await 所有 async handler。
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from typing import TypeAlias

Handler: TypeAlias = Callable[[object], Awaitable[None] | None]
"""事件处理器签名：接收 ``object`` 数据，可同步或异步。"""

Unsubscribe: TypeAlias = Callable[[], None]
"""取消订阅函数签名。"""


class EventBus:
    """轻量级事件总线。

    支持任意字符串频道，handler 可以是同步或异步函数。emit 是同步的，
    但会为 async handler 创建 task 执行。
    """

    _handlers: dict[str, list[Handler]]

    def __init__(self) -> None:
        self._handlers = {}

    def emit(self, channel: str, data: object) -> None:
        """向指定频道发送事件。

        Args:
            channel: 频道名称。
            data: 事件数据。
        """
        handlers = self._handlers.get(channel, [])
        for handler in handlers:
            result = handler(data)
            if inspect.isawaitable(result):
                asyncio.ensure_future(result)

    def on(self, channel: str, handler: Handler) -> Unsubscribe:
        """订阅指定频道的事件。

        Args:
            channel: 频道名称。
            handler: 事件处理器（可同步或异步）。

        Returns:
            取消订阅函数。
        """
        if channel not in self._handlers:
            self._handlers[channel] = []
        self._handlers[channel].append(handler)

        def unsubscribe() -> None:
            _handlers = self._handlers.get(channel)
            if _handlers is not None:
                try:
                    _handlers.remove(handler)
                except ValueError:
                    pass

        return unsubscribe

    def clear(self) -> None:
        """清除所有频道的所有订阅者。"""
        self._handlers.clear()
