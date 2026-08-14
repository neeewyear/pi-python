"""事件流工具。

提供 ``EventStream`` 通用异步可迭代类，以及 ``AssistantMessageEventStream``。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Generic, TypeVar

from ..types import AssistantMessage, AssistantMessageEvent

T = TypeVar("T")
R = TypeVar("R")


class EventStream(Generic[T, R], AsyncIterator[T]):
    """通用事件流类。

    支持 producer-consumer 模式：生产者 ``push``/``end``，消费者 ``async for``。
    """

    def __init__(
        self,
        is_complete: Callable[[T], bool],
        extract_result: Callable[[T], R],
    ) -> None:
        self._queue: list[T] = []
        self._waiting: list[asyncio.Future[IteratorResult[T]]] = []
        self._done = False
        self._final_result: asyncio.Future[R] = asyncio.get_event_loop().create_future()
        self._is_complete = is_complete
        self._extract_result = extract_result

    def push(self, event: T) -> None:
        """推送一个事件到流中。"""
        if self._done:
            return

        if self._is_complete(event):
            self._done = True
            if not self._final_result.done():
                self._final_result.set_result(self._extract_result(event))

        if self._waiting:
            waiter = self._waiting.pop(0)
            if not waiter.done():
                waiter.set_result(IteratorResult(value=event, done=False))
        else:
            self._queue.append(event)

    def end(self, result: R | None = None) -> None:
        """结束流。"""
        self._done = True
        if result is not None and not self._final_result.done():
            self._final_result.set_result(result)
        for waiter in self._waiting:
            if not waiter.done():
                waiter.set_result(IteratorResult(value=None, done=True))
        self._waiting.clear()

    async def __anext__(self) -> T:
        while True:
            if self._queue:
                return self._queue.pop(0)
            if self._done:
                raise StopAsyncIteration
            future: asyncio.Future[IteratorResult[T]] = (
                asyncio.get_event_loop().create_future()
            )
            self._waiting.append(future)
            result = await future
            if result.done or result.value is None:
                raise StopAsyncIteration
            return result.value

    async def result(self) -> R:
        return await self._final_result

    def __aiter__(self) -> EventStream[T, R]:
        return self


class AssistantMessageEventStream(EventStream[AssistantMessageEvent, AssistantMessage]):
    """Assistant 消息事件流。"""

    def __init__(self) -> None:
        def _is_complete(event: AssistantMessageEvent) -> bool:
            return event.type in ("stream_end", "error")

        def _extract_result(event: AssistantMessageEvent) -> AssistantMessage:
            if event.type == "stream_end" and event.message is not None:
                return event.message
            if event.type == "error":
                return event.error
            raise ValueError("Unexpected event type for final result")

        super().__init__(_is_complete, _extract_result)


def create_assistant_message_event_stream() -> AssistantMessageEventStream:
    """创建 ``AssistantMessageEventStream`` 工厂函数。"""
    return AssistantMessageEventStream()


# 辅助类型
import asyncio
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class IteratorResult(Generic[T]):
    """迭代器结果包装。"""

    value: T | None
    done: bool
