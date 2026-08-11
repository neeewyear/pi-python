"""Lazy API 加载器（对应 ``api/lazy.ts``）。

提供 ``lazy_api`` 函数，用于延迟加载 API 实现模块。
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from typing import Any

from ..types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from ..utils.event_stream import AssistantMessageEventStream


def _create_setup_error_message(model: Model, error: Exception) -> AssistantMessage:
    """创建设置错误消息。"""
    import time

    return AssistantMessage(
        content=[],
        api=model.api,
        provider=model.provider,
        model=model.model_id,
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


async def _forward_stream(
    target: AssistantMessageEventStream,
    source: AsyncIterable[AssistantMessageEvent],
) -> None:
    """将源流的事件转发到目标流。"""
    async for event in source:
        target.push(event)
    # 尝试获取 result（如果 source 有 result 方法）
    result_fn = getattr(source, "result", None)
    if result_fn is not None:
        result = await result_fn()
        target.end(result)
    else:
        target.end()


def lazy_stream(
    model: Model,
    setup: Callable[[], Any],
) -> AssistantMessageEventStream:
    """延迟流创建（同步返回流，异步执行 setup）。

    Setup 失败时以错误事件终止流。
    """
    import asyncio

    outer = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            inner = await setup()
            await _forward_stream(outer, inner)
        except Exception as error:
            message = _create_setup_error_message(model, error)
            outer.push(AssistantErrorEvent(error=str(error)))
            outer.end(message)
        finally:
            if not outer._done:
                outer.end()

    asyncio.ensure_future(_run())
    return outer


class LazyApiCapabilities:
    """Lazy API 能力标记。"""

    def __init__(
        self,
        fetch_deferred: bool = False,
        cancel_deferred: bool = False,
    ) -> None:
        self.fetch_deferred = fetch_deferred
        self.cancel_deferred = cancel_deferred


def lazy_api(
    load: Callable[[], Any],
    capabilities: LazyApiCapabilities | None = None,
) -> Any:
    """包装动态导入的 API 实现模块为 ``ProviderStreams``。

    Args:
        load: 异步加载函数，返回 ``ProviderStreams`` 兼容对象。
        capabilities: 可选的能力标记。

    Returns:
        包含 ``stream``/``stream_simple`` 等方法的对象。
    """
    caps = capabilities or LazyApiCapabilities()

    class _LazyApi:
        async def _load(self) -> Any:
            impl = await load()
            return impl

        def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            async def _setup() -> Any:
                impl = await self._load()
                return impl.stream(model, context, options)

            return lazy_stream(model, _setup)

        def stream_simple(
            self,
            model: Model,
            context: Context,
            options: SimpleStreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            async def _setup() -> Any:
                impl = await self._load()
                return impl.stream_simple(model, context, options)

            return lazy_stream(model, _setup)

        if caps.fetch_deferred:

            async def fetch_deferred(
                self,
                model: Model,
                handle: DeferredHandle,
                options: DeferredFetchOptions | None = None,
            ) -> AssistantMessageEventStream:
                impl = await self._load()
                if not hasattr(impl, "fetch_deferred"):
                    raise ValueError("API does not support deferred responses")
                return lazy_stream(
                    model, lambda: impl.fetch_deferred(model, handle, options)
                )

        if caps.cancel_deferred:

            async def cancel_deferred(
                self,
                model: Model,
                handle: DeferredHandle,
                options: DeferredCancelOptions | None = None,
            ) -> None:
                impl = await self._load()
                if not hasattr(impl, "cancel_deferred"):
                    raise ValueError("API cannot cancel deferred responses")
                await impl.cancel_deferred(model, handle, options)

    return _LazyApi()


from ..types import AssistantErrorEvent