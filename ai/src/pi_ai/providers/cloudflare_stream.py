"""Cloudflare 流式传输封装。"""

from __future__ import annotations

from typing import Any, cast

from ..types import Model, ProviderEnv, ProviderStreams

CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"


def resolve_cloudflare_model(
    model: Model,
    env: ProviderEnv | None,
) -> Model:
    """解析 Cloudflare 模型 URL 中的占位符。"""
    if not env:
        return model
    base_url = getattr(model, "base_url", "") or ""
    new_base_url = base_url.replace(
        f"{{{CLOUDFLARE_ACCOUNT_ID}}}",
        env.get(CLOUDFLARE_ACCOUNT_ID, f"{{{CLOUDFLARE_ACCOUNT_ID}}}"),
    ).replace(
        f"{{{CLOUDFLARE_GATEWAY_ID}}}",
        env.get(CLOUDFLARE_GATEWAY_ID, f"{{{CLOUDFLARE_GATEWAY_ID}}}"),
    )
    if new_base_url == base_url:
        return model
    # 创建一个新的 model 对象（ModelRecord 或 dict）
    if hasattr(model, "model_copy"):
        return cast(Model, model.model_copy(update={"base_url": new_base_url}))
    # 回退：使用 setattr 修改属性
    try:
        model.base_url = new_base_url  # type: ignore[attr-defined]
        return model
    except Exception:
        return model


def cloudflare_stream_api(streams: ProviderStreams) -> ProviderStreams:
    """包装 ProviderStreams，在分发前解析 Cloudflare 端点占位符。

    Args:
        streams: 原始 ProviderStreams 实现。

    Returns:
        包装后的 ProviderStreams。
    """
    return _CloudflareStreams(streams)


class _CloudflareStreams:
    """Cloudflare 流式传输封装。"""

    def __init__(self, streams: ProviderStreams) -> None:
        self._streams = streams

    def stream(
        self,
        model: Model,
        context: Any,
        options: Any | None = None,
    ) -> Any:
        env = options.env if options and hasattr(options, "env") else None
        resolved = resolve_cloudflare_model(model, env)
        return self._streams.stream(resolved, context, options)

    def stream_simple(
        self,
        model: Model,
        context: Any,
        options: Any | None = None,
    ) -> Any:
        env = options.env if options and hasattr(options, "env") else None
        resolved = resolve_cloudflare_model(model, env)
        return self._streams.stream_simple(resolved, context, options)

    def fetch_deferred(
        self,
        model: Model,
        handle: Any,
        options: Any | None = None,
    ) -> Any:
        return self._streams.fetch_deferred(model, handle, options)

    async def cancel_deferred(
        self,
        model: Model,
        handle: Any,
        options: Any | None = None,
    ) -> None:
        await self._streams.cancel_deferred(model, handle, options)
