"""OpenRouter 图片生成 API 延迟加载。"""

from __future__ import annotations

from typing import Any, cast

from ..types import ProviderImages


class _OpenRouterImagesProvider:
    """OpenRouter 图片生成 API 实现（延迟加载）。"""

    async def generate_images(
        self,
        model: Any,
        context: Any,
        options: Any | None = None,
    ) -> dict[str, Any]:
        from .openrouter_images import generate_images

        return await generate_images(model, context, options)


def openrouter_images_api() -> ProviderImages:
    """返回 OpenRouter 图片生成 API 的 ProviderImages 实例。"""
    return cast(ProviderImages, _OpenRouterImagesProvider())