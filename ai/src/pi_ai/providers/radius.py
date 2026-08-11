"""Radius gateway Provider（对应 ``radius.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.pi_messages_lazy import pi_messages_api
from ..models import CreateProviderOptions, create_provider
from .radius_config import (
    DEFAULT_RADIUS_GATEWAY,
    normalize_radius_gateway_url,
)


def radius_provider(
    *,
    provider_id: str = "radius",
    name: str = "Radius",
    gateway: str | None = None,
) -> Any:
    """创建 Radius gateway Provider。

    Args:
        provider_id: Provider ID（默认 "radius"）。
        name: 显示名称（默认 "Radius"）。
        gateway: 网关 URL（默认 ``DEFAULT_RADIUS_GATEWAY``）。

    Returns:
        Provider 实例。
    """
    resolved_gateway = normalize_radius_gateway_url(gateway or DEFAULT_RADIUS_GATEWAY)
    models: list[Any] = []
    streams = pi_messages_api()

    # 构建一个简单的 provider 对象
    provider_obj = create_provider(
        CreateProviderOptions(
            id=provider_id,
            name=name,
            base_url=resolved_gateway,
            models=models,
            api=streams,
        )
    )

    # 在 provider 对象上添加 refresh_models 支持
    # 使用闭包保存动态模型列表
    _dynamic_models: list[Any] = []

    original_get_models = provider_obj.get_models

    def _get_models() -> list[Any]:
        merged = list(models)
        merged_ids = {
            getattr(m, "model_id", None) or getattr(m, "id", None) for m in merged
        }
        for dm in _dynamic_models:
            dm_id = getattr(dm, "model_id", None) or getattr(dm, "id", None)
            if dm_id in merged_ids:
                for i, m in enumerate(merged):
                    if (
                        getattr(m, "model_id", None) or getattr(m, "id", None)
                    ) == dm_id:
                        merged[i] = dm
                        break
            else:
                merged.append(dm)
        return merged

    provider_obj.get_models = _get_models  # type: ignore[method-assign]

    return provider_obj
