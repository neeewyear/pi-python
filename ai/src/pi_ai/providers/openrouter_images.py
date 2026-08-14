"""OpenRouter Images Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openrouter_images_lazy import openrouter_images_api
from ..models import CreateProviderOptions, create_provider
from .openrouter_models import OPENROUTER_MODELS


def openrouter_images_provider() -> Any:
    """创建 OpenRouter Images Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="openrouter",
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            models=list(OPENROUTER_MODELS.values()),
            api=openrouter_images_api(),
        )
    )
