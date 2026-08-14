"""Xiaomi Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .xiaomi_models import XIAOMI_MODELS


def xiaomi_provider() -> Any:
    """创建 Xiaomi Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xiaomi",
            name="Xiaomi",
            base_url="https://api.xiaomi.com/v1",
            models=list(XIAOMI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
