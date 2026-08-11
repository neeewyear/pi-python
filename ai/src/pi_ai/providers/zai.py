"""ZAI Provider（对应 ``zai.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .zai_models import ZAI_MODELS


def zai_provider() -> Any:
    """创建 ZAI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="zai",
            name="ZAI",
            base_url="https://api.zai.com/v1",
            models=list(ZAI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
