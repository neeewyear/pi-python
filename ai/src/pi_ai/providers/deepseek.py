"""DeepSeek Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .deepseek_models import DEEPSEEK_MODELS


def deepseek_provider() -> Any:
    """创建 DeepSeek Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="deepseek",
            name="DeepSeek",
            base_url="https://api.deepseek.com",
            models=list(DEEPSEEK_MODELS.values()),
            api=openai_completions_api(),
        )
    )
