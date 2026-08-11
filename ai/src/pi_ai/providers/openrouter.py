"""OpenRouter Provider（对应 ``openrouter.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .openrouter_models import OPENROUTER_MODELS


def openrouter_provider() -> Any:
    """创建 OpenRouter Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="openrouter",
            name="OpenRouter",
            base_url="https://openrouter.ai/api/v1",
            models=list(OPENROUTER_MODELS.values()),
            api=openai_completions_api(),
        )
    )
