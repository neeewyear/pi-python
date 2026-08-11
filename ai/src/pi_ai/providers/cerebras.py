"""Cerebras Provider（对应 ``cerebras.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .cerebras_models import CEREBRAS_MODELS


def cerebras_provider() -> Any:
    """创建 Cerebras Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="cerebras",
            name="Cerebras",
            base_url="https://api.cerebras.ai/v1",
            models=list(CEREBRAS_MODELS.values()),
            api=openai_completions_api(),
        )
    )
