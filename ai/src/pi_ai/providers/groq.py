"""Groq Provider（对应 ``groq.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .groq_models import GROQ_MODELS


def groq_provider() -> Any:
    """创建 Groq Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="groq",
            name="Groq",
            base_url="https://api.groq.com/openai/v1",
            models=list(GROQ_MODELS.values()),
            api=openai_completions_api(),
        )
    )
