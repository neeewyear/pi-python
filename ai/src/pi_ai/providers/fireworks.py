"""Fireworks AI Provider（对应 ``fireworks.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .fireworks_models import FIREWORKS_MODELS


def fireworks_provider() -> Any:
    """创建 Fireworks AI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="fireworks",
            name="Fireworks AI",
            base_url="https://api.fireworks.ai/inference/v1",
            models=list(FIREWORKS_MODELS.values()),
            api=openai_completions_api(),
        )
    )
