"""Kimi Coding Provider（对应 ``kimi-coding.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .kimi_coding_models import KIMI_CODING_MODELS


def kimi_coding_provider() -> Any:
    """创建 Kimi Coding Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="kimi-coding",
            name="Kimi Coding",
            base_url="https://kimi-coding.com/v1",
            models=list(KIMI_CODING_MODELS.values()),
            api=openai_completions_api(),
        )
    )
