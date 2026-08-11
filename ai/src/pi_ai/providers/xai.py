"""xAI Provider（对应 ``xai.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .xai_models import XAI_MODELS


def xai_provider() -> Any:
    """创建 xAI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xai",
            name="xAI",
            base_url="https://api.x.ai",
            models=list(XAI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
