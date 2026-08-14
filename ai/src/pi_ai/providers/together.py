"""Together AI Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .together_models import TOGETHER_MODELS


def together_provider() -> Any:
    """创建 Together AI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="together",
            name="Together AI",
            base_url="https://api.together.xyz/v1",
            models=list(TOGETHER_MODELS.values()),
            api=openai_completions_api(),
        )
    )
