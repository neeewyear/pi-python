"""OpenAI Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_responses_lazy import openai_responses_api
from ..models import CreateProviderOptions, create_provider
from .openai_models import OPENAI_MODELS


def openai_provider() -> Any:
    """创建 OpenAI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="openai",
            name="OpenAI",
            base_url="https://api.openai.com/v1",
            models=list(OPENAI_MODELS.values()),
            api=openai_responses_api(),
        )
    )
