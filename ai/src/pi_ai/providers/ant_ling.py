"""Ant Ling Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .ant_ling_models import ANT_LING_MODELS


def ant_ling_provider() -> Any:
    """创建 Ant Ling Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="ant-ling",
            name="Ant Ling",
            base_url="https://api.antling.com/v1",
            models=list(ANT_LING_MODELS.values()),
            api=openai_completions_api(),
        )
    )
