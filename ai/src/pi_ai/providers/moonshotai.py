"""Moonshot AI Provider（对应 ``moonshotai.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .moonshotai_models import MOONSHOTAI_MODELS


def moonshotai_provider() -> Any:
    """创建 Moonshot AI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="moonshotai",
            name="Moonshot AI",
            base_url="https://api.moonshot.cn/v1",
            models=list(MOONSHOTAI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
