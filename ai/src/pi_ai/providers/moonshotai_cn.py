"""Moonshot AI CN Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .moonshotai_models import MOONSHOTAI_MODELS


def moonshotai_cn_provider() -> Any:
    """创建 Moonshot AI CN Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="moonshotai-cn",
            name="Moonshot AI CN",
            base_url="https://api.moonshot.cn/v1",
            models=list(MOONSHOTAI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
