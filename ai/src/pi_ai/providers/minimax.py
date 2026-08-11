"""MiniMax Provider（对应 ``minimax.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .minimax_models import MINIMAX_MODELS


def minimax_provider() -> Any:
    """创建 MiniMax Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="minimax",
            name="MiniMax",
            base_url="https://api.minimax.chat/v1",
            models=list(MINIMAX_MODELS.values()),
            api=openai_completions_api(),
        )
    )
