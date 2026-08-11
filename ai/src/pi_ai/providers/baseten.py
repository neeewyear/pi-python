"""Baseten Provider（对应 ``baseten.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .baseten_models import BASETEN_MODELS


def baseten_provider() -> Any:
    """创建 Baseten Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="baseten",
            name="Baseten",
            base_url="https://bridge.baseten.co/v1",
            models=list(BASETEN_MODELS.values()),
            api=openai_completions_api(),
        )
    )
