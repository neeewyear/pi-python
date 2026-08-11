"""Mistral Provider（对应 ``mistral.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.mistral_conversations_lazy import mistral_conversations_api
from ..models import CreateProviderOptions, create_provider
from .mistral_models import MISTRAL_MODELS


def mistral_provider() -> Any:
    """创建 Mistral Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="mistral",
            name="Mistral",
            base_url="https://api.mistral.ai/v1",
            models=list(MISTRAL_MODELS.values()),
            api=mistral_conversations_api(),
        )
    )
