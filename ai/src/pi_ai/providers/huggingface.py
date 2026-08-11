"""HuggingFace Provider（对应 ``huggingface.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .huggingface_models import HUGGINGFACE_MODELS


def huggingface_provider() -> Any:
    """创建 HuggingFace Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="huggingface",
            name="HuggingFace",
            base_url="https://api-inference.huggingface.co/v1",
            models=list(HUGGINGFACE_MODELS.values()),
            api=openai_completions_api(),
        )
    )
