"""NVIDIA Provider（对应 ``nvidia.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .nvidia_models import NVIDIA_MODELS


def nvidia_provider() -> Any:
    """创建 NVIDIA Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="nvidia",
            name="NVIDIA",
            base_url="https://integrate.api.nvidia.com/v1",
            models=list(NVIDIA_MODELS.values()),
            api=openai_completions_api(),
        )
    )
