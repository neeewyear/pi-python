"""OpenCode Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .opencode_models import OPENCODE_MODELS


def opencode_provider() -> Any:
    """创建 OpenCode Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="opencode",
            name="OpenCode",
            base_url="https://api.opencode.ai/v1",
            models=list(OPENCODE_MODELS.values()),
            api=openai_completions_api(),
        )
    )
