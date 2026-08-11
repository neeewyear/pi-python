"""OpenCode Go Provider（对应 ``opencode-go.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .opencode_go_models import OPENCODE_GO_MODELS


def opencode_go_provider() -> Any:
    """创建 OpenCode Go Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="opencode-go",
            name="OpenCode Go",
            base_url="https://go.opencode.ai/v1",
            models=list(OPENCODE_GO_MODELS.values()),
            api=openai_completions_api(),
        )
    )
