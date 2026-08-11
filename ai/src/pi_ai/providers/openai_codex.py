"""OpenAI Codex Provider（对应 ``openai-codex.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_codex_responses_lazy import openai_codex_responses_api
from ..models import CreateProviderOptions, create_provider
from .openai_codex_models import OPENAI_CODEX_MODELS


def openai_codex_provider() -> Any:
    """创建 OpenAI Codex Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="openai-codex",
            name="OpenAI Codex",
            base_url="https://chatgpt.com/backend-api",
            models=list(OPENAI_CODEX_MODELS.values()),
            api=openai_codex_responses_api(),
        )
    )
