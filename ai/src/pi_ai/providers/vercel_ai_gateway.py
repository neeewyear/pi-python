"""Vercel AI Gateway Provider（对应 ``vercel-ai-gateway.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .vercel_ai_gateway_models import VERCEL_AI_GATEWAY_MODELS


def vercel_ai_gateway_provider() -> Any:
    """创建 Vercel AI Gateway Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="vercel-ai-gateway",
            name="Vercel AI Gateway",
            base_url="https://gateway.vercel.ai/v1",
            models=list(VERCEL_AI_GATEWAY_MODELS.values()),
            api=openai_completions_api(),
        )
    )
