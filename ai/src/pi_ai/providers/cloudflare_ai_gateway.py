"""Cloudflare AI Gateway Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .cloudflare_ai_gateway_models import CLOUDFLARE_AI_GATEWAY_MODELS


def cloudflare_ai_gateway_provider() -> Any:
    """创建 Cloudflare AI Gateway Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="cloudflare-ai-gateway",
            name="Cloudflare AI Gateway",
            base_url="https://gateway.ai.cloudflare.com/v1/.../compat",
            models=list(CLOUDFLARE_AI_GATEWAY_MODELS.values()),
            api=openai_completions_api(),
        )
    )
