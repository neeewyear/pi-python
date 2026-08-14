"""Cloudflare Workers AI Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .cloudflare_workers_ai_models import CLOUDFLARE_WORKERS_AI_MODELS


def cloudflare_workers_ai_provider() -> Any:
    """创建 Cloudflare Workers AI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="cloudflare-workers-ai",
            name="Cloudflare Workers AI",
            base_url="https://api.cloudflare.com/client/v4/accounts/.../ai/v1",
            models=list(CLOUDFLARE_WORKERS_AI_MODELS.values()),
            api=openai_completions_api(),
        )
    )
