"""Xiaomi Token Plan AMS Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .xiaomi_token_plan_ams_models import XIAOMI_TOKEN_PLAN_AMS_MODELS


def xiaomi_token_plan_ams_provider() -> Any:
    """创建 Xiaomi Token Plan AMS Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xiaomi-token-plan-ams",
            name="Xiaomi Token Plan AMS",
            base_url="(from config)",
            models=list(XIAOMI_TOKEN_PLAN_AMS_MODELS.values()),
            api=openai_completions_api(),
        )
    )
