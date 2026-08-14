"""Xiaomi Token Plan SGP Provider。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .xiaomi_token_plan_sgp_models import XIAOMI_TOKEN_PLAN_SGP_MODELS


def xiaomi_token_plan_sgp_provider() -> Any:
    """创建 Xiaomi Token Plan SGP Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xiaomi-token-plan-sgp",
            name="Xiaomi Token Plan SGP",
            base_url="(from config)",
            models=list(XIAOMI_TOKEN_PLAN_SGP_MODELS.values()),
            api=openai_completions_api(),
        )
    )
