"""Xiaomi Token Plan CN Provider（对应 ``xiaomi-token-plan-cn.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .xiaomi_token_plan_cn_models import XIAOMI_TOKEN_PLAN_CN_MODELS


def xiaomi_token_plan_cn_provider() -> Any:
    """创建 Xiaomi Token Plan CN Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="xiaomi-token-plan-cn",
            name="Xiaomi Token Plan CN",
            base_url="(from config)",
            models=list(XIAOMI_TOKEN_PLAN_CN_MODELS.values()),
            api=openai_completions_api(),
        )
    )
