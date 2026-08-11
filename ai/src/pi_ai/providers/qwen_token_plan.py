"""Qwen Token Plan Provider（对应 ``qwen-token-plan.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .qwen_token_plan_models import QWEN_TOKEN_PLAN_MODELS


def qwen_token_plan_provider() -> Any:
    """创建 Qwen Token Plan Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="qwen-token-plan",
            name="Qwen Token Plan",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            models=list(QWEN_TOKEN_PLAN_MODELS.values()),
            api=openai_completions_api(),
        )
    )
