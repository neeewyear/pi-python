"""ZAI Coding CN Provider（对应 ``zai-coding-cn.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .zai_coding_cn_models import ZAI_CODING_CN_MODELS


def zai_coding_cn_provider() -> Any:
    """创建 ZAI Coding CN Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="zai-coding-cn",
            name="ZAI Coding CN",
            base_url="https://api.zai.cn/v1",
            models=list(ZAI_CODING_CN_MODELS.values()),
            api=openai_completions_api(),
        )
    )
