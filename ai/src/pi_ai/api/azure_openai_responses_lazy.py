"""Azure OpenAI Responses API 延迟加载。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def azure_openai_responses_api() -> Any:
    """返回 Azure OpenAI Responses API 的 ProviderStreams 实例。"""
    from . import azure_openai_responses  # 延迟导入

    return lazy_api(lambda: azure_openai_responses)
