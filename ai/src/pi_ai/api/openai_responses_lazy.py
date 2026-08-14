"""OpenAI Responses API 延迟加载。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def openai_responses_api() -> Any:
    """返回 OpenAI Responses API 的 ProviderStreams 实例。"""
    from . import openai_responses  # 延迟导入

    return lazy_api(lambda: openai_responses)
