"""OpenAI Completions API 延迟加载。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def openai_completions_api() -> Any:
    """返回 OpenAI Completions API 的 ProviderStreams 实例。"""
    from . import openai_completions  # 延迟导入

    return lazy_api(lambda: openai_completions)
