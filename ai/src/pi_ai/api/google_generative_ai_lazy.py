"""Google Generative AI API 延迟加载（对应 ``google-generative-ai.lazy.ts``）。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def google_generative_ai_api() -> Any:
    """返回 Google Generative AI API 的 ProviderStreams 实例。"""
    from . import google_generative_ai  # 延迟导入

    return lazy_api(lambda: google_generative_ai)
