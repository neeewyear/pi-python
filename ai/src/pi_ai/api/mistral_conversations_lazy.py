"""Mistral Conversations API 延迟加载（对应 ``mistral-conversations.lazy.ts``）。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def mistral_conversations_api() -> Any:
    """返回 Mistral Conversations API 的 ProviderStreams 实例。"""
    from . import mistral_conversations  # 延迟导入

    return lazy_api(lambda: mistral_conversations)
