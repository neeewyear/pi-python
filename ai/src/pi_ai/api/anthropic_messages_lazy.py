"""Anthropic Messages API 延迟加载。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def anthropic_messages_api() -> Any:
    """返回 Anthropic Messages API 的 ProviderStreams 实例。"""
    from . import anthropic_messages  # 延迟导入

    return lazy_api(lambda: anthropic_messages)
