"""Pi Messages API 延迟加载。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def pi_messages_api() -> Any:
    """返回 Pi Messages API 的 ProviderStreams 实例。"""
    from . import pi_messages  # 延迟导入

    return lazy_api(lambda: pi_messages)
