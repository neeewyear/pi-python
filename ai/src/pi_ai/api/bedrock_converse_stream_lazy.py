"""Bedrock Converse Stream API 延迟加载（对应 ``bedrock-converse-stream.lazy.ts``）。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def bedrock_converse_stream_api() -> Any:
    """返回 Bedrock Converse Stream API 的 ProviderStreams 实例。"""
    from . import bedrock_converse_stream  # 延迟导入

    return lazy_api(lambda: bedrock_converse_stream)
