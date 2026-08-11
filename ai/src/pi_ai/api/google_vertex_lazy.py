"""Google Vertex AI API 延迟加载（对应 ``google-vertex.lazy.ts``）。"""

from __future__ import annotations

from typing import Any

from .lazy import lazy_api


def google_vertex_api() -> Any:
    """返回 Google Vertex AI API 的 ProviderStreams 实例。"""
    from . import google_vertex  # 延迟导入

    return lazy_api(lambda: google_vertex)
