"""模块级默认 streamFn。  

宿主可在此安装默认模型运行时的 stream 函数，避免 pi-agent 依赖具体 provider。
"""

from __future__ import annotations

from .types import StreamFn

_default_stream_fn: StreamFn | None = None


def set_default_stream_fn(stream_fn: StreamFn | None) -> None:
    """配置 Agent 与低层循环在调用方省略 ``streamFn`` 时使用的回退函数。"""
    global _default_stream_fn
    _default_stream_fn = stream_fn


def get_default_stream_fn() -> StreamFn:
    """返回默认 streamFn；未配置时抛出错误。"""
    if _default_stream_fn is None:
        raise RuntimeError(
            "No default stream function configured. "
            "Pass streamFn explicitly or call set_default_stream_fn()."
        )
    return _default_stream_fn


__all__ = [
    "get_default_stream_fn",
    "set_default_stream_fn",
]