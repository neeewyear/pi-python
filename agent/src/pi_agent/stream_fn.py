"""模块级默认 streamFn（从 ``pi_ai.stream_fn`` 再导出）。

保留 ``pi_agent.stream_fn`` 模块作为向后兼容入口。
"""

from __future__ import annotations

from pi_ai.stream_fn import get_default_stream_fn, set_default_stream_fn

__all__ = [
    "get_default_stream_fn",
    "set_default_stream_fn",
]