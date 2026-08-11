"""DeepSeek API 的 StreamFn 实现（从 ``pi_ai.deepseek_provider`` 再导出）。

保留 ``pi_agent.deepseek_provider`` 模块作为向后兼容入口。
"""

from __future__ import annotations

from pi_ai.deepseek_provider import (  # noqa: F401
    DeepSeekModel,
    DeepSeekStreamFn,
    create_deepseek_stream_fn,
)

__all__ = [
    "DeepSeekModel",
    "DeepSeekStreamFn",
    "create_deepseek_stream_fn",
]