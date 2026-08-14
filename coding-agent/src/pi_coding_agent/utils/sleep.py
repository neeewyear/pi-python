"""异步睡眠工具。

提供 ``sleep`` 函数，用于异步等待指定毫秒数。
"""

from __future__ import annotations

import asyncio


async def sleep(ms: int) -> None:
    """异步睡眠指定毫秒数。

    Args:
        ms: 睡眠毫秒数。
    """
    await asyncio.sleep(ms / 1000)