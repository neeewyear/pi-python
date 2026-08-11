"""``python -m src.pi_coding_agent`` 入口点。

避免从 ``__init__.py`` 导入 ``main`` 导致的循环导入问题。
"""
from __future__ import annotations

import asyncio
import sys

from .main import main

if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))