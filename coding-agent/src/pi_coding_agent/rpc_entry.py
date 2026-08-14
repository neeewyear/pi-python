"""RPC 入口。

解析命令行参数，配置 HTTP 分发器，以 RPC 模式启动主循环。
"""

from __future__ import annotations

import asyncio
import os
import sys

from .config import APP_NAME
from .core.http_dispatcher import configure_http_dispatcher
from .main import main


def run_rpc_entry() -> None:
    """RPC 入口函数。

    设置环境变量、配置 HTTP 分发器，然后以 RPC 模式启动主循环。
    """
    # 设置进程标题（仅限 Unix）
    try:
        import ctypes

        libc = ctypes.CDLL("libc.dylib")
        libc.setproctitle.restype = None
        libc.setproctitle(ctypes.c_char_p(f"{APP_NAME}-rpc".encode()))
    except Exception:
        pass

    os.environ.setdefault("PI_CODING_AGENT", "true")
    os.environ.setdefault("AI_AGENT", "pi")
    sys.warnoptions.insert(0, "ignore")

    configure_http_dispatcher()

    asyncio.run(main(["--mode", "rpc", *sys.argv[2:]]))


__all__: list[str] = [
    "run_rpc_entry",
]
