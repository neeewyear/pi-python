"""User-Agent 字符串工具（对应 TS ``utils/pi-user-agent.ts``）。

提供 ``get_pi_user_agent`` 函数，用于获取 Pi 应用的 User-Agent 字符串。
"""

from __future__ import annotations

import platform
import sys

from pi_coding_agent.config import VERSION


def get_pi_user_agent() -> str:
    """获取 Pi 应用的 User-Agent 字符串。

    格式: ``pi/<version> (<system>; <python/<version>>; <arch>)``

    Returns:
        User-Agent 字符串。
    """
    system = platform.system().lower()
    arch = platform.machine()
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    return f"pi/{VERSION} ({system}; python/{python_version}; {arch})"