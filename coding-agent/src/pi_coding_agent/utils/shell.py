"""Shell 配置工具（对应 TS ``utils/shell.ts``）。

提供 ``get_shell_config`` 和 ``sanitize_binary_output`` 函数。
"""

from __future__ import annotations

import os
import platform
from pathlib import Path


def get_shell_config() -> dict[str, str | list[str]]:
    """获取当前平台的 shell 配置。

    返回包含 shell 路径和参数列表的字典：
    - macOS: ``/bin/zsh`` (或 ``/bin/bash`` 作为降级)
    - Linux: ``/bin/bash`` (或 ``sh`` 作为降级)
    - Windows: 通过 ``COMSPEC`` 环境变量获取

    Returns:
        包含 ``shell`` 和 ``args`` 键的字典。
    """
    system = platform.system()

    if system == "Windows":
        shell = os.environ.get("COMSPEC", "cmd.exe")
        return {"shell": shell, "args": ["/c"]}

    # macOS 默认使用 zsh
    if system == "Darwin":
        if Path("/bin/zsh").exists():
            return {"shell": "/bin/zsh", "args": ["-c"]}
        if Path("/bin/bash").exists():
            return {"shell": "/bin/bash", "args": ["-c"]}
        return {"shell": "/bin/sh", "args": ["-c"]}

    # Linux 默认使用 bash
    if Path("/bin/bash").exists():
        return {"shell": "/bin/bash", "args": ["-c"]}

    bash_on_path = _find_bash_on_path()
    if bash_on_path:
        return {"shell": bash_on_path, "args": ["-c"]}

    return {"shell": "/bin/sh", "args": ["-c"]}


def sanitize_binary_output(text: str) -> str:
    """清理二进制输出。

    移除可能导致显示问题的字符：
    - 控制字符（保留 Tab、换行、回车）
    - Unicode 格式字符
    - 未定义的码点

    Args:
        text: 要清理的文本。

    Returns:
        清理后的文本。
    """
    result: list[str] = []
    for char in text:
        code = ord(char)

        # 保留 Tab、换行、回车
        if code in (0x09, 0x0A, 0x0D):
            result.append(char)
            continue

        # 过滤控制字符 (0x00-0x1F)
        if code <= 0x1F:
            continue

        # 过滤 Unicode 格式字符
        if 0xFFF9 <= code <= 0xFFFB:
            continue

        result.append(char)

    return "".join(result)


def _find_bash_on_path() -> str | None:
    """在 PATH 中查找 bash。"""
    path_dirs = os.environ.get("PATH", "").split(os.pathsep)
    for directory in path_dirs:
        bash_path = Path(directory) / "bash"
        if bash_path.exists():
            return str(bash_path)
    return None
