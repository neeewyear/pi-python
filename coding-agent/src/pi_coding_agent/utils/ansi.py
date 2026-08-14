"""ANSI 转义码处理工具。

提供 ``strip_ansi`` 函数，用于移除文本中的 ANSI 转义序列。
"""

from __future__ import annotations

import re

# ANSI 转义序列匹配模式
# 来自 chalk/ansi-regex 的 Python 移植
# 匹配 OSC 序列（ESC ] ... ST）和 CSI 序列（ESC [ ... 最终字节）
_ANSI_PATTERN = re.compile(
    r"(?:\u001B\][\s\S]*?(?:\u0007|\u001B\\)|\u009C)"
    r"|"
    r"(?:\u001B|\u009B)[\[\]()#;?]*(?:\d{1,4}(?:[;:]\d{0,4})*)?[\dA-PR-TZcf-nq-uy=><~]"
)


def strip_ansi(text: str) -> str:
    """移除文本中的 ANSI 转义序列。

    Args:
        text: 包含 ANSI 转义码的文本。

    Returns:
        移除所有 ANSI 转义序列后的纯文本。
    """
    if "\u001B" not in text and "\u009B" not in text:
        return text
    return _ANSI_PATTERN.sub("", text)