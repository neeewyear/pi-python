"""MIME 类型工具（对应 TS ``utils/mime.ts``）。

提供 ``get_mime_type`` 函数，用于获取文件的 MIME 类型。
"""

from __future__ import annotations

import mimetypes
from pathlib import Path


def get_mime_type(path: str) -> str:
    """获取文件的 MIME 类型。

    基于文件扩展名猜测 MIME 类型。如果无法识别，
    返回 ``application/octet-stream``。

    Args:
        path: 文件路径或文件名。

    Returns:
        文件的 MIME 类型字符串。
    """
    mime_type, _ = mimetypes.guess_type(str(Path(path)))
    return mime_type or "application/octet-stream"