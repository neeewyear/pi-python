"""JSON 工具。

提供 ``safe_json_parse`` 函数，用于安全解析 JSON 字符串。
"""

from __future__ import annotations

import json
from typing import Any


def safe_json_parse(text: str) -> Any:
    """安全解析 JSON 字符串。

    尝试解析 JSON 字符串，失败时返回 ``None`` 而非抛出异常。

    Args:
        text: JSON 字符串。

    Returns:
        解析后的 Python 对象，解析失败返回 ``None``。
    """
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None