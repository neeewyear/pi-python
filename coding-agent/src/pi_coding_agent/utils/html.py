"""HTML 处理工具（对应 TS ``utils/html.ts``）。

提供 ``escape_html`` 函数，用于 HTML 转义。
"""

from __future__ import annotations

import html as _html_module


def escape_html(text: str) -> str:
    """转义 HTML 特殊字符。

    将 ``&``、``<``、``>``、``"``、``'`` 转义为对应的 HTML 实体。

    Args:
        text: 要转义的文本。

    Returns:
        转义后的 HTML 文本。
    """
    return _html_module.escape(text, quote=True)