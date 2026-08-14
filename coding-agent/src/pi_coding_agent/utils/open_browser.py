"""浏览器打开工具。

提供 ``open_browser`` 函数，用于在系统默认浏览器中打开 URL。
"""

from __future__ import annotations

import webbrowser


def open_browser(url: str) -> None:
    """在系统默认浏览器中打开 URL。

    Args:
        url: 要打开的 URL 地址。
    """
    webbrowser.open(url)