"""交互模式主题（对标 origin_pi ``modes/interactive/theme/theme.ts``）。

提供 Markdown / Editor / SelectList 的 ANSI 主题工厂，支持暗色与亮色两套，
默认暗色（与多数终端配色一致）。语法高亮钩子注入 ``highlight_code``。
"""

from __future__ import annotations

from typing import Callable

from pi_tui import EditorTheme, MarkdownTheme, SelectListTheme

from .syntax_highlight import highlight_code

# ---------------------------------------------------------------------------
# ANSI 颜色辅助
# ---------------------------------------------------------------------------


def _dim(s: str) -> str:
    return f"\x1b[2m{s}\x1b[22m"


def _bold(s: str) -> str:
    return f"\x1b[1m{s}\x1b[22m"


def _italic(s: str) -> str:
    return f"\x1b[3m{s}\x1b[23m"


def _cyan(s: str) -> str:
    return f"\x1b[36m{s}\x1b[39m"


def _green(s: str) -> str:
    return f"\x1b[32m{s}\x1b[39m"


def _yellow(s: str) -> str:
    return f"\x1b[33m{s}\x1b[39m"


def _red(s: str) -> str:
    return f"\x1b[31m{s}\x1b[39m"


def _blue(s: str) -> str:
    return f"\x1b[34m{s}\x1b[39m"


def _magenta(s: str) -> str:
    return f"\x1b[35m{s}\x1b[39m"


def _reset(s: str) -> str:
    return f"\x1b[0m{s}\x1b[0m"


# ---------------------------------------------------------------------------
# 主题工厂
# ---------------------------------------------------------------------------

StyleFn = Callable[[str], str]


def get_markdown_theme(dark: bool = True) -> MarkdownTheme:
    """构建 Markdown 主题（含代码语法高亮钩子）。

    Args:
        dark: True 使用暗色系配色，False 使用亮色系。

    Returns:
        配置好的 MarkdownTheme。
    """
    code_style: StyleFn = (
        (lambda s: f"\x1b[38;5;136m{s}\x1b[39m") if dark else _red
    )
    code_block_style: StyleFn = (
        (lambda s: f"\x1b[38;5;136m{s}\x1b[39m") if dark else _red
    )
    link_style: StyleFn = _cyan if dark else _blue
    bullet_style: StyleFn = _cyan if dark else _magenta
    quote_style: StyleFn = _dim if dark else _dim
    hr_style: StyleFn = _dim if dark else _dim

    return MarkdownTheme(
        heading=_bold,
        link=link_style,
        link_url=lambda s: _dim(s),
        code=code_style,
        code_block=code_block_style,
        code_block_border=_dim,
        quote=quote_style,
        quote_border=_dim,
        hr=hr_style,
        list_bullet=bullet_style,
        bold=_bold,
        italic=_italic,
        strikethrough=lambda s: f"\x1b[9m{s}\x1b[29m",
        underline=lambda s: f"\x1b[4m{s}\x1b[24m",
        highlight_code=highlight_code,
        code_block_indent="  ",
    )


def get_editor_theme(dark: bool = True) -> EditorTheme:
    """构建编辑器主题。"""
    select_theme = SelectListTheme(
        selected_text=_cyan if dark else _blue,
        description=_dim,
        scroll_info=_dim,
        no_match=_dim,
    )
    return EditorTheme(
        border_color=_dim,
        select_list=select_theme,
    )
