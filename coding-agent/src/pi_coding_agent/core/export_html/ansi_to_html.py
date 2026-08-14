"""ANSI 转 HTML。

将终端 ANSI 颜色/样式码转换为带内联样式的 HTML。
支持：
- 标准前景色（30-37）和亮色变体（90-97）
- 标准背景色（40-47）和亮色变体（100-107）
- 256 色调色板（38;5;N 和 48;5;N）
- RGB 真彩色（38;2;R;G;B 和 48;2;R;G;B）
- 文本样式：bold（1）、dim（2）、italic（3）、underline（4）
- Reset（0）
"""

from __future__ import annotations

import re

# 标准 ANSI 调色板（0-15）
ANSI_COLORS: list[str] = [
    "#000000",  # 0: black
    "#800000",  # 1: red
    "#008000",  # 2: green
    "#808000",  # 3: yellow
    "#000080",  # 4: blue
    "#800080",  # 5: magenta
    "#008080",  # 6: cyan
    "#c0c0c0",  # 7: white
    "#808080",  # 8: bright black
    "#ff0000",  # 9: bright red
    "#00ff00",  # 10: bright green
    "#ffff00",  # 11: bright yellow
    "#0000ff",  # 12: bright blue
    "#ff00ff",  # 13: bright magenta
    "#00ffff",  # 14: bright cyan
    "#ffffff",  # 15: bright white
]


def color256_to_hex(index: int) -> str:
    """将 256 色索引转换为十六进制颜色。

    标准色（0-15）：直接查表。
    色立方（16-231）：6x6x6 = 216 色。
    灰度（232-255）：24 级灰度。
    """
    # 标准色（0-15）
    if index < 16:
        return ANSI_COLORS[index]

    # 色立方（16-231）：6x6x6 = 216 色
    if index < 232:
        cube_index = index - 16
        r = cube_index // 36
        g = (cube_index % 36) // 6
        b = cube_index % 6

        def to_component(n: int) -> int:
            return 0 if n == 0 else 55 + n * 40

        def to_hex(n: int) -> str:
            return f"{to_component(n):02x}"

        return f"#{to_hex(r)}{to_hex(g)}{to_hex(b)}"

    # 灰度（232-255）：24 级灰度
    gray = 8 + (index - 232) * 10
    gray_hex = f"{gray:02x}"
    return f"#{gray_hex}{gray_hex}{gray_hex}"


def escape_html(text: str) -> str:
    """转义 HTML 特殊字符。"""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#039;")
    )


class TextStyle:
    """文本样式。"""

    fg: str | None
    bg: str | None
    bold: bool
    dim: bool
    italic: bool
    underline: bool

    def __init__(self) -> None:
        self.fg = None
        self.bg = None
        self.bold = False
        self.dim = False
        self.italic = False
        self.underline = False


def style_to_inline_css(style: TextStyle) -> str:
    """将 TextStyle 转换为内联 CSS 字符串。"""
    parts: list[str] = []
    if style.fg is not None:
        parts.append(f"color:{style.fg}")
    if style.bg is not None:
        parts.append(f"background-color:{style.bg}")
    if style.bold:
        parts.append("font-weight:bold")
    if style.dim:
        parts.append("opacity:0.6")
    if style.italic:
        parts.append("font-style:italic")
    if style.underline:
        parts.append("text-decoration:underline")
    return ";".join(parts)


def has_style(style: TextStyle) -> bool:
    """检查是否有任何样式设置。"""
    return (
        style.fg is not None
        or style.bg is not None
        or style.bold
        or style.dim
        or style.italic
        or style.underline
    )


def apply_sgr_code(params: list[int], style: TextStyle) -> None:
    """解析 ANSI SGR（Select Graphic Rendition）码并更新样式。"""
    i = 0
    while i < len(params):
        code = params[i]

        if code == 0:
            # Reset all
            style.fg = None
            style.bg = None
            style.bold = False
            style.dim = False
            style.italic = False
            style.underline = False
        elif code == 1:
            style.bold = True
        elif code == 2:
            style.dim = True
        elif code == 3:
            style.italic = True
        elif code == 4:
            style.underline = True
        elif code == 22:
            # Reset bold/dim
            style.bold = False
            style.dim = False
        elif code == 23:
            style.italic = False
        elif code == 24:
            style.underline = False
        elif 30 <= code <= 37:
            # 标准前景色
            style.fg = ANSI_COLORS[code - 30]
        elif code == 38:
            # 扩展前景色
            if i + 2 < len(params) and params[i + 1] == 5:
                # 256-color: 38;5;N
                style.fg = color256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                # RGB: 38;2;R;G;B
                r = params[i + 2]
                g = params[i + 3]
                b = params[i + 4]
                style.fg = f"rgb({r},{g},{b})"
                i += 4
        elif code == 39:
            # 默认前景
            style.fg = None
        elif 40 <= code <= 47:
            # 标准背景色
            style.bg = ANSI_COLORS[code - 40]
        elif code == 48:
            # 扩展背景色
            if i + 2 < len(params) and params[i + 1] == 5:
                # 256-color: 48;5;N
                style.bg = color256_to_hex(params[i + 2])
                i += 2
            elif i + 4 < len(params) and params[i + 1] == 2:
                # RGB: 48;2;R;G;B
                r = params[i + 2]
                g = params[i + 3]
                b = params[i + 4]
                style.bg = f"rgb({r},{g},{b})"
                i += 4
        elif code == 49:
            # 默认背景
            style.bg = None
        elif 90 <= code <= 97:
            # 亮前景色
            style.fg = ANSI_COLORS[code - 90 + 8]
        elif 100 <= code <= 107:
            # 亮背景色
            style.bg = ANSI_COLORS[code - 100 + 8]
        # 忽略未识别的码

        i += 1


# 匹配 ANSI 转义序列：ESC[ 后跟参数，以 'm' 结尾
ANSI_REGEX = re.compile(r"\x1b\[([\d;]*)m")


def ansi_to_html(text: str) -> str:
    """将 ANSI 转义文本转换为带内联样式的 HTML。"""
    style = TextStyle()
    result = ""
    last_index = 0
    in_span = False

    for match in ANSI_REGEX.finditer(text):
        # 添加此转义序列之前的文本
        before_text = text[last_index : match.start()]
        if before_text:
            result += escape_html(before_text)

        # 解析 SGR 参数
        param_str = match.group(1)
        if param_str:
            params = [int(p) if p else 0 for p in param_str.split(";")]
        else:
            params = [0]

        # 关闭现有 span
        if in_span:
            result += "</span>"
            in_span = False

        # 应用码
        apply_sgr_code(params, style)

        # 如果有样式则打开新 span
        if has_style(style):
            result += f'<span style="{style_to_inline_css(style)}">'
            in_span = True

        last_index = match.end()

    # 添加剩余文本
    remaining_text = text[last_index:]
    if remaining_text:
        result += escape_html(remaining_text)

    # 关闭任何打开的 span
    if in_span:
        result += "</span>"

    return result


def ansi_lines_to_html(lines: list[str]) -> str:
    """将 ANSI 转义行数组转换为 HTML。

    每行包裹在 div 元素中。
    """
    return "".join(
        f'<div class="ansi-line">{ansi_to_html(line) or "&nbsp;"}</div>'
        for line in lines
    )