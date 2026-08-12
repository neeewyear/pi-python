"""语法高亮钩子（对标 origin_pi ``utils/syntax-highlight.ts``）。

用 Pygments 将代码块渲染为逐行 ANSI 序列，供 pi_tui 的
``MarkdownTheme.highlight_code`` 钩子使用。
"""

from __future__ import annotations


def highlight_code(code: str, lang: str | None) -> list[str]:
    """将代码渲染为带语法高亮的行列表。

    使用 Pygments（与 rich 同源），未知语言/解析失败时回退为纯文本行。
    每行以 ANSI reset 结尾，防止跨行颜色串扰。

    Args:
        code: 代码块原始文本。
        lang: 语言标识（如 "python"），None 时尝试自动猜测。

    Returns:
        逐行 ANSI 字符串列表。
    """
    try:
        from pygments import highlight  # type: ignore[import-untyped]
        from pygments.formatters import (  # type: ignore[import-untyped]
            Terminal256Formatter,
        )
        from pygments.lexers import (  # type: ignore[import-untyped]
            get_lexer_by_name,
            guess_lexer,
        )
    except ImportError:
        return code.split("\n")

    try:
        lexer = get_lexer_by_name(lang) if lang else guess_lexer(code)
    except Exception:
        return code.split("\n")

    try:
        formatter = Terminal256Formatter(bg="dark")
        ansi = highlight(code.rstrip("\n"), lexer, formatter)
    except Exception:
        return code.split("\n")

    lines = ansi.rstrip("\n").split("\n")
    result: list[str] = []
    for line in lines:
        if line and not line.endswith("\x1b[0m"):
            line = line + "\x1b[0m"
        result.append(line)
    return result
