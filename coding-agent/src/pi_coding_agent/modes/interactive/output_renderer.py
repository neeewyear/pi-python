"""输出渲染器——代码块检测与语法高亮。

使用 Rich 的 Syntax 对 ```lang ... ``` 代码块应用 Pygments 语法高亮。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from rich.syntax import Syntax

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rich.console import RenderableType

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_CODE_BLOCK_PATTERN = re.compile(r"```(\w*)\n(.*?)```", re.DOTALL)
"""匹配 markdown 代码块的正则表达式。"""

_SYNTAX_THEME = "monokai"
"""Pygments 语法高亮主题。"""

_MAX_INLINE_LENGTH = 80
"""行内代码块的最大渲染宽度。"""

# ---------------------------------------------------------------------------
# OutputRenderer
# ---------------------------------------------------------------------------


class OutputRenderer:
    """输出渲染器，将纯文本转换为带语法高亮的 Rich 可渲染对象列表。

    用法::

        renderer = OutputRenderer()
        items = renderer.render_message(text)
        for item in items:
            rich_log.write(item)
    """

    def render_message(self, text: str) -> list[RenderableType]:
        """将文本渲染为 Rich 可渲染对象列表。

        对 markdown 代码块（`` ```lang ... ``` ``）应用 Pygments 语法高亮，
        非代码文本保持原样。

        Args:
            text: 要渲染的文本。

        Returns:
            Rich 可渲染对象列表，可直接传递给 ``RichLog.write()``。
        """
        items: list[RenderableType] = []
        last_end = 0

        for match in _CODE_BLOCK_PATTERN.finditer(text):
            # 代码块前的文本
            before = text[last_end : match.start()]
            if before:
                items.append(before)

            # 代码块 —— 应用语法高亮
            lang = (match.group(1) or "text").strip().lower()
            code = match.group(2)
            try:
                syntax = Syntax(
                    code,
                    lang,
                    theme=_SYNTAX_THEME,
                    line_numbers=True,
                    word_wrap=True,
                    indent_guides=True,
                )
                items.append(syntax)
            except Exception:
                # 语言未知或语法错误时，以纯文本回退
                items.append(f"```{lang}\n{code}```")

            last_end = match.end()

        # 剩余文本
        after = text[last_end:]
        if after:
            items.append(after)

        return items

    @staticmethod
    def find_code_blocks(text: str) -> list[dict[str, Any]]:
        """查找文本中的所有代码块，返回元信息列表。

        Returns:
            ``[{ "lang": str, "code": str, "start": int, "end": int }, ...]``
        """
        blocks: list[dict[str, Any]] = []
        for match in _CODE_BLOCK_PATTERN.finditer(text):
            blocks.append(
                {
                    "lang": (match.group(1) or "text").strip().lower(),
                    "code": match.group(2),
                    "start": match.start(),
                    "end": match.end(),
                }
            )
        return blocks

    @staticmethod
    def count_code_blocks(text: str) -> int:
        """统计文本中代码块的数量。"""
        return len(_CODE_BLOCK_PATTERN.findall(text))


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------

_renderer_instance: OutputRenderer | None = None


def get_renderer() -> OutputRenderer:
    """获取全局 OutputRenderer 单例。"""
    global _renderer_instance  # noqa: F824
    if _renderer_instance is None:
        _renderer_instance = OutputRenderer()
    return _renderer_instance


def render_text(text: str) -> Sequence[RenderableType]:
    """便捷函数：渲染文本。"""
    return get_renderer().render_message(text)
