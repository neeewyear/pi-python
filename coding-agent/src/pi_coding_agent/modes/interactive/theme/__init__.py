"""交互模式主题包。

提供 Markdown / Editor 主题工厂与语法高亮钩子。
"""

from .syntax_highlight import highlight_code
from .theme import get_editor_theme, get_markdown_theme

__all__ = ["get_editor_theme", "get_markdown_theme", "highlight_code"]
