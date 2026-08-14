"""文本处理工具。"""

from __future__ import annotations

from ..types import TextContent, ImageContent


Content = TextContent | ImageContent | dict[str, object]


def content_text(
    content: str | list[Content], separator: str = "\n"
) -> str:
    """从消息内容中提取并连接文本。"""
    if isinstance(content, str):
        return content
    text_parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    text_parts.append(text)
        elif block.type == "text":
            text_parts.append(block.text)
    return separator.join(text_parts)