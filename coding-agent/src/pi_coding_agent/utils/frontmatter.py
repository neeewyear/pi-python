"""Frontmatter 解析工具。

提供 ``strip_frontmatter`` 和 ``parse_frontmatter`` 函数，
用于解析 YAML frontmatter。
"""

from __future__ import annotations

import re
from typing import Any

import yaml

_FRONTMATTER_PATTERN = re.compile(
    r"^---\s*\n(.*?)\n---\s*\n?(.*)",
    re.DOTALL,
)


def strip_frontmatter(text: str) -> str:
    """移除 frontmatter。

    移除文本开头的 YAML frontmatter（``---`` 包围的部分），
    只返回正文内容。

    Args:
        text: 包含 frontmatter 的文本。

    Returns:
        移除 frontmatter 后的正文。
    """
    match = _FRONTMATTER_PATTERN.match(text)
    if match:
        return match.group(2)
    return text


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    """解析 frontmatter。

    解析文本开头的 YAML frontmatter，返回 frontmatter 字典和正文。

    Args:
        text: 包含 frontmatter 的文本。

    Returns:
        ``(frontmatter, body)`` 元组，其中 frontmatter 为解析后的字典，
        body 为正文内容。
    """
    match = _FRONTMATTER_PATTERN.match(text)
    if not match:
        return {}, text

    yaml_string = match.group(1)
    body = match.group(2)

    try:
        parsed = yaml.safe_load(yaml_string)
        if isinstance(parsed, dict):
            return parsed, body
        return {}, body
    except yaml.YAMLError:
        return {}, body