"""Git 工具（对应 TS ``utils/git.ts``）。

提供 Git 仓库 URL 解析和操作。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class GitSource:
    """Git 源信息。"""

    url: str
    """Git 仓库 URL。"""
    ref: str | None = None
    """分支/标签/提交引用。"""
    subdir: str | None = None
    """仓库子目录。"""


def parse_git_url(source: str) -> GitSource | None:
    """解析 Git URL 字符串。

    Args:
        source: Git URL 字符串（如 ``https://github.com/user/repo.git`` 或 ``github:user/repo``）。

    Returns:
        ``GitSource`` 实例，解析失败返回 ``None``。
    """
    # 支持格式：
    # - https://github.com/user/repo.git
    # - git@github.com:user/repo.git
    # - github:user/repo
    # - https://github.com/user/repo/tree/branch

    patterns = [
        # github:user/repo 简写格式（可能后跟 #ref 和 /subdir）
        re.compile(
            r"^github:(?P<user>[^/#]+)/(?P<repo>[^/#]+?)(?:#(?P<ref>[^/]+))?(?:/(?P<subdir>.+))?$"
        ),
        # https://github.com/user/repo.git 标准格式
        re.compile(
            r"^https?://github\.com/(?P<user>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?(?:#(?P<ref>[^/]+))?(?:/(?P<subdir>.+))?$"
        ),
        # git@github.com:user/repo.git SSH 格式
        re.compile(
            r"^git@github\.com:(?P<user>[^/]+)/(?P<repo>[^/]+?)(?:\.git)?(?:#(?P<ref>[^/]+))?(?:/(?P<subdir>.+))?$"
        ),
    ]

    for pattern in patterns:
        match = pattern.match(source)
        if match:
            g = match.groupdict()
            return GitSource(
                url=source,
                ref=g.get("ref"),
                subdir=g.get("subdir"),
            )

    return None