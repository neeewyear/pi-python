"""路径解析与规范化工具。

提供 ``resolve_path``、``normalize_path``、``canonicalize_path``、
``is_local_path`` 和 ``mark_path_ignored_by_cloud_sync`` 函数。
"""

from __future__ import annotations

import os
import platform
import re
import subprocess
from typing import Any
from urllib.parse import urlparse

_FILE_URL_PATTERN = re.compile(r"^file://", re.IGNORECASE)
_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")

# 已知的非本地路径前缀
_NON_LOCAL_PREFIXES = ("npm:", "git:", "github:", "http:", "https:", "ssh:")


def resolve_path(
    path_str: str,
    base_dir: str | None = None,
    options: dict[str, Any] | None = None,
) -> str:
    """解析路径（展开 ``~`` 和 ``file://``）。

    将输入路径解析为绝对路径：
    - 展开 ``~`` 和 ``~user``
    - 处理 ``file://`` URL
    - 相对路径基于 ``base_dir`` 解析

    Args:
        path_str: 待解析的路径字符串。
        base_dir: 相对路径的基准目录，默认为当前工作目录。
        options: 可选配置字典，支持：
            - ``trim``: 是否去除首尾空白
            - ``expand_tilde``: 是否展开 ``~``（默认 True）
            - ``home_dir``: 用于 ``~`` 展开的用户主目录
            - ``strip_at_prefix``: 是否去除开头的 ``@``
            - ``normalize_unicode_spaces``: 是否规范化 Unicode 空格

    Returns:
        解析后的绝对路径。
    """
    opts = options or {}
    normalized = path_str

    # 去除首尾空白
    if opts.get("trim"):
        normalized = normalized.strip()

    # 规范化 Unicode 空格
    if opts.get("normalize_unicode_spaces"):
        normalized = _UNICODE_SPACES.sub(" ", normalized)

    # 去除 @ 前缀
    if opts.get("strip_at_prefix") and normalized.startswith("@"):
        normalized = normalized[1:]

    # 展开 ~
    expand_tilde = opts.get("expand_tilde", True)
    if expand_tilde:
        home_dir = opts.get("home_dir")
        if home_dir is not None:
            normalized = _expand_tilde_with_home(normalized, home_dir)
        else:
            normalized = os.path.expanduser(normalized)

    # 处理 file:// URL
    if _FILE_URL_PATTERN.match(normalized):
        parsed = urlparse(normalized)
        normalized = parsed.path

    # 解析为绝对路径
    if os.path.isabs(normalized):
        return os.path.normpath(normalized)

    base = base_dir if base_dir is not None else os.getcwd()
    return os.path.normpath(os.path.join(base, normalized))


def normalize_path(
    path_str: str,
    options: dict[str, Any] | None = None,
) -> str:
    """规范化路径。

    移除路径中的冗余分隔符和上级引用（``..``），
    并展开 ``~``。

    Args:
        path_str: 待规范化的路径字符串。
        options: 可选配置字典，支持 ``trim``。

    Returns:
        规范化后的路径。
    """
    result = path_str
    if options and options.get("trim"):
        result = result.strip()
    return os.path.normpath(os.path.expanduser(result))


def canonicalize_path(path: str) -> str:
    """规范化路径为绝对形式，跟随符号链接。

    解析符号链接到真实路径。如果解析失败（如路径不存在），
    回退到规范化后的绝对路径。

    Args:
        path: 待规范化的路径。

    Returns:
        规范化后的绝对路径。
    """
    try:
        return os.path.realpath(path)
    except OSError:
        return os.path.normpath(os.path.abspath(os.path.expanduser(path)))


def is_local_path(value: str) -> bool:
    """检查路径是否为本地路径。

    检查值是否以已知的非本地协议前缀开头（如 ``npm:``、``git:`` 等）。
    如果值不是非本地路径，则视为本地路径。

    Args:
        value: 待检查的路径字符串。

    Returns:
        如果是本地路径返回 ``True``，否则返回 ``False``。
    """
    trimmed = value.strip()
    return not trimmed.startswith(_NON_LOCAL_PREFIXES)


def mark_path_ignored_by_cloud_sync(path: str) -> None:
    """标记路径为云同步忽略。

    在 macOS 上设置 Dropbox 和 Apple File Provider 忽略属性，
    在 Linux 上设置 Dropbox 忽略属性。

    Args:
        path: 要标记的路径。
    """
    system = platform.system()
    if system == "Darwin":
        for attr in ["com.dropbox.ignored", "com.apple.fileprovider.ignore#P"]:
            try:
                subprocess.run(
                    ["xattr", "-w", attr, "1", path],
                    capture_output=True,
                    timeout=5,
                )
            except Exception:
                pass
    elif system == "Linux":
        try:
            subprocess.run(
                ["setfattr", "-n", "user.com.dropbox.ignored", "-v", "1", path],
                capture_output=True,
                timeout=5,
            )
        except Exception:
            pass


def _expand_tilde_with_home(path: str, home_dir: str) -> str:
    """使用指定的 home 目录展开 ``~``。"""
    if path == "~":
        return home_dir
    if path.startswith("~/"):
        return os.path.join(home_dir, path[2:])
    return path
