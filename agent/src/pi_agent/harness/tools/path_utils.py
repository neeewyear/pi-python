"""路径解析工具。

提供 Unicode 空格规范化与文件读取路径解析。
"""

from __future__ import annotations

import re

from ...cancellation import CancellationToken
from ...result import get_or_throw
from ..types import ExecutionEnv

_UNICODE_SPACES = re.compile(r"[\u00A0\u2000-\u200A\u202F\u205F\u3000]")
_NARROW_NO_BREAK_SPACE = "\u202F"


def _normalize_tool_path(path: str) -> str:
    """规范化工具路径：Unicode 空格 → 常规空格，去除 @ 前缀。"""
    normalized = _UNICODE_SPACES.sub(" ", path)
    return normalized[1:] if normalized.startswith("@") else normalized


async def resolve_tool_path(
    env: ExecutionEnv,
    path: str,
    signal: CancellationToken | None = None,
) -> str:
    """解析工具路径为绝对路径。"""
    return get_or_throw(await env.absolute_path(_normalize_tool_path(path), signal))


async def resolve_read_tool_path(
    env: ExecutionEnv,
    path: str,
    signal: CancellationToken | None = None,
) -> str:
    """解析读取工具路径，尝试多种变体。"""
    resolved = await resolve_tool_path(env, path, signal)
    variants = [
        resolved,
        re.sub(r" (AM|PM)\.", rf"{_NARROW_NO_BREAK_SPACE}\1.", resolved, flags=re.IGNORECASE),
        _normalize_nfd(resolved),
        resolved.replace("'", "\u2019"),
        _normalize_nfd(resolved).replace("'", "\u2019"),
    ]

    seen: set[str] = set()
    for variant in variants:
        if variant in seen:
            continue
        seen.add(variant)
        exists_result = await env.exists(variant, signal)
        if exists_result.is_ok() and exists_result.value:
            return variant
    return resolved


def _normalize_nfd(text: str) -> str:
    """NFD 归一化。"""
    import unicodedata
    return unicodedata.normalize("NFD", text)