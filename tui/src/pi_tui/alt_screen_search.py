"""
备用屏幕搜索 — mirrors packages/tui/src/alt-screen-search.ts

提供：
- find_alt_screen_search_matches(): 在剥离终端序列后的文档语料中查找查询匹配
- get_alt_screen_search_match_key(): 匹配的唯一键（用于跨渲染定位当前选中项）
- AltScreenSearchComponent: 顶部标签栏 + 输入框的搜索组件（Component + Focusable）
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from .components.input import Input
from .utils import (
    _grapheme_width,
    _segment_graphemes,
    extract_ansi_code,
    truncate_to_width,
    visible_width,
)


# ─────────────────────────────────────────────────────────────────────────────
# 数据类型
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SearchSourceSpan:
    """语料中一段文本对应的原始屏幕行与列区间。"""

    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchSegment:
    """匹配在某一屏幕行上的列区间。"""

    row: int
    start_col: int
    end_col: int


@dataclass
class AltScreenSearchMatch:
    """一次完整匹配，可能跨多个屏幕行。"""

    segments: list[AltScreenSearchSegment]


# ─────────────────────────────────────────────────────────────────────────────
# 语料构建
# ─────────────────────────────────────────────────────────────────────────────


def _strip_terminal_sequences(text: str) -> str:
    """移除 ANSI/OSC/APC 控制序列，保留可见文本。"""
    if "\x1b" not in text:
        return text
    result: list[str] = []
    i = 0
    while i < len(text):
        ansi = extract_ansi_code(text, i)
        if ansi is not None:
            i += ansi.length
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


def _build_search_corpus(
    lines: list[str],
) -> tuple[str, list[SearchSourceSpan | None]]:
    """构建搜索语料。

    逐行剥离终端序列并按字素遍历；空白字素折叠为单个分隔空格
    （跨行时同样折叠），非空白字素映射回其屏幕行/列区间。
    返回 (语料文本, 每个码元的来源区间或 None)。
    """
    text_parts: list[str] = []
    source: list[SearchSourceSpan | None] = []
    pending_separator = False

    for row, line in enumerate(lines):
        clean = _strip_terminal_sequences(line)
        column = 0
        for grapheme in _segment_graphemes(clean):
            width = _grapheme_width(grapheme)
            if re.fullmatch(r"\s+", grapheme) is not None:
                if text_parts:
                    pending_separator = True
                column += width
                continue
            if pending_separator:
                source.append(None)
                text_parts.append(" ")
                pending_separator = False
            span = SearchSourceSpan(row=row, start_col=column, end_col=column + width)
            text_parts.append(grapheme)
            # 每个码元字符映射到同一个区间（对齐 TS 逐码元 push）
            source.extend([span] * len(grapheme))
            column += width
        if text_parts:
            pending_separator = True

    return "".join(text_parts), source


# ─────────────────────────────────────────────────────────────────────────────
# 匹配查找
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_query(query: str) -> str:
    """将查询中的连续空白折叠为单个空格并去除首尾空白。"""
    return re.sub(r"\s+", " ", query).strip()


def find_alt_screen_search_matches(
    lines: list[str], query: str
) -> list[AltScreenSearchMatch]:
    """在屏幕行中查找查询的匹配（不区分大小写，ANSI 序列不计入）。

    Args:
        lines: 渲染出的屏幕行（可能含 ANSI 序列）。
        query: 原始查询文本。

    Returns:
        匹配列表；空查询/纯空白查询返回空列表。
    """
    normalized = _normalize_query(query)
    if not normalized:
        return []

    corpus_text, source = _build_search_corpus(lines)
    expression = re.compile(re.escape(normalized), re.IGNORECASE)
    matches: list[AltScreenSearchMatch] = []

    for match in expression.finditer(corpus_text):
        start = match.start()
        end = match.end()
        segments: list[AltScreenSearchSegment] = []
        for index in range(start, end):
            span = source[index] if index < len(source) else None
            if span is None:
                continue
            previous = segments[-1] if segments else None
            if (
                previous is not None
                and previous.row == span.row
                and span.start_col <= previous.end_col
            ):
                previous.end_col = max(previous.end_col, span.end_col)
            else:
                segments.append(
                    AltScreenSearchSegment(
                        row=span.row, start_col=span.start_col, end_col=span.end_col
                    )
                )
        if segments:
            matches.append(AltScreenSearchMatch(segments=segments))

    return matches


def get_alt_screen_search_match_key(match: AltScreenSearchMatch) -> str:
    """返回匹配的唯一键：``首行:首列:末行:末列``，无段时返回空字符串。"""
    first = match.segments[0] if match.segments else None
    last = match.segments[-1] if match.segments else None
    if first is not None and last is not None:
        return f"{first.row}:{first.start_col}:{last.row}:{last.end_col}"
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# 搜索组件
# ─────────────────────────────────────────────────────────────────────────────


class AltScreenSearchComponent:
    """
    备用屏幕搜索组件：顶部标签栏（含匹配状态）+ 输入框。
    对标 TS AltScreenSearchComponent。
    """

    def __init__(self, on_query_change: Callable[[str], None]) -> None:
        self._input = Input()
        self._on_query_change = on_query_change
        self._result_count = 0
        self._result_index = -1
        self._focused = False

    @property
    def focused(self) -> bool:
        return self._focused

    @focused.setter
    def focused(self, value: bool) -> None:
        self._focused = value
        self._input.focused = value

    def set_result(self, index: int, count: int) -> None:
        """由渲染器设置当前匹配索引与总数，用于状态栏显示。"""
        self._result_index = index
        self._result_count = count

    def handle_input(self, data: str) -> None:
        previous = self._input.get_value()
        self._input.handle_input(data)
        query = self._input.get_value()
        if query != previous:
            self._on_query_change(query)

    def invalidate(self) -> None:
        self._input.invalidate()

    def render(self, width: int) -> list[str]:
        safe_width = max(1, width)
        label = " Find transcript"
        query = self._input.get_value()
        if not query:
            status = ""
        elif self._result_count == 0:
            status = "No matches "
        else:
            status = f"{self._result_index + 1}/{self._result_count} "
        label_width = visible_width(label)
        status_width = visible_width(status)
        gap = " " * max(1, safe_width - label_width - status_width)
        title = truncate_to_width(f"{label}{gap}{status}", safe_width, "")
        padding = " " * max(0, safe_width - visible_width(title))
        return [f"\x1b[7m{title}{padding}\x1b[27m", *self._input.render(safe_width)]
