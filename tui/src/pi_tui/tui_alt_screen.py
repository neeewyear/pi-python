"""
备用屏幕渲染器 — mirrors packages/tui/src/tui-alt-screen.ts

TuiAltScreen: 渲染进终端备用屏幕缓冲区的全屏 TUI，带应用自有视口滚动。
对齐 TS 结构：继承 TUI（对标 TuiBase），内部组合隐式 ScrollView + 布局帧
（简化版 render_layout_frame，视口 = 整个终端）、AltScreenFlashContainer，
并提供鼠标选区/复制、滚动条、搜索（alt_screen_search）与键盘滚动绑定。

与 TS 的差异（均已在相应位置注释）：
- Kitty 图片占位缓存/裁剪未移植（保持 is_image_line 的逐行处理）。
- 布局系统简化为单一滚动视口（layout_root 需为 ScrollView 或普通组件）。
"""

from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import NamedTuple

from .alt_screen_search import (
    AltScreenSearchComponent,
    AltScreenSearchMatch,
    find_alt_screen_search_matches,
    get_alt_screen_search_match_key,
)
from .components.alt_screen_flash import AltScreenFlashContainer
from .keys import KeyId, is_key_release, matches_key
from .terminal import Terminal
from .terminal_image import delete_all_kitty_images, get_capabilities, is_image_line
from .tui import (
    CURSOR_MARKER,
    TUI,
    Component,
    OverlayHandle,
    OverlayOptions,
)
from .utils import (
    _grapheme_width,
    _segment_graphemes,
    extract_ansi_code,
    is_punctuation_char,
    is_whitespace_char,
    slice_by_column,
    visible_width,
)

# ─────────────────────────────────────────────────────────────────────────────
# 终端序列常量 — 对齐 tui-alt-screen.ts
# ─────────────────────────────────────────────────────────────────────────────

ENTER_ALT_SCREEN = "\x1b[?1049h"
EXIT_ALT_SCREEN = "\x1b[?1049l"
DISABLE_AUTOWRAP = "\x1b[?7l"
ENABLE_AUTOWRAP = "\x1b[?7h"
ENABLE_BUTTON_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1004h\x1b[?1006h"
ENABLE_ALL_MOTION_MOUSE = "\x1b[?1000h\x1b[?1002h\x1b[?1003h\x1b[?1004h\x1b[?1006h"
DISABLE_MOUSE = "\x1b[?1006l\x1b[?1004l\x1b[?1003l\x1b[?1002l\x1b[?1000l"
FOCUS_IN = "\x1b[I"
FOCUS_OUT = "\x1b[O"
BEGIN_SYNCHRONIZED_OUTPUT = "\x1b[?2026h"
END_SYNCHRONIZED_OUTPUT = "\x1b[?2026l"
PAGE_SCROLL_OVERLAP = 4
DOUBLE_CLICK_INTERVAL_S = 0.5
SELECTION_AUTO_SCROLL_INTERVAL_S = 0.05

_OSC133_ZONE_PREFIX = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")
_OSC133_PROMPT_START = re.compile(r"^\x1b\]133;A(?:\x07|\x1b\\)")
_SGR_MOUSE_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)([Mm])$")
_WHEEL_SGR_RE = re.compile(r"^\x1b\[<(\d+);(\d+);(\d+)[Mm]$")


# ─────────────────────────────────────────────────────────────────────────────
# 滚动视图（简化移植 components/scroll-view.ts）
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class ScrollViewOptions:
    axis: str | None = None
    follow: str | None = None  # "none" | "end"
    primary: bool = False
    overscroll: str | None = None  # "chain" | "contain"
    scrollbar: str | None = None  # "hidden" | "auto" | "always"
    scrollbar_style: Callable[[str], str] | None = None
    scrollbar_hide_delay_ms: int | None = None


class ScrollView:
    """
    垂直滚动视图：包装单个子组件并管理滚动位置 / 滚动条。
    对标 TS ScrollView（仅支持 vertical 轴；children 管理抛错）。
    """

    def __init__(
        self, component: Component, options: ScrollViewOptions | None = None
    ) -> None:
        opts = options or ScrollViewOptions()
        if opts.axis is not None and opts.axis != "vertical":
            raise ValueError(f"Unsupported ScrollView axis: {opts.axis}")
        self._child = component
        self._follow_end = (opts.follow or "none") == "end"
        self._following_end = self._follow_end
        self.primary = opts.primary
        self.overscroll = opts.overscroll or "chain"
        self._scrollbar = opts.scrollbar or "hidden"
        self._scrollbar_style: Callable[[str], str] = opts.scrollbar_style or (
            lambda text: f"\x1b[100m{text}\x1b[49m"
        )
        self._scrollbar_hide_delay_ms = max(
            0, math.floor(opts.scrollbar_hide_delay_ms or 1000)
        )
        self._current_scroll_top = 0
        self._content_height = 0
        self._current_viewport_height = 0
        self._follow_suppressed_at_end = False
        self._request_render_callback: Callable[[], None] | None = None
        self._transient_scrollbar_visible = False
        self._scrollbar_active = False
        self._scrollbar_hide_timer: threading.Timer | None = None

    @property
    def scroll_top(self) -> int:
        return self._current_scroll_top

    @property
    def is_following_end(self) -> bool:
        return self._following_end

    @property
    def viewport_height(self) -> int:
        return self._current_viewport_height

    @property
    def scrollbar(self) -> str:
        return self._scrollbar

    @property
    def scrollbar_style(self) -> Callable[[str], str]:
        return self._scrollbar_style

    @property
    def is_scrollbar_visible(self) -> bool:
        if self._scrollbar == "always":
            return self._current_viewport_height > 0
        return (
            self._scrollbar == "auto"
            and self._content_height > self._current_viewport_height
            and self._transient_scrollbar_visible
        )

    def set_scrollbar(self, scrollbar: str) -> None:
        if scrollbar == self._scrollbar:
            return
        self._scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self._scrollbar_active:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def get_content_width(self, width: int) -> int:
        """滚动条常驻时内容宽度让出一列。"""
        return width - 1 if self._scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        if (
            self._scrollbar != "auto"
            or self._content_height <= self._current_viewport_height
        ):
            return
        self._transient_scrollbar_visible = True
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self._scrollbar_active:
            return
        timer = threading.Timer(
            self._scrollbar_hide_delay_ms / 1000.0, self._hide_transient_scrollbar
        )
        timer.daemon = True
        self._scrollbar_hide_timer = timer
        timer.start()

    def _hide_transient_scrollbar(self) -> None:
        self._transient_scrollbar_visible = False
        if self._scrollbar_hide_timer is None:
            return
        self._scrollbar_hide_timer = None
        if self._request_render_callback is not None:
            self._request_render_callback()

    def set_scrollbar_active(self, active: bool) -> None:
        if active == self._scrollbar_active:
            return
        self._scrollbar_active = active
        self._mark_scrollbar_activity()

    def scroll_to(self, scroll_top: int, disable_follow: bool = False) -> None:
        """滚动到指定行；disable_follow=True 时即使到达内容末尾也不恢复跟随。"""
        requested = (
            scroll_top if math.isfinite(scroll_top) else self._current_scroll_top
        )
        requested = math.trunc(requested)
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        next_top = max(0, min(max_scroll_top, requested))
        next_follow_suppressed = disable_follow and next_top == max_scroll_top
        next_following_end = (
            not next_follow_suppressed
            and self._follow_end
            and next_top == max_scroll_top
        )
        if (
            next_top == self._current_scroll_top
            and next_following_end == self._following_end
            and next_follow_suppressed == self._follow_suppressed_at_end
        ):
            return
        moved = next_top != self._current_scroll_top
        self._current_scroll_top = next_top
        self._following_end = next_following_end
        self._follow_suppressed_at_end = next_follow_suppressed
        if moved:
            self._mark_scrollbar_activity()
        if self._request_render_callback is not None:
            self._request_render_callback()

    def scroll_by(self, lines: int) -> int:
        """相对滚动；返回未能消耗的剩余行数（用于 overscroll 链）。"""
        requested = lines if math.isfinite(lines) else 0
        requested = math.trunc(requested)
        if requested == 0:
            return 0
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        start = max_scroll_top if self._following_end else self._current_scroll_top
        next_top = max(0, min(max_scroll_top, start + requested))
        moved = next_top - start
        was_following_end = self._following_end
        self._current_scroll_top = next_top
        self._following_end = self._follow_end and next_top == max_scroll_top
        self._follow_suppressed_at_end = False
        if moved != 0:
            self._mark_scrollbar_activity()
        if moved != 0 or self._following_end != was_following_end:
            if self._request_render_callback is not None:
                self._request_render_callback()
        return requested - moved

    def scroll_to_start(self) -> None:
        changed = self._current_scroll_top != 0 or self._following_end != (
            self._follow_end and self._content_height <= self._current_viewport_height
        )
        self._current_scroll_top = 0
        self._following_end = (
            self._follow_end and self._content_height <= self._current_viewport_height
        )
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def scroll_to_end(self) -> None:
        next_top = max(0, self._content_height - self._current_viewport_height)
        changed = (
            self._current_scroll_top != next_top
            or self._following_end != self._follow_end
        )
        self._current_scroll_top = next_top
        self._following_end = self._follow_end
        self._follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self._request_render_callback is not None:
                self._request_render_callback()

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        """由布局帧在渲染时调用，更新内容/视口尺寸并夹取滚动位置。"""
        self._content_height = max(0, math.floor(content_height))
        self._current_viewport_height = max(0, math.floor(viewport_height))
        self._request_render_callback = request_render
        max_scroll_top = max(0, self._content_height - self._current_viewport_height)
        if self._following_end:
            self._current_scroll_top = max_scroll_top
        else:
            self._current_scroll_top = max(
                0, min(self._current_scroll_top, max_scroll_top)
            )
        if self._current_scroll_top < max_scroll_top:
            self._follow_suppressed_at_end = False
        if (
            self._follow_end
            and self._current_scroll_top == max_scroll_top
            and not self._follow_suppressed_at_end
        ):
            self._following_end = True
        if self._content_height <= self._current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, _component: Component) -> None:
        raise ValueError("ScrollView has exactly one child")

    def remove_child(self, _component: Component) -> None:
        raise ValueError("ScrollView child cannot be removed")

    def clear(self) -> None:
        raise ValueError("ScrollView child cannot be cleared")

    def handle_input(self, _data: str) -> None:
        pass

    def invalidate(self) -> None:
        self._child.invalidate()

    def render(self, width: int) -> list[str]:
        content_width = self.get_content_width(width)
        lines = self._child.render(content_width)
        if content_width == width:
            return lines
        return [f"{line} " for line in lines]


# ─────────────────────────────────────────────────────────────────────────────
# 布局数据结构与简化的布局帧渲染（对标 layout.ts，仅单一滚动视口）
# ─────────────────────────────────────────────────────────────────────────────


class LayoutRect(NamedTuple):
    x: int
    y: int
    width: int
    height: int


@dataclass
class LayoutBox:
    component: Component
    rect: LayoutRect
    clip: LayoutRect
    children: list[LayoutBox]
    lines: list[str] | None = None
    line_offset: int = 0
    scroll_view: ScrollView | None = None
    scroll_content_lines: list[str] | None = None
    layer: int = 0
    parent: LayoutBox | None = None


@dataclass
class LayoutFrame:
    root: LayoutBox
    width: int
    height: int
    lines: list[str]
    primary_scroll_view: ScrollView | None = None


@dataclass
class ScrollbarGeometry:
    column: int
    track_top: int
    track_height: int
    thumb_top: int
    thumb_height: int
    max_scroll_top: int


def render_layout_frame(
    root: Component,
    width: int,
    height: int,
    request_render: Callable[[], None],
) -> LayoutFrame:
    """渲染布局帧：将根组件渲染为内容行并裁剪出视口窗口。

    当根组件为 ScrollView 时应用其滚动位置并绘制滚动条；
    普通组件则直接取前 height 行。
    """
    safe_width = max(1, math.floor(width))
    safe_height = max(1, math.floor(height))
    scroll_view = root if isinstance(root, ScrollView) else None
    content_lines = root.render(safe_width)
    if scroll_view is not None:
        scroll_view.update_layout(len(content_lines), safe_height, request_render)
    viewport_top = scroll_view.scroll_top if scroll_view is not None else 0
    lines = content_lines[viewport_top : viewport_top + safe_height]
    while len(lines) < safe_height:
        lines.append("")
    box = LayoutBox(
        component=root,
        rect=LayoutRect(0, 0, safe_width, safe_height),
        clip=LayoutRect(0, 0, safe_width, safe_height),
        children=[],
        lines=content_lines,
        line_offset=0,
        scroll_view=scroll_view,
        scroll_content_lines=content_lines,
        layer=0,
    )
    frame = LayoutFrame(
        root=box,
        width=safe_width,
        height=safe_height,
        lines=lines,
        primary_scroll_view=scroll_view,
    )
    if scroll_view is not None:
        _paint_scrollbar(box, frame.lines, safe_width)
    return frame


def get_scroll_view_box(
    frame: LayoutFrame, scroll_view: ScrollView
) -> LayoutBox | None:
    """返回包含指定 ScrollView 的布局盒；单视口布局下即根盒。"""
    if frame.primary_scroll_view is scroll_view:
        return frame.root
    return None


def get_scroll_views_at(frame: LayoutFrame | None, x: int, y: int) -> list[ScrollView]:
    """返回 (x, y) 位置命中的滚动视图（自内向外）。"""
    if frame is None or frame.primary_scroll_view is None:
        return []
    rect = frame.root.rect
    if rect.x <= x < rect.x + rect.width and rect.y <= y < rect.y + rect.height:
        return [frame.primary_scroll_view]
    return []


def get_scrollbar_geometry(box: LayoutBox) -> ScrollbarGeometry | None:
    """计算滚动条几何（列号/轨道/滑块），不可见时返回 None。"""
    scroll_view = box.scroll_view
    if (
        scroll_view is None
        or not scroll_view.is_scrollbar_visible
        or box.rect.width <= 0
        or box.rect.height <= 0
    ):
        return None
    content_height = (
        len(box.scroll_content_lines) if box.scroll_content_lines is not None else 0
    )
    track_height = box.rect.height
    min_thumb_height = min(2, track_height)
    thumb_height = max(
        min_thumb_height,
        min(
            track_height,
            round((track_height * track_height) / max(1, content_height)),
        ),
    )
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_offset = (
        0
        if max_scroll_top == 0
        else round((scroll_view.scroll_top / max_scroll_top) * max_thumb_top)
    )
    column = box.rect.x + box.rect.width - 1
    if column < box.clip.x or column >= box.clip.x + box.clip.width:
        return None
    return ScrollbarGeometry(
        column=column,
        track_top=box.rect.y,
        track_height=track_height,
        thumb_top=box.rect.y + thumb_offset,
        thumb_height=thumb_height,
        max_scroll_top=max_scroll_top,
    )


def _style_scrollbar_cell(
    line: str,
    column: int,
    total_width: int,
    style: Callable[[str], str],
) -> str:
    """将行上滚动条列的字素替换为样式化版本。"""
    if is_image_line(line):
        return line
    grapheme_range = _get_grapheme_cell_range(line, column)
    start = grapheme_range[0] if grapheme_range is not None else column
    end = grapheme_range[1] if grapheme_range is not None else column + 1
    before = slice_by_column(line, 0, start, True)
    target = slice_by_column(line, start, end - start, True)
    after = slice_by_column(line, end, max(0, total_width - end), True)

    target_prefix = ""
    target_index = 0
    while target_index < len(target):
        ansi = extract_ansi_code(target, target_index)
        if ansi is None:
            break
        target_prefix += ansi.code
        target_index += ansi.length
    target_text = target[target_index:] or " " * (end - start)
    before_padding = " " * max(0, start - visible_width(before))
    return f"{before}{before_padding}{target_prefix}{style(target_text)}{after}"


def _paint_scrollbar(box: LayoutBox, screen: list[str], total_width: int) -> None:
    """将滚动条滑块绘制到屏幕行的最后一列。"""
    geometry = get_scrollbar_geometry(box)
    if geometry is None or box.scroll_view is None:
        return
    for offset in range(geometry.thumb_height):
        row = geometry.thumb_top + offset
        if (
            row < box.clip.y
            or row >= box.clip.y + box.clip.height
            or row < 0
            or row >= len(screen)
        ):
            continue
        screen[row] = _style_scrollbar_cell(
            screen[row], geometry.column, total_width, box.scroll_view.scrollbar_style
        )


def _strip_osc133_zone(line: str) -> str:
    return _OSC133_ZONE_PREFIX.sub("", line)


# ─────────────────────────────────────────────────────────────────────────────
# 文本工具（utils.ts 中未被 Python utils 暴露的等价实现）
# ─────────────────────────────────────────────────────────────────────────────


def _strip_terminal_sequences(text: str) -> str:
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


def _get_grapheme_cell_range(line: str, column: int) -> tuple[int, int] | None:
    """返回可见列 column 所在字素的终端单元区间 [start, end)。"""
    current_col = 0
    i = 0
    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi is not None:
            i += ansi.length
            continue
        text_end = i
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1
        for segment in _segment_graphemes(line[i:text_end]):
            width = _grapheme_width(segment)
            if width > 0 and column >= current_col and column < current_col + width:
                return (current_col, current_col + width)
            current_col += width
        i = text_end
    return None


def _get_osc8_link_at_column(line: str, column: int) -> str | None:
    """返回覆盖可见列 column 的 OSC 8 超链接 URL（如有）。"""
    active_url: str | None = None
    current_col = 0
    i = 0
    while i < len(line):
        ansi = extract_ansi_code(line, i)
        if ansi is not None:
            link = re.match(r"^\x1b\]8;[^;]*;([^\x07\x1b]*)(?:\x07|\x1b\\)$", ansi.code)
            if link is not None:
                url = link.group(1)
                active_url = url if url else None
            i += ansi.length
            continue
        text_end = i
        while text_end < len(line) and extract_ansi_code(line, text_end) is None:
            text_end += 1
        for segment in _segment_graphemes(line[i:text_end]):
            width = 3 if segment == "\t" else _grapheme_width(segment)
            if column >= current_col and column < current_col + width:
                return active_url
            current_col += width
        i = text_end
    return None


def _word_segments(line: str) -> list[tuple[int, int]]:
    """将一行按字素切分为 (空白 | 标点 | 单词) 连续段，返回各段的列区间。"""
    result: list[tuple[int, int]] = []
    current_type: str | None = None
    start_col = 0
    col = 0
    for grapheme in _segment_graphemes(line):
        width = _grapheme_width(grapheme)
        if is_whitespace_char(grapheme):
            kind = "ws"
        elif is_punctuation_char(grapheme):
            kind = "punct"
        else:
            kind = "word"
        if current_type is None:
            current_type = kind
            start_col = col
        elif kind != current_type:
            result.append((start_col, col))
            current_type = kind
            start_col = col
        col += width
    if current_type is not None:
        result.append((start_col, col))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# 事件/状态数据结构 — 对齐 tui-alt-screen.ts 内部接口
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class SelectionPoint:
    row: int
    col: int
    scroll_view: ScrollView | None = None
    boundary: bool = False


@dataclass
class SelectionRange:
    start: SelectionPoint
    end: SelectionPoint


@dataclass
class ClickTarget:
    timestamp: float
    count: int
    row: int
    scroll_view: ScrollView | None
    word_start: int
    word_end: int


@dataclass
class SgrMouseEvent:
    button: int
    x: int
    y: int
    release: bool


@dataclass
class WheelEvent:
    direction: int  # -1（上） | 1（下）
    x: int
    y: int


@dataclass
class ScrollbarDrag:
    scroll_view: ScrollView
    grab_offset: int


@dataclass
class ScrollbarTarget:
    scroll_view: ScrollView
    geometry: ScrollbarGeometry


@dataclass
class ActiveSearch:
    component: AltScreenSearchComponent
    overlay: OverlayHandle | None
    query: str
    matches: list[AltScreenSearchMatch]
    selected_index: int
    selected_key: str | None
    anchor_row: int
    selection_mode: str  # "query" | "retain" | "next" | "previous"


@dataclass
class SearchHighlightRange:
    start_col: int
    end_col: int
    current: bool


@dataclass
class TuiAltScreenOptions:
    """备用屏幕选项（对标 TuiAltScreenOptions）。"""

    wheel_scroll_lines: int | None = None
    mouse: bool | None = None
    search_match_style: Callable[[str], str] | None = None
    search_current_match_style: Callable[[str], str] | None = None
    open_url: Callable[[str], None] | None = None
    on_right_click_paste: Callable[[], None] | None = None


@dataclass
class TuiStopOptions:
    """停止选项（对标 TuiStopOptions）。"""

    preserve_screen: bool = False


# ─────────────────────────────────────────────────────────────────────────────
# 备用屏幕按键绑定 — 对齐 keybindings.ts 中 tui.altScreen.* 的默认键
# ─────────────────────────────────────────────────────────────────────────────

ALT_SCREEN_ACTION_KEYS: dict[str, list[KeyId]] = {
    "tui.altScreen.pageUp": ["pageUp"],
    "tui.altScreen.pageDown": ["pageDown"],
    "tui.altScreen.halfPageUp": [],
    "tui.altScreen.halfPageDown": [],
    "tui.altScreen.lineUp": [],
    "tui.altScreen.lineDown": [],
    "tui.altScreen.previousPrompt": ["ctrl+shift+up"],
    "tui.altScreen.nextPrompt": ["ctrl+shift+down"],
    "tui.altScreen.search": ["ctrl+shift+f"],
    "tui.altScreen.searchNext": ["enter", "ctrl+g"],
    "tui.altScreen.searchPrevious": ["shift+enter", "ctrl+shift+g"],
    "tui.altScreen.searchClose": ["escape"],
    "tui.altScreen.top": ["home"],
    "tui.altScreen.bottom": ["end"],
}


def matches_alt_screen_action(data: str, action: str) -> bool:
    """判断输入是否匹配某个备用屏幕按键动作。"""
    keys = ALT_SCREEN_ACTION_KEYS.get(action)
    if not keys:
        return False
    return any(matches_key(data, key) for key in keys)


# ─────────────────────────────────────────────────────────────────────────────
# 隐式文档包装
# ─────────────────────────────────────────────────────────────────────────────


class _ImplicitDocument:
    """将 TuiAltScreen 的 children 暴露为可滚动文档（对齐 TS implicitDocument）。"""

    def __init__(self, owner: TuiAltScreen) -> None:
        self._owner = owner

    def render(self, width: int) -> list[str]:
        return TUI.render(self._owner, width)

    def invalidate(self) -> None:
        for child in self._owner.children:
            if hasattr(child, "invalidate"):
                child.invalidate()  # type: ignore[attr-defined]

    def handle_input(self, _data: str) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TuiAltScreen
# ─────────────────────────────────────────────────────────────────────────────


class TuiAltScreen(TUI):
    """
    备用屏幕全屏 TUI 渲染器：可滚动视口 + 应用自有鼠标选区/复制 + 搜索。
    对标 TS TuiAltScreen（继承 TUI，对应 TS TuiBase + Container）。
    """

    mode = "fullscreen"

    def __init__(
        self,
        terminal: Terminal,
        show_hardware_cursor: bool | None = None,
        options: TuiAltScreenOptions | None = None,
    ) -> None:
        super().__init__(terminal, show_hardware_cursor)
        opts = options or TuiAltScreenOptions()
        self._implicit_document = _ImplicitDocument(self)
        self._implicit_scroll_view = ScrollView(
            self._implicit_document,
            ScrollViewOptions(follow="end", primary=True),
        )
        self._flashes = AltScreenFlashContainer(lambda: self.request_render())
        self._wheel_scroll_lines = max(
            1,
            math.floor(
                opts.wheel_scroll_lines if opts.wheel_scroll_lines is not None else 1
            ),
        )
        self._mouse_enabled = opts.mouse if opts.mouse is not None else True
        self._search_match_style: Callable[[str], str] = opts.search_match_style or (
            lambda text: f"\x1b[4m{text}\x1b[24m"
        )
        self._search_current_match_style: Callable[[str], str] = (
            opts.search_current_match_style
            or (lambda text: f"\x1b[1;7m{text}\x1b[22;27m")
        )
        self._open_url = opts.open_url
        self._on_right_click_paste = opts.on_right_click_paste
        # AltScreen 渲染状态
        self._previous_screen: list[str] = []
        self._last_document: list[str] = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._layout_root: Component | None = None
        self._current_layout: LayoutFrame | None = None
        self._alt_screen_active = False
        self._image_protocol = get_capabilities().images
        # 选区 / 鼠标状态
        self._selection_anchor: SelectionPoint | None = None
        self._selection_focus: SelectionPoint | None = None
        self._selection_granularity = "character"
        self._selection_initial_range: SelectionRange | None = None
        self._last_click: ClickTarget | None = None
        self._selection_drag_pointer: tuple[int, int] | None = None
        self._selection_auto_scroll_direction = 0
        self._selection_auto_scroll_timer: threading.Timer | None = None
        self._selection_press_active = False
        self._scrollbar_drag: ScrollbarDrag | None = None
        self._scrollbar_hover: ScrollView | None = None
        self._active_search: ActiveSearch | None = None
        self._pressed_url: str | None = None
        self._selection_dragged = False
        # 视口输入监听（对齐 TS constructor 中 addInputListener）
        self.add_input_listener(lambda data: self._handle_viewport_input(data))

    # ── 公共属性 / 视口 API ───────────────────────────────────────────────

    @property
    def viewport_top(self) -> int:
        return self._get_primary_scroll_view().scroll_top

    @property
    def is_following_output(self) -> bool:
        return self._get_primary_scroll_view().is_following_end

    def set_layout_root(self, component: Component | None) -> None:
        """设置显式布局根组件（ScrollView 或普通组件）。"""
        if self._layout_root is component:
            return
        self._layout_root = component
        self._current_layout = None
        self.request_render()

    def render(self, width: int) -> list[str]:
        if self._layout_root is not None:
            return self._layout_root.render(width)
        return super().render(width)

    def invalidate(self) -> None:
        if self._layout_root is not None:
            self._layout_root.invalidate()
        else:
            super().invalidate()
        for overlay in self._overlay_stack:
            if hasattr(overlay.component, "invalidate"):
                overlay.component.invalidate()  # type: ignore[attr-defined]

    def request_render(self, force: bool = False) -> None:
        if force:
            self._previous_screen = []
            self._previous_screen_width = 0
            self._previous_screen_height = 0
            self._current_layout = None
        super().request_render(force)

    def _get_primary_scroll_view(self) -> ScrollView:
        if (
            self._current_layout is not None
            and self._current_layout.primary_scroll_view is not None
        ):
            return self._current_layout.primary_scroll_view
        return self._implicit_scroll_view

    def scroll_by(self, lines: int) -> None:
        """滚动主滚动视图指定的逻辑行数。"""
        self._get_primary_scroll_view().scroll_by(lines)
        self.request_render()

    def scroll_to_top(self) -> None:
        """滚动到主滚动视图顶部。"""
        self._get_primary_scroll_view().scroll_to_start()
        self.request_render()

    def scroll_to_bottom(self) -> None:
        """滚动到主滚动视图底部（恢复跟随输出）。"""
        self._get_primary_scroll_view().scroll_to_end()
        self.request_render()

    def flash(self, message: str, duration_ms: int | None = None) -> None:
        """在备用屏幕顶部显示一条瞬时消息。"""
        self._flashes.flash(message, duration_ms)

    # ── 生命周期（对齐 TS beforeTerminalStart / beforeTerminalStop / afterTerminalStop）──

    def start(self) -> None:
        self._stopped = False
        self._before_terminal_start()
        super().start()

    def stop(self, options: TuiStopOptions | None = None) -> None:
        """停止备用屏幕渲染并恢复主屏幕（可选保留屏幕内容）。"""
        self._stopped = True
        # TS beforeTerminalStop
        self._close_search()
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        if not self._alt_screen_active:
            self.terminal.show_cursor()
            self.terminal.stop()
            return
        self.terminal.write(
            f"{BEGIN_SYNCHRONIZED_OUTPUT}{self._delete_kitty_images()}"
            f"{DISABLE_MOUSE if self._mouse_enabled else ''}{ENABLE_AUTOWRAP}"
            f"{END_SYNCHRONIZED_OUTPUT}"
        )
        self.terminal.show_cursor()
        self.terminal.stop()
        # TS afterTerminalStop
        self._after_terminal_stop(options)

    def _before_terminal_start(self) -> None:
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._stop_scrollbar_hover()
        self._stop_scrollbar_drag()
        self._flashes.dispose()
        self._alt_screen_active = True
        self._image_protocol = get_capabilities().images
        self._last_document = []
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        # 对齐 TS resetRenderState()
        self._previous_screen = []
        self._previous_screen_width = 0
        self._previous_screen_height = 0
        self._current_layout = None
        term = os.environ.get("TERM", "").lower()
        # 复用器（tmux/zellij/screen）下仅启用按钮移动跟踪，避免逐指针转发滞后
        if (
            os.environ.get("TMUX") is not None
            or os.environ.get("ZELLIJ") is not None
            or os.environ.get("STY") is not None
            or term.startswith("tmux")
            or term.startswith("screen")
        ):
            mouse_sequence = ENABLE_BUTTON_MOTION_MOUSE
        else:
            mouse_sequence = ENABLE_ALL_MOTION_MOUSE
        self.terminal.write(
            f"{ENTER_ALT_SCREEN}{DISABLE_AUTOWRAP}"
            f"{mouse_sequence if self._mouse_enabled else ''}"
            f"\x1b[2J\x1b[H\x1b[?25l"
        )

    def _delete_kitty_images(self) -> str:
        return delete_all_kitty_images() if self._image_protocol == "kitty" else ""

    def _after_terminal_stop(self, options: TuiStopOptions | None = None) -> None:
        if not self._alt_screen_active:
            return
        if options is not None and options.preserve_screen:
            self.terminal.write(
                f"{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}\x1b[?25h"
                f"{END_SYNCHRONIZED_OUTPUT}"
            )
        else:
            # 将最终文档回显到主屏幕（对齐 TS afterTerminalStop 非 preserve 分支）
            width = max(1, self.terminal.columns)
            document_lines = [_strip_osc133_zone(line) for line in self.render(width)]
            self._last_document = self._apply_line_resets(
                [line.replace(CURSOR_MARKER, "") for line in document_lines]
            )
            self._last_document = [
                line
                if is_image_line(line) or visible_width(line) <= width
                else slice_by_column(line, 0, width, True)
                for line in self._last_document
            ]
            buffer = f"{BEGIN_SYNCHRONIZED_OUTPUT}{EXIT_ALT_SCREEN}{DISABLE_AUTOWRAP}"
            for row, line in enumerate(self._last_document):
                if row > 0:
                    buffer += "\r\n"
                buffer += f"\r\x1b[2K{line}"
            buffer += f"\x1b[0m{ENABLE_AUTOWRAP}\r\n\x1b[?25h{END_SYNCHRONIZED_OUTPUT}"
            self.terminal.write(buffer)
        self._alt_screen_active = False

    # ── 搜索 ──────────────────────────────────────────────────────────────

    def _open_search(self) -> None:
        if self._active_search is not None:
            if self._active_search.overlay is not None:
                self._active_search.overlay.focus()
            return
        component = AltScreenSearchComponent(
            lambda query: self._update_search_query(query)
        )
        search = ActiveSearch(
            component=component,
            overlay=None,
            query="",
            matches=[],
            selected_index=-1,
            selected_key=None,
            anchor_row=self._get_primary_scroll_view().scroll_top,
            selection_mode="query",
        )
        self._active_search = search
        search.overlay = self.show_overlay(
            component,
            OverlayOptions(width="40%", min_width=24, anchor="top-right", margin=1),
        )

    def _close_search(self) -> None:
        search = self._active_search
        if search is None:
            return
        self._active_search = None
        if search.overlay is not None:
            search.overlay.hide()
        self.request_render()

    def _update_search_query(self, query: str) -> None:
        search = self._active_search
        if search is None or query == search.query:
            return
        selected = (
            search.matches[search.selected_index]
            if 0 <= search.selected_index < len(search.matches)
            else None
        )
        if selected is not None and selected.segments:
            search.anchor_row = selected.segments[0].row
        else:
            search.anchor_row = self._get_primary_scroll_view().scroll_top
        search.query = query
        search.selection_mode = "query"
        search.component.set_result(-1, 0)
        self.request_render()

    def _navigate_search(self, direction: int) -> None:
        search = self._active_search
        if search is None or not search.query:
            return
        search.selection_mode = "previous" if direction < 0 else "next"
        self.request_render()

    def _refresh_search(self, layout: LayoutFrame) -> bool:
        """根据当前布局重新计算匹配；返回是否发生了滚动（需要二次布局）。"""
        search = self._active_search
        if search is None:
            return False
        scroll_view = (
            layout.primary_scroll_view
            if layout.primary_scroll_view is not None
            else self._implicit_scroll_view
        )
        box = get_scroll_view_box(layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if lines is None or not search.query.strip():
            search.matches = []
            search.selected_index = -1
            search.selected_key = None
            search.selection_mode = "retain"
            search.component.set_result(-1, 0)
            return False

        should_reveal_selection = search.selection_mode != "retain"
        matches = find_alt_screen_search_matches(lines, search.query)
        exact_index = -1
        if search.selected_key is not None:
            for index, match in enumerate(matches):
                if get_alt_screen_search_match_key(match) == search.selected_key:
                    exact_index = index
                    break
        selected_index = -1
        if matches:
            if search.selection_mode == "query":
                selected_index = next(
                    (
                        index
                        for index, match in enumerate(matches)
                        if (match.segments[0].row if match.segments else 0)
                        >= search.anchor_row
                    ),
                    -1,
                )
                selected_index = max(selected_index, 0)
            elif search.selection_mode == "next":
                base_index = (
                    exact_index
                    if exact_index >= 0
                    else min(search.selected_index, len(matches) - 1)
                )
                selected_index = (
                    0 if base_index < 0 else (base_index + 1) % len(matches)
                )
            elif search.selection_mode == "previous":
                base_index = (
                    exact_index
                    if exact_index >= 0
                    else min(search.selected_index, len(matches) - 1)
                )
                selected_index = (
                    len(matches) - 1
                    if base_index < 0
                    else (base_index - 1 + len(matches)) % len(matches)
                )
            else:
                selected_index = (
                    exact_index
                    if exact_index >= 0
                    else min(max(0, search.selected_index), len(matches) - 1)
                )

        search.matches = matches
        search.selected_index = selected_index
        search.selected_key = (
            get_alt_screen_search_match_key(matches[selected_index])
            if selected_index >= 0
            else None
        )
        search.selection_mode = "retain"
        search.component.set_result(selected_index, len(matches))
        if not should_reveal_selection:
            return False

        selected = (
            matches[selected_index] if 0 <= selected_index < len(matches) else None
        )
        first_segment = selected.segments[0] if selected and selected.segments else None
        last_segment = selected.segments[-1] if selected and selected.segments else None
        if (
            box is None
            or first_segment is None
            or last_segment is None
            or scroll_view.viewport_height <= 0
        ):
            return False
        before = scroll_view.scroll_top
        visible_bottom = before + scroll_view.viewport_height - 1
        target = before
        if first_segment.row < before or last_segment.row > visible_bottom:
            target = first_segment.row - scroll_view.viewport_height // 3
        scroll_view.scroll_to(target, disable_follow=True)
        return scroll_view.scroll_top != before

    def _apply_search_text_highlight(self, text: str, current: bool) -> str:
        """给匹配文本应用样式，保留其中的 ANSI 序列（SGR 恢复除外）。"""
        style = (
            self._search_current_match_style if current else self._search_match_style
        )
        result = ""
        plain_start = 0
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if ansi is None:
                index += 1
                continue
            if index > plain_start:
                result += style(text[plain_start:index])
            result += ansi.code
            index += ansi.length
            plain_start = index
        if plain_start < len(text):
            result += style(text[plain_start:])
        return result

    def _apply_search_highlights(
        self, screen: list[str], layout: LayoutFrame
    ) -> list[str]:
        """在视口内为搜索匹配叠加下划线/反显高亮。"""
        search = self._active_search
        if search is None or search.selected_index < 0 or not search.matches:
            return screen
        scroll_view = layout.primary_scroll_view
        if scroll_view is None:
            return screen
        box = get_scroll_view_box(layout, scroll_view)
        if box is None:
            return screen

        ranges_by_row: dict[int, list[SearchHighlightRange]] = {}
        geometry = get_scrollbar_geometry(box)
        scrollbar_column = geometry.column if geometry is not None else None
        min_row = max(0, box.rect.y, box.clip.y)
        max_row = min(
            len(screen), box.rect.y + box.rect.height, box.clip.y + box.clip.height
        )
        min_column = max(0, box.rect.x, box.clip.x)
        max_column = min(
            self.terminal.columns,
            box.rect.x + box.rect.width,
            box.clip.x + box.clip.width,
        )
        if scrollbar_column is not None:
            max_column = min(max_column, scrollbar_column)
        for match_index, match in enumerate(search.matches):
            for segment in match.segments:
                row = box.rect.y + segment.row - scroll_view.scroll_top
                if row < min_row or row >= max_row:
                    continue
                start_col = max(min_column, box.rect.x + segment.start_col)
                end_col = min(max_column, box.rect.x + segment.end_col)
                if end_col <= start_col:
                    continue
                ranges = ranges_by_row.get(row, [])
                ranges.append(
                    SearchHighlightRange(
                        start_col=start_col,
                        end_col=end_col,
                        current=match_index == search.selected_index,
                    )
                )
                ranges_by_row[row] = ranges

        result = list(screen)
        for row, ranges in ranges_by_row.items():
            line = result[row] if row < len(result) else ""
            if is_image_line(line):
                continue
            line_width = visible_width(line)
            for rng in sorted(ranges, key=lambda r: -r.start_col):
                start_col = min(rng.start_col, line_width)
                end_col = min(rng.end_col, line_width)
                if end_col <= start_col:
                    continue
                before = slice_by_column(line, 0, start_col, True)
                highlighted = slice_by_column(
                    line, start_col, end_col - start_col, True
                )
                after = slice_by_column(
                    line, end_col, max(0, line_width - end_col), True
                )
                line = (
                    f"{before}{self._apply_search_text_highlight(highlighted, rng.current)}"
                    f"{after}"
                )
            result[row] = line
        return result

    # ── 视口输入（对齐 TS handleViewportInput）──────────────────────────────

    def _should_defer_viewport_input_to_overlay(self) -> bool:
        return self._is_overlay_focused() and (
            self._active_search is None
            or self._active_search.overlay is None
            or not self._active_search.overlay.is_focused()
        )

    def _is_overlay_focused(self) -> bool:
        for entry in self._overlay_stack:
            if entry.component is self._focused_component and self._is_overlay_visible(
                entry
            ):
                return True
        return False

    def _handle_viewport_input(self, data: str) -> dict[str, object] | None:
        if data == FOCUS_OUT:
            had_active_selection = self._selection_press_active
            had_non_empty = (
                had_active_selection and self._get_selection_bounds() is not None
            )
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            self._stop_scrollbar_hover()
            self._stop_scrollbar_drag()
            self._pressed_url = None
            self._selection_dragged = False
            if had_active_selection:
                self._selection_anchor = None
                self._selection_focus = None
                self._selection_granularity = "character"
                self._selection_initial_range = None
                if had_non_empty:
                    self.request_render()
            self._last_click = None
            return {"consume": True}
        if data == FOCUS_IN:
            return {"consume": True}

        wheel_event = self._parse_wheel_event(data)
        if wheel_event is not None:
            if self._should_defer_viewport_input_to_overlay():
                return None
            self._route_wheel(wheel_event)
            return {"consume": True}

        mouse_event = self._parse_sgr_mouse_event(data)
        if mouse_event is not None:
            if self._handle_right_click_paste(mouse_event):
                return {"consume": True}
            handled = self._handle_scrollbar_mouse_event(mouse_event)
            if self._scrollbar_drag is None:
                self._update_scrollbar_hover(mouse_event.x, mouse_event.y)
            if not handled:
                self._handle_selection_mouse_event(mouse_event)
            return {"consume": True}
        if self._is_mouse_sequence(data):
            return {"consume": True}

        is_release = is_key_release(data)
        if matches_alt_screen_action(data, "tui.altScreen.search"):
            if not is_release:
                self._open_search()
            return {"consume": True}
        if (
            self._active_search is not None
            and self._active_search.overlay is not None
            and self._active_search.overlay.is_focused()
        ):
            if matches_alt_screen_action(data, "tui.altScreen.searchNext"):
                if not is_release:
                    self._navigate_search(1)
                return {"consume": True}
            if matches_alt_screen_action(data, "tui.altScreen.searchPrevious"):
                if not is_release:
                    self._navigate_search(-1)
                return {"consume": True}
            if matches_alt_screen_action(data, "tui.altScreen.searchClose"):
                if not is_release:
                    self._close_search()
                return {"consume": True}
        if self._should_defer_viewport_input_to_overlay():
            return None
        if matches_alt_screen_action(data, "tui.altScreen.pageUp"):
            if not is_release:
                self.scroll_by(
                    -max(
                        1,
                        self._get_primary_scroll_view().viewport_height
                        - PAGE_SCROLL_OVERLAP,
                    )
                )
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.pageDown"):
            if not is_release:
                self.scroll_by(
                    max(
                        1,
                        self._get_primary_scroll_view().viewport_height
                        - PAGE_SCROLL_OVERLAP,
                    )
                )
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.halfPageUp"):
            if not is_release:
                self.scroll_by(
                    -max(
                        1,
                        math.floor(self._get_primary_scroll_view().viewport_height / 2),
                    )
                )
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.halfPageDown"):
            if not is_release:
                self.scroll_by(
                    max(
                        1,
                        math.floor(self._get_primary_scroll_view().viewport_height / 2),
                    )
                )
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.lineUp"):
            if not is_release:
                self.scroll_by(-1)
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.lineDown"):
            if not is_release:
                self.scroll_by(1)
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.previousPrompt"):
            if not is_release:
                self._scroll_to_prompt(-1)
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.nextPrompt"):
            if not is_release:
                self._scroll_to_prompt(1)
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.top"):
            if not is_release:
                self.scroll_to_top()
            return {"consume": True}
        if matches_alt_screen_action(data, "tui.altScreen.bottom"):
            if not is_release:
                self.scroll_to_bottom()
            return {"consume": True}
        return None

    # ── 鼠标 / 滚轮解析 ────────────────────────────────────────────────────

    def _parse_wheel_event(self, data: str) -> WheelEvent | None:
        sgr = _WHEEL_SGR_RE.match(data)
        if sgr is not None:
            button = int(sgr.group(1))
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction != 0 and direction != 1:
                return None
            return WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=int(sgr.group(2)) - 1,
                y=int(sgr.group(3)) - 1,
            )
        if len(data) == 6 and data.startswith("\x1b[M"):
            button = ord(data[3]) - 32
            if (button & 64) == 0:
                return None
            direction = button & 3
            if direction != 0 and direction != 1:
                return None
            return WheelEvent(
                direction=-1 if direction == 0 else 1,
                x=ord(data[4]) - 33,
                y=ord(data[5]) - 33,
            )
        return None

    def _route_wheel(self, event: WheelEvent) -> None:
        remaining = event.direction * self._wheel_scroll_lines
        seen: set[ScrollView] = set()
        for scroll_view in get_scroll_views_at(self._current_layout, event.x, event.y):
            seen.add(scroll_view)
            remaining = scroll_view.scroll_by(remaining)
            if remaining == 0 or scroll_view.overscroll == "contain":
                break
        primary = self._get_primary_scroll_view()
        if remaining != 0 and primary not in seen:
            primary.scroll_by(remaining)
        self._update_scrollbar_hover(event.x, event.y)
        self.request_render()

    def _parse_sgr_mouse_event(self, data: str) -> SgrMouseEvent | None:
        match = _SGR_MOUSE_RE.match(data)
        if match is None:
            return None
        return SgrMouseEvent(
            button=int(match.group(1)),
            x=int(match.group(2)) - 1,
            y=int(match.group(3)) - 1,
            release=match.group(4) == "m",
        )

    def _handle_right_click_paste(self, event: SgrMouseEvent) -> bool:
        if (
            self._on_right_click_paste is None
            or sys.platform != "win32"
            or event.release
            or event.button != 2
        ):
            return False
        try:
            self._on_right_click_paste()
        except Exception:
            # 剪贴板粘贴尽力而为
            pass
        return True

    def _is_mouse_sequence(self, data: str) -> bool:
        return bool(re.match(r"^\x1b\[<\d+;\d+;\d+[Mm]$", data)) or (
            len(data) == 6 and data.startswith("\x1b[M")
        )

    # ── 滚动条 ─────────────────────────────────────────────────────────────

    def _get_scrollbar_target_at(self, x: int, y: int) -> ScrollbarTarget | None:
        if self.has_overlay() or self._current_layout is None:
            return None
        for scroll_view in get_scroll_views_at(self._current_layout, x, y):
            box = get_scroll_view_box(self._current_layout, scroll_view)
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if (
                geometry is not None
                and x == geometry.column
                and geometry.thumb_top <= y < geometry.thumb_top + geometry.thumb_height
            ):
                return ScrollbarTarget(scroll_view=scroll_view, geometry=geometry)
        return None

    def _set_scrollbar_hover(self, scroll_view: ScrollView | None) -> None:
        if scroll_view is self._scrollbar_hover:
            return
        if self._scrollbar_hover is not None:
            self._scrollbar_hover.set_scrollbar_active(False)
        self._scrollbar_hover = scroll_view
        if scroll_view is not None:
            scroll_view.set_scrollbar_active(True)

    def _update_scrollbar_hover(self, x: int, y: int) -> None:
        target = self._get_scrollbar_target_at(x, y)
        self._set_scrollbar_hover(target.scroll_view if target is not None else None)

    def _stop_scrollbar_hover(self) -> None:
        self._set_scrollbar_hover(None)

    def _handle_scrollbar_mouse_event(self, event: SgrMouseEvent) -> bool:
        if self._scrollbar_drag is not None:
            if event.release:
                self._stop_scrollbar_drag()
                return True
            box = (
                get_scroll_view_box(
                    self._current_layout, self._scrollbar_drag.scroll_view
                )
                if self._current_layout is not None
                else None
            )
            geometry = get_scrollbar_geometry(box) if box is not None else None
            if geometry is not None:
                max_thumb_offset = geometry.track_height - geometry.thumb_height
                thumb_offset = max(
                    0,
                    min(
                        max_thumb_offset,
                        event.y - geometry.track_top - self._scrollbar_drag.grab_offset,
                    ),
                )
                scroll_top = (
                    0
                    if max_thumb_offset == 0
                    else round(
                        (thumb_offset / max_thumb_offset) * geometry.max_scroll_top
                    )
                )
                self._scrollbar_drag.scroll_view.scroll_to(scroll_top)
            return True

        if event.release or (event.button & 32) != 0 or (event.button & 3) != 0:
            return False
        target = self._get_scrollbar_target_at(event.x, event.y)
        if target is None:
            return False
        self._stop_selection_auto_scroll()
        self._selection_press_active = False
        self._selection_anchor = None
        self._selection_focus = None
        self._selection_granularity = "character"
        self._selection_initial_range = None
        self._last_click = None
        self._pressed_url = None
        self._selection_dragged = False
        self._set_scrollbar_hover(target.scroll_view)
        self._scrollbar_drag = ScrollbarDrag(
            scroll_view=target.scroll_view,
            grab_offset=event.y - target.geometry.thumb_top,
        )
        return True

    def _stop_scrollbar_drag(self) -> None:
        self._scrollbar_drag = None

    # ── 选区 ───────────────────────────────────────────────────────────────

    def _get_scroll_selection_point(
        self, scroll_view: ScrollView, x: int, y: int
    ) -> SelectionPoint | None:
        if self._current_layout is None:
            return None
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            return None
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        if visible_bottom < visible_top:
            return None
        pointer_row = max(visible_top, min(visible_bottom, y))
        max_content_row = max(
            0,
            (
                len(box.scroll_content_lines)
                if box.scroll_content_lines is not None
                else 1
            )
            - 1,
        )
        return SelectionPoint(
            row=max(
                0,
                min(max_content_row, scroll_view.scroll_top + pointer_row - box.rect.y),
            ),
            col=max(0, min(box.rect.width - 1, x - box.rect.x)),
            scroll_view=scroll_view,
        )

    def _get_selection_point(
        self, event: SgrMouseEvent, scroll_view: ScrollView | None = None
    ) -> SelectionPoint:
        if scroll_view is not None:
            point = self._get_scroll_selection_point(scroll_view, event.x, event.y)
            if point is not None:
                return point
        return SelectionPoint(
            row=max(0, min(self.terminal.rows - 1, event.y)),
            col=max(0, min(self.terminal.columns - 1, event.x)),
        )

    def _get_selection_source_line(self, point: SelectionPoint) -> str:
        if point.scroll_view is not None and self._current_layout is not None:
            box = get_scroll_view_box(self._current_layout, point.scroll_view)
            if (
                box is not None
                and box.scroll_content_lines is not None
                and point.row < len(box.scroll_content_lines)
            ):
                return box.scroll_content_lines[point.row]
        if point.row < len(self._previous_screen):
            return self._previous_screen[point.row]
        return ""

    def _get_word_selection(self, point: SelectionPoint) -> SelectionRange | None:
        line = _strip_terminal_sequences(self._get_selection_source_line(point))
        for start, end in _word_segments(line):
            if point.col >= start and point.col < end:
                return SelectionRange(
                    start=SelectionPoint(
                        row=point.row, col=start, scroll_view=point.scroll_view
                    ),
                    end=SelectionPoint(
                        row=point.row,
                        col=end,
                        scroll_view=point.scroll_view,
                        boundary=True,
                    ),
                )
        return None

    def _get_line_selection(self, point: SelectionPoint) -> SelectionRange:
        return SelectionRange(
            start=SelectionPoint(row=point.row, col=0, scroll_view=point.scroll_view),
            end=SelectionPoint(
                row=point.row,
                col=visible_width(self._get_selection_source_line(point)),
                scroll_view=point.scroll_view,
                boundary=True,
            ),
        )

    def _update_selection_focus(self, point: SelectionPoint) -> None:
        if (
            self._selection_granularity == "character"
            or self._selection_initial_range is None
        ):
            self._selection_focus = point
            return
        if self._selection_granularity == "word":
            range_ = self._get_word_selection(point)
        else:
            range_ = self._get_line_selection(point)
        if range_ is None:
            return
        initial = self._selection_initial_range
        target_before_initial = range_.start.row < initial.start.row or (
            range_.start.row == initial.start.row
            and range_.start.col < initial.start.col
        )
        if target_before_initial:
            self._selection_anchor = initial.end
            self._selection_focus = range_.start
        else:
            self._selection_anchor = initial.start
            self._selection_focus = range_.end

    def _get_click_count(
        self, point: SelectionPoint, word: SelectionRange | None
    ) -> int:
        now = time.monotonic()
        previous = self._last_click
        if (
            word is not None
            and previous is not None
            and now - previous.timestamp <= DOUBLE_CLICK_INTERVAL_S
            and previous.row == point.row
            and previous.scroll_view is point.scroll_view
            and previous.word_start == word.start.col
            and previous.word_end == word.end.col
        ):
            count = (previous.count % 3) + 1
        else:
            count = 1
        if word is not None:
            self._last_click = ClickTarget(
                timestamp=now,
                count=count,
                row=point.row,
                scroll_view=point.scroll_view,
                word_start=word.start.col,
                word_end=word.end.col,
            )
        else:
            self._last_click = None
        return count

    def _update_selection_auto_scroll(self, event: SgrMouseEvent) -> None:
        scroll_view = (
            self._selection_anchor.scroll_view
            if self._selection_anchor is not None
            else None
        )
        if scroll_view is None or self._current_layout is None:
            self._stop_selection_auto_scroll()
            return
        box = get_scroll_view_box(self._current_layout, scroll_view)
        if box is None or box.rect.height <= 0 or box.clip.height <= 0:
            self._stop_selection_auto_scroll()
            return
        visible_top = max(0, box.rect.y, box.clip.y)
        visible_bottom = min(
            self.terminal.rows - 1,
            box.rect.y + box.rect.height - 1,
            box.clip.y + box.clip.height - 1,
        )
        self._selection_drag_pointer = (event.x, event.y)
        if event.y <= visible_top:
            direction = -1
        elif event.y >= visible_bottom:
            direction = 1
        else:
            direction = 0
        self._selection_auto_scroll_direction = direction
        if direction == 0:
            self._stop_selection_auto_scroll()
            return
        if self._selection_auto_scroll_timer is not None:
            return
        self._start_selection_auto_scroll()

    def _start_selection_auto_scroll(self) -> None:
        if self._selection_auto_scroll_timer is not None:
            return

        def _tick() -> None:
            self._auto_scroll_selection()
            if (
                self._selection_auto_scroll_direction != 0
                and self._selection_auto_scroll_timer is not None
            ):
                timer = threading.Timer(SELECTION_AUTO_SCROLL_INTERVAL_S, _tick)
                timer.daemon = True
                self._selection_auto_scroll_timer = timer
                timer.start()

        timer = threading.Timer(SELECTION_AUTO_SCROLL_INTERVAL_S, _tick)
        timer.daemon = True
        self._selection_auto_scroll_timer = timer
        timer.start()

    def _auto_scroll_selection(self) -> None:
        scroll_view = (
            self._selection_anchor.scroll_view
            if self._selection_anchor is not None
            else None
        )
        pointer = self._selection_drag_pointer
        direction = self._selection_auto_scroll_direction
        if scroll_view is None or pointer is None or direction == 0:
            self._stop_selection_auto_scroll()
            return
        remaining = scroll_view.scroll_by(direction)
        if remaining == direction:
            self._stop_selection_auto_scroll()
            return
        point = self._get_scroll_selection_point(scroll_view, pointer[0], pointer[1])
        if point is not None:
            self._update_selection_focus(point)
        self.request_render()

    def _stop_selection_auto_scroll(self) -> None:
        if self._selection_auto_scroll_timer is not None:
            self._selection_auto_scroll_timer.cancel()
            self._selection_auto_scroll_timer = None
        self._selection_auto_scroll_direction = 0
        self._selection_drag_pointer = None

    def _handle_selection_mouse_event(self, event: SgrMouseEvent) -> None:
        if (event.button & 3) != 0:
            return
        anchor_scroll_view = (
            self._selection_anchor.scroll_view
            if self._selection_anchor is not None
            else None
        )
        point = self._get_selection_point(event, anchor_scroll_view)
        if event.release:
            if not self._selection_press_active:
                return
            self._selection_press_active = False
            self._stop_selection_auto_scroll()
            if self._selection_anchor is None:
                return
            self._update_selection_focus(point)
            clicked_url: str | None = None
            if (
                not self._selection_dragged
                and self._selection_anchor.scroll_view is point.scroll_view
                and self._selection_anchor.row == point.row
                and self._selection_anchor.col == point.col
            ):
                clicked_url = self._pressed_url
            self._pressed_url = None
            if clicked_url is not None and self._open_url is not None:
                self._selection_anchor = None
                self._selection_focus = None
                try:
                    self._open_url(clicked_url)
                except Exception:
                    # URL 激活尽力而为
                    pass
                self.request_render()
                return
            self._copy_selection_to_clipboard()
            self.request_render()
            return
        if (event.button & 32) != 0:
            if not self._selection_press_active or self._selection_anchor is None:
                return
            self._selection_dragged = True
            self._last_click = None
            self._pressed_url = None
            self._update_selection_focus(point)
            self._update_selection_auto_scroll(event)
            self.request_render()
            return
        self._stop_selection_auto_scroll()
        self._selection_press_active = True
        scroll_view: ScrollView | None = None
        if not self.has_overlay() and self._current_layout is not None:
            views = get_scroll_views_at(self._current_layout, event.x, event.y)
            if views:
                scroll_view = views[0]
        anchor = self._get_selection_point(event, scroll_view)
        word = self._get_word_selection(anchor)
        click_count = self._get_click_count(anchor, word)
        if click_count == 2:
            range_: SelectionRange | None = word
        elif click_count == 3:
            range_ = self._get_line_selection(anchor)
        else:
            range_ = None
        if range_ is not None and click_count == 2:
            self._selection_granularity = "word"
        elif range_ is not None:
            self._selection_granularity = "line"
        else:
            self._selection_granularity = "character"
        self._selection_initial_range = range_
        self._selection_anchor = range_.start if range_ is not None else anchor
        self._selection_focus = range_.end if range_ is not None else anchor
        self._selection_dragged = False
        if range_ is not None:
            self._pressed_url = None
        else:
            row = max(0, min(self.terminal.rows - 1, event.y))
            col = max(0, min(self.terminal.columns - 1, event.x))
            line = (
                self._previous_screen[row] if row < len(self._previous_screen) else ""
            )
            self._pressed_url = _get_osc8_link_at_column(line, col)
        self.request_render()

    def _get_selection_bounds(self) -> SelectionRange | None:
        if self._selection_anchor is None or self._selection_focus is None:
            return None
        if self._selection_anchor.scroll_view is not self._selection_focus.scroll_view:
            return None
        anchor_before_focus = (
            self._selection_anchor.row < self._selection_focus.row
            or (
                self._selection_anchor.row == self._selection_focus.row
                and self._selection_anchor.col < self._selection_focus.col
            )
        )
        if (
            self._selection_anchor.row == self._selection_focus.row
            and self._selection_anchor.col == self._selection_focus.col
        ):
            return None
        if anchor_before_focus:
            return SelectionRange(
                start=self._selection_anchor, end=self._selection_focus
            )
        return SelectionRange(start=self._selection_focus, end=self._selection_anchor)

    def _get_selection_columns(
        self,
        line: str,
        row: int,
        selection: SelectionRange,
        min_column: int = 0,
        max_column: int | None = None,
    ) -> tuple[int, int]:
        line_width = visible_width(line)
        if max_column is None:
            max_column = line_width
        start = max(0, min_column)
        end = min(line_width, max_column)
        if row == selection.start.row:
            range_ = _get_grapheme_cell_range(line, selection.start.col)
            start = (
                range_[0]
                if range_ is not None
                else min(selection.start.col, line_width)
            )
        if row == selection.end.row:
            if selection.end.boundary:
                end = min(selection.end.col, line_width)
            else:
                range_ = _get_grapheme_cell_range(line, selection.end.col)
                end = (
                    range_[1]
                    if range_ is not None
                    else min(selection.end.col + 1, line_width)
                )
        return max(min_column, start), min(max_column, end)

    def _copy_selection_to_clipboard(self) -> None:
        import base64

        selection = self._get_selection_bounds()
        if selection is None:
            return
        source_lines: list[str] = self._previous_screen
        if selection.start.scroll_view is not None:
            if self._current_layout is None:
                return
            box = get_scroll_view_box(self._current_layout, selection.start.scroll_view)
            if box is None or box.scroll_content_lines is None:
                return
            source_lines = box.scroll_content_lines
        lines: list[str] = []
        for row in range(selection.start.row, selection.end.row + 1):
            line = source_lines[row] if row < len(source_lines) else ""
            columns = self._get_selection_columns(line, row, selection)
            lines.append(
                _strip_terminal_sequences(
                    slice_by_column(
                        line, columns[0], max(0, columns[1] - columns[0]), True
                    )
                ).rstrip()
            )
        text = "\n".join(lines)
        if not text:
            return
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        self.terminal.write(f"\x1b]52;c;{encoded}\x07")
        self.flash("Copied!")

    def _apply_selection_highlight(self, text: str) -> str:
        """将选区文本渲染为反显，保留行内 ANSI 序列（SGR 后重新开启反显）。"""
        result = "\x1b[7m"
        index = 0
        while index < len(text):
            ansi = extract_ansi_code(text, index)
            if ansi is None:
                result += text[index]
                index += 1
                continue
            result += ansi.code
            if ansi.code.endswith("m"):
                result += "\x1b[7m"
            index += ansi.length
        return f"{result}\x1b[27m"

    def _apply_selection(
        self, screen: list[str], layout: LayoutFrame | None = None
    ) -> list[str]:
        layout = layout if layout is not None else self._current_layout
        selection = self._get_selection_bounds()
        if selection is None:
            return screen
        screen_selection = selection
        min_row = 0
        max_row = len(screen) - 1
        min_column = 0
        max_column = self.terminal.columns
        if selection.start.scroll_view is not None:
            if layout is None:
                return screen
            box = get_scroll_view_box(layout, selection.start.scroll_view)
            if box is None:
                return screen
            min_row = max(0, box.rect.y, box.clip.y)
            max_row = min(
                len(screen) - 1,
                box.rect.y + box.rect.height - 1,
                box.clip.y + box.clip.height - 1,
            )
            min_column = max(0, box.rect.x, box.clip.x)
            max_column = min(
                self.terminal.columns,
                box.rect.x + box.rect.width,
                box.clip.x + box.clip.width,
            )
            start = SelectionPoint(
                row=box.rect.y
                + selection.start.row
                - selection.start.scroll_view.scroll_top,
                col=box.rect.x + selection.start.col,
                scroll_view=selection.start.scroll_view,
                boundary=selection.start.boundary,
            )
            end = SelectionPoint(
                row=box.rect.y
                + selection.end.row
                - selection.start.scroll_view.scroll_top,
                col=box.rect.x + selection.end.col,
                scroll_view=selection.start.scroll_view,
                boundary=selection.end.boundary,
            )
            screen_selection = SelectionRange(start=start, end=end)
        result: list[str] = []
        for row, line in enumerate(screen):
            if (
                row < min_row
                or row > max_row
                or row < screen_selection.start.row
                or row > screen_selection.end.row
                or is_image_line(line)
            ):
                result.append(line)
                continue
            line_width = visible_width(line)
            columns = self._get_selection_columns(
                line, row, screen_selection, min_column, max_column
            )
            if columns[1] <= columns[0]:
                result.append(line)
                continue
            before = slice_by_column(line, 0, columns[0], True)
            selected = slice_by_column(line, columns[0], columns[1] - columns[0], True)
            after = slice_by_column(
                line, columns[1], max(0, line_width - columns[1]), True
            )
            result.append(f"{before}{self._apply_selection_highlight(selected)}{after}")
        return result

    # ── 闪烁消息合成 ────────────────────────────────────────────────────────

    def _composite_flashes(
        self, screen: list[str], width: int, height: int
    ) -> list[str]:
        flash_lines = self._flashes.render(width)[-height:]
        if not flash_lines:
            return screen
        result = list(screen)
        while len(result) < height:
            result.append("")
        for row, line in enumerate(flash_lines):
            flash_width = visible_width(line)
            if flash_width == 0:
                continue
            result[row] = self._composite_line_at(
                result[row], line, width - flash_width, flash_width, width
            )
        return result

    # ── 提示符跳转 ──────────────────────────────────────────────────────────

    def _scroll_to_prompt(self, direction: int) -> None:
        if self._current_layout is None:
            return
        scroll_view = self._get_primary_scroll_view()
        box = get_scroll_view_box(self._current_layout, scroll_view)
        lines = box.scroll_content_lines if box is not None else None
        if not lines:
            return
        row = scroll_view.scroll_top + direction
        while 0 <= row < len(lines):
            if _OSC133_PROMPT_START.match(lines[row]):
                scroll_view.scroll_to(row)
                self.request_render()
                return
            row += direction

    # ── 主渲染管线（对齐 TS doRender）──────────────────────────────────────

    def _do_render(self) -> None:
        if self._stopped or not self._alt_screen_active:
            return
        width = max(1, self.terminal.columns)
        height = max(1, self.terminal.rows)
        root = (
            self._layout_root
            if self._layout_root is not None
            else self._implicit_scroll_view
        )
        next_layout = render_layout_frame(
            root, width, height, lambda: self.request_render()
        )
        if self._refresh_search(next_layout):
            next_layout = render_layout_frame(
                root, width, height, lambda: self.request_render()
            )
        screen = [_strip_osc133_zone(line) for line in next_layout.lines]
        screen = self._apply_search_highlights(screen, next_layout)
        screen = self._composite_overlays(screen, width, height)
        if len(screen) > height:
            screen = screen[len(screen) - height :]
        screen = self._apply_selection(screen, next_layout)
        screen = self._composite_flashes(screen, width, height)

        cursor_pos = self._extract_cursor_position(screen, height)
        screen = self._apply_line_resets(screen)
        screen = [
            line
            if is_image_line(line) or visible_width(line) <= width
            else slice_by_column(line, 0, width, True)
            for line in screen
        ]

        full_redraw = (
            len(self._previous_screen) == 0
            or self._previous_screen_width != width
            or self._previous_screen_height != height
        )
        buffer = BEGIN_SYNCHRONIZED_OUTPUT
        if full_redraw:
            self._full_redraw_count += 1
            buffer += f"{self._delete_kitty_images()}\x1b[2J"
        for row in range(height):
            line = screen[row] if row < len(screen) else ""
            if not full_redraw and self._previous_screen[row] == line:
                continue
            buffer += f"\x1b[{row + 1};1H\x1b[2K{line}"
        if cursor_pos is not None:
            buffer += f"\x1b[{cursor_pos[0] + 1};{min(width, cursor_pos[1]) + 1}H"
            buffer += "\x1b[?25h" if self.get_show_hardware_cursor() else "\x1b[?25l"
        else:
            buffer += "\x1b[?25l"
        buffer += END_SYNCHRONIZED_OUTPUT
        self.terminal.write(buffer)

        self._previous_screen = screen
        self._previous_screen_width = width
        self._previous_screen_height = height
        self._current_layout = next_layout
