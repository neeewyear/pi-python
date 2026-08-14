"""布局引擎 — mirrors layout.ts

对应 TypeScript 源文件：packages/tui/src/layout.ts。

负责把组件树布局为 LayoutBox 树（含 vstack / hstack 弹性尺寸分配、
ScrollView 视口裁剪与滚动状态更新），并把各盒子按 clip 区域绘制到
屏幕行数组上（含滚动条绘制）。
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import NamedTuple

from .components.stack import allocate_stack_sizes, visible_stack_entries
from .layout_node import (
    AlignValue,
    LayoutViewport,
    ScrollLayoutNode,
    ScrollLayoutState,
    StackLayoutEntry,
    get_layout_node,
)
from .terminal_image import is_image_line
from .tui import CURSOR_MARKER, Component
from .utils import (
    extract_ansi_code,
    extract_segments,
    slice_by_column,
    slice_with_width,
    visible_width,
)

#: 段重置序列（SGR 重置 + OSC 8 超链接终止），等价 tui.ts 的 SEGMENT_RESET
_SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"

#: OSC 133 提示区域前缀（iTerm2 shell 集成），绘制时剥离，等价 layout.ts 的 OSC133_ZONE_PREFIX
_OSC133_ZONE_PREFIX_RE = re.compile(r"^(?:\x1b\]133;[ABC](?:\x07|\x1b\\))+")


@dataclass
class LayoutRect:
    """布局矩形（x/y 为左上角），对应 layout.ts 的 LayoutRect 接口。"""

    x: int
    y: int
    width: int
    height: int


@dataclass
class LayoutBox:
    """布局盒子树节点，对应 layout.ts 的 LayoutBox 接口。"""

    component: Component
    rect: LayoutRect
    clip: LayoutRect
    children: list[LayoutBox] = field(default_factory=list)
    parent: LayoutBox | None = None
    lines: list[str] | None = None
    line_offset: int = 0
    scroll_view: ScrollLayoutState | None = None
    scroll_content_lines: list[str] | None = None
    layer: int = 0


@dataclass
class LayoutFrame:
    """一帧的布局结果，对应 layout.ts 的 LayoutFrame 接口。"""

    root: LayoutBox
    width: int
    height: int
    lines: list[str]
    primary_scroll_view: ScrollLayoutState | None = None


@dataclass
class ScrollbarGeometry:
    """滚动条几何信息，对应 layout.ts 的 ScrollbarGeometry 接口。"""

    column: int
    track_top: int
    track_height: int
    thumb_top: int
    thumb_height: int
    max_scroll_top: int


@dataclass
class _LayoutContext:
    """单帧布局上下文（视口、按组件/宽度的渲染缓存、重渲染回调）。"""

    viewport: LayoutViewport
    request_render: Callable[[], None]
    render_cache: dict[object, dict[int, list[str]]] = field(default_factory=dict)
    primary_scroll_view: ScrollLayoutState | None = None


def _intersect(a: LayoutRect, b: LayoutRect) -> LayoutRect:
    """求两个矩形相交区域（无交集时宽高为 0）。"""
    x = max(a.x, b.x)
    y = max(a.y, b.y)
    right = min(a.x + a.width, b.x + b.width)
    bottom = min(a.y + a.height, b.y + b.height)
    return LayoutRect(x, y, max(0, right - x), max(0, bottom - y))


def _render_cached(
    context: _LayoutContext,
    component: Component,
    width: int,
) -> list[str]:
    """带每帧缓存的组件渲染。"""    
    safe_width = max(1, math.floor(width))
    widths = context.render_cache.get(component)
    if widths is None:
        widths = {}
        context.render_cache[component] = widths
    lines = widths.get(safe_width)
    if lines is None:
        lines = component.render(safe_width)
        widths[safe_width] = lines
    return lines


def _measure_height(
    context: _LayoutContext,
    component: Component,
    width: int,
) -> int:
    return len(_render_cached(context, component, width))


def _measure_width(
    context: _LayoutContext,
    component: Component,
    width: int,
) -> int:
    return max(
        (visible_width(line) for line in _render_cached(context, component, width)),
        default=0,
    )


def _with_parent(box: LayoutBox, parent: LayoutBox) -> LayoutBox:
    box.parent = parent
    return box


def _translate_box(box: LayoutBox, delta_y: int) -> None:
    """递归平移盒子的 y 坐标。"""
    box.rect.y += delta_y
    for child in box.children:
        _translate_box(child, delta_y)


def _update_clips(box: LayoutBox, parent_clip: LayoutRect) -> None:
    """递归更新 clip（与父 clip 求交）。"""
    box.clip = _intersect(parent_clip, box.rect)
    for child in box.children:
        _update_clips(child, box.clip)


def _find_cursor_line(lines: list[str]) -> int | None:
    """查找含 CURSOR_MARKER 的行号；不存在返回 None。"""
    for index, line in enumerate(lines):
        if CURSOR_MARKER in line:
            return index
    return None


def _layout_non_node(
    context: _LayoutContext,
    component: Component,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    """普通组件（无布局节点）的布局：行内按渲染行数，行间允许裁剪。"""
    lines = _render_cached(context, component, width)
    allocated_height = len(lines) if height is None else max(0, math.floor(height))
    line_offset = 0
    if len(lines) > allocated_height > 0:
        cursor_line = _find_cursor_line(lines)
        if cursor_line is not None and cursor_line >= allocated_height:
            line_offset = cursor_line - allocated_height + 1
    rect = LayoutRect(x, y, width, allocated_height)
    return LayoutBox(
        component=component,
        rect=rect,
        clip=_intersect(clip, rect),
        children=[],
        lines=lines,
        line_offset=line_offset,
        layer=0,
    )


def _layout_scroll_node(
    context: _LayoutContext,
    component: Component,
    node: ScrollLayoutNode,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    """ScrollView 节点的布局：视口裁剪 + 滚动状态更新。

    对应 layout.ts layoutComponent() 的 scroll 分支：先以
    y - scrollTop 布局子内容，再根据 updateLayout 后的新 scrollTop 平移。
    """
    previous_scroll_top = node.state.scroll_top
    content_width = node.state.get_content_width(width)
    child_box = _layout_component(
        context,
        node.component,
        x,
        y - previous_scroll_top,
        content_width,
        None,
        clip,
    )
    content_height = child_box.rect.height
    viewport_height = content_height if height is None else max(0, math.floor(height))
    node.state.update_layout(content_height, viewport_height, context.request_render)
    _translate_box(child_box, previous_scroll_top - node.state.scroll_top)
    if node.state.primary or context.primary_scroll_view is None:
        context.primary_scroll_view = node.state
    rect = LayoutRect(x, y, width, viewport_height)
    child_clip = _intersect(clip, rect)
    box = LayoutBox(
        component=component,
        rect=rect,
        clip=child_clip,
        children=[child_box],
        scroll_view=node.state,
        scroll_content_lines=_render_cached(context, node.component, content_width),
        layer=0,
    )
    child_box.parent = box
    _update_clips(child_box, child_clip)
    return box


def _layout_vstack_node(
    context: _LayoutContext,
    component: Component,
    entries: list[StackLayoutEntry],
    gap: int,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    """VStack 布局：按高度分配子项并依次排列。"""
    intrinsic_heights = [
        entry.basis
        if isinstance(entry.basis, int)
        else _measure_height(context, entry.component, width)
        for entry in entries
    ]
    sizes = allocate_stack_sizes(entries, intrinsic_heights, height, gap)
    natural_height = sum(sizes) + max(0, len(entries) - 1) * gap
    allocated_height = natural_height if height is None else max(0, math.floor(height))
    rect = LayoutRect(x, y, width, allocated_height)
    box = LayoutBox(
        component=component,
        rect=rect,
        clip=_intersect(clip, rect),
        children=[],
        layer=0,
    )
    child_y = y
    for index, entry in enumerate(entries):
        box.children.append(
            _with_parent(
                _layout_component(
                    context, entry.component, x, child_y, width, sizes[index], box.clip
                ),
                box,
            )
        )
        child_y += sizes[index] + gap
    return box


def _layout_hstack_node(
    context: _LayoutContext,
    component: Component,
    entries: list[StackLayoutEntry],
    gap: int,
    align: AlignValue,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    """HStack 布局：按宽度分配子项，水平排列并做垂直对齐。"""
    intrinsic_widths = [
        entry.basis
        if isinstance(entry.basis, int)
        else _measure_width(context, entry.component, width)
        for entry in entries
    ]
    widths = allocate_stack_sizes(entries, intrinsic_widths, width, gap)
    intrinsic_heights = [
        _measure_height(context, entry.component, max(1, widths[index]))
        for index, entry in enumerate(entries)
    ]
    allocated_height = (
        max(intrinsic_heights, default=0)
        if height is None
        else max(0, math.floor(height))
    )
    rect = LayoutRect(x, y, width, allocated_height)
    box = LayoutBox(
        component=component,
        rect=rect,
        clip=_intersect(clip, rect),
        children=[],
        layer=0,
    )
    child_x = x
    for index, entry in enumerate(entries):
        natural_child_height = intrinsic_heights[index]
        child_height = (
            allocated_height
            if align == "stretch"
            else min(allocated_height, natural_child_height)
        )
        child_y = y
        if align == "center":
            child_y += (allocated_height - child_height) // 2
        elif align == "end":
            child_y += allocated_height - child_height
        child_width = widths[index]
        if child_width == 0:
            box.children.append(
                LayoutBox(
                    component=entry.component,
                    rect=LayoutRect(child_x, child_y, 0, child_height),
                    clip=LayoutRect(child_x, child_y, 0, 0),
                    children=[],
                    parent=box,
                    layer=0,
                )
            )
        else:
            box.children.append(
                _with_parent(
                    _layout_component(
                        context,
                        entry.component,
                        child_x,
                        child_y,
                        child_width,
                        child_height,
                        box.clip,
                    ),
                    box,
                )
            )
        child_x += child_width + gap
    return box


def _layout_component(
    context: _LayoutContext,
    component: Component,
    x: int,
    y: int,
    width: int,
    height: int | None,
    clip: LayoutRect,
) -> LayoutBox:
    """把单个组件布局为 LayoutBox。

    对应 layout.ts 的 layoutComponent()：按布局节点类型分发
    （普通组件 / scroll / vstack / hstack）。
    """
    safe_width = max(1, math.floor(width))
    node = get_layout_node(component)
    if node is None:
        return _layout_non_node(context, component, x, y, safe_width, height, clip)
    if isinstance(node, ScrollLayoutNode):
        return _layout_scroll_node(
            context, component, node, x, y, safe_width, height, clip
        )
    entries = visible_stack_entries(node.entries, context.viewport)
    if node.type == "vstack":
        return _layout_vstack_node(
            context, component, entries, node.gap, x, y, safe_width, height, clip
        )
    return _layout_hstack_node(
        context,
        component,
        entries,
        node.gap,
        node.align,
        x,
        y,
        safe_width,
        height,
        clip,
    )


def composite_tui_line(
    base_line: str,
    overlay_line: str,
    start_col: int,
    overlay_width: int,
    total_width: int,
) -> str:
    """把 overlay_line 合成到 base_line 的 [start_col, start_col+overlay_width) 区间。

    等价 tui.ts 的 compositeTuiLine()（HStack 渲染与 paintBox 合成共用）。
    说明：该函数在 TS 中定义于 tui.ts；Python 端 tui.py 仅有 TUI 类内部的
    等价方法，故在 layout.py 提供模块级实现供组件复用。
    """
    if is_image_line(base_line):
        return base_line
    after_start = start_col + overlay_width
    base = extract_segments(
        base_line, start_col, after_start, total_width - after_start, True
    )
    overlay = slice_with_width(overlay_line, 0, overlay_width, True)
    before_pad = max(0, start_col - base.before_width)
    overlay_pad = max(0, overlay_width - overlay.width)
    actual_before_width = max(start_col, base.before_width)
    actual_overlay_width = max(overlay_width, overlay.width)
    after_target = max(0, total_width - actual_before_width - actual_overlay_width)
    after_pad = max(0, after_target - base.after_width)
    result = (
        base.before
        + " " * before_pad
        + _SEGMENT_RESET
        + overlay.text
        + " " * overlay_pad
        + _SEGMENT_RESET
        + base.after
        + " " * after_pad
    )
    if visible_width(result) <= total_width:
        return result
    return slice_by_column(result, 0, total_width, True)


def _style_scrollbar_cell(
    line: str,
    column: int,
    total_width: int,
    style: Callable[[str], str],
) -> str:
    """对指定列单元应用滚动条样式。
    Python 端无 getGraphemeCellRange 工具，按 1 个单元宽度
    切片近似处理（宽字符落在滚动条列时以样式空格覆盖）。
    """
    if is_image_line(line):
        return line
    before = slice_by_column(line, 0, column, True)
    target = slice_by_column(line, column, 1, True)
    after = slice_by_column(line, column + 1, max(0, total_width - column - 1), True)
    before_padding = " " * max(0, column - visible_width(before))
    # 提取 target 前缀的 ANSI 代码，使样式只包裹纯文本部分
    target_prefix = ""
    target_index = 0
    while target_index < len(target):
        ansi = extract_ansi_code(target, target_index)
        if ansi is None:
            break
        target_prefix += ansi.code
        target_index += ansi.length
    target_text = target[target_index:] or " "
    return f"{before}{before_padding}{target_prefix}{style(target_text)}{after}"


def get_scrollbar_geometry(box: LayoutBox) -> ScrollbarGeometry | None:
    """计算滚动条几何信息；不可见或超出 clip 时返回 None。

    对应 layout.ts 的 getScrollbarGeometry()。
    """
    if (
        box.scroll_view is None
        or not box.scroll_view.is_scrollbar_visible
        or box.rect.width <= 0
        or box.rect.height <= 0
    ):
        return None
    if box.children:
        content_height = box.children[0].rect.height
    elif box.scroll_content_lines is not None:
        content_height = len(box.scroll_content_lines)
    else:
        content_height = 0
    track_height = box.rect.height
    min_thumb_height = min(2, track_height)
    if content_height <= 0:
        thumb_height = track_height
    else:
        thumb_height = max(
            min_thumb_height,
            min(
                track_height,
                round((track_height * track_height) / content_height),
            ),
        )
    max_scroll_top = max(0, content_height - track_height)
    max_thumb_top = track_height - thumb_height
    thumb_offset = (
        0
        if max_scroll_top == 0
        else round((box.scroll_view.scroll_top / max_scroll_top) * max_thumb_top)
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


def _paint_scrollbar(box: LayoutBox, screen: list[str], total_width: int) -> None:
    """绘制滚动条滑块（thumb）。对应 layout.ts 的 paintScrollbar()。"""
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


def _paint_box(box: LayoutBox, screen: list[str], total_width: int) -> None:
    """把盒子（含子盒子）按 clip 绘制到屏幕行数组。

    对应 layout.ts 的 paintBox()。
    """
    if box.lines is not None:
        offset = box.line_offset
        first_row = max(box.rect.y, box.clip.y, 0)
        last_row = min(
            box.rect.y + box.rect.height,
            box.clip.y + box.clip.height,
            len(screen),
        )
        for row in range(first_row, last_row):
            source_index = offset + row - box.rect.y
            if source_index < 0 or source_index >= len(box.lines):
                continue
            source_line = box.lines[source_index]
            line = _OSC133_ZONE_PREFIX_RE.sub("", source_line)
            metadata = get_kitty_image_metadata(line)
            if metadata is not None:
                clip_bottom = min(len(screen), box.clip.y + box.clip.height)
                visible_rows = min(metadata.rows, clip_bottom - row)
                if visible_rows < metadata.rows:
                    line = crop_kitty_image_line(line, 0, visible_rows)
            if (
                box.rect.x == 0
                and box.rect.width >= total_width
                and (is_image_line(line) or not screen[row])
            ):
                screen[row] = line
            else:
                screen[row] = composite_tui_line(
                    screen[row], line, box.rect.x, box.rect.width, total_width
                )
    for child in box.children:
        _paint_box(child, screen, total_width)

    # 滚动视图滚出内容中 Kitty 图像行的处理
    # 说明：Python 端 terminal_image 尚无图像注册机制，注册表默认为空，此块
    # 在图像功能接入后生效。
    if (
        box.scroll_view is not None
        and box.scroll_content_lines is not None
        and box.scroll_view.scroll_top > 0
        and box.rect.height > 0
    ):
        for image_row in range(box.scroll_view.scroll_top - 1, -1, -1):
            image_line = (
                box.scroll_content_lines[image_row]
                if image_row < len(box.scroll_content_lines)
                else ""
            )
            metadata = get_kitty_image_metadata(image_line)
            if metadata is not None:
                hidden_rows = box.scroll_view.scroll_top - image_row
                if hidden_rows < metadata.rows:
                    visible_rows = min(box.rect.height, metadata.rows - hidden_rows)
                    cropped = crop_kitty_image_line(
                        image_line, hidden_rows, visible_rows
                    )
                    if box.rect.x == 0 and box.rect.width >= total_width:
                        if 0 <= box.rect.y < len(screen):
                            screen[box.rect.y] = cropped
                break
            if image_line != "":
                break

    _paint_scrollbar(box, screen, total_width)


def render_layout_frame(
    root: Component,
    width: int,
    height: int,
    request_render: Callable[[], None],
) -> LayoutFrame:
    """渲染一帧：布局组件树并按 clip 绘制，返回帧结果。

    对应 layout.ts 的 renderLayoutFrame()。
    """
    safe_width = max(1, math.floor(width))
    safe_height = max(1, math.floor(height))
    context = _LayoutContext(
        viewport=LayoutViewport(safe_width, safe_height),
        request_render=request_render,
    )
    root_box = _layout_component(
        context,
        root,
        0,
        0,
        safe_width,
        safe_height,
        LayoutRect(0, 0, safe_width, safe_height),
    )
    lines = [""] * safe_height
    _paint_box(root_box, lines, safe_width)
    return LayoutFrame(
        root=root_box,
        width=safe_width,
        height=safe_height,
        lines=lines,
        primary_scroll_view=context.primary_scroll_view,
    )


def _contains_point(rect: LayoutRect, x: int, y: int) -> bool:
    return (
        x >= rect.x
        and x < rect.x + rect.width
        and y >= rect.y
        and y < rect.y + rect.height
    )


def get_scroll_view_box(
    frame: LayoutFrame,
    scroll_view: ScrollLayoutState,
) -> LayoutBox | None:
    """在帧的盒子树中查找指定 ScrollView 对应的盒子。

    对应 layout.ts 的 getScrollViewBox()。
    """

    def visit(box: LayoutBox) -> LayoutBox | None:
        if box.scroll_view is scroll_view:
            return box
        for child in box.children:
            found = visit(child)
            if found is not None:
                return found
        return None

    return visit(frame.root)


def get_scroll_views_at(frame: LayoutFrame, x: int, y: int) -> list[ScrollLayoutState]:
    """返回 (x, y) 处命中的 ScrollView 列表（深度优先、命中者按深度降序）。

    对应 layout.ts 的 getScrollViewsAt()。
    """
    result: list[tuple[ScrollLayoutState, int]] = []

    def visit(box: LayoutBox, depth: int) -> None:
        if not _contains_point(box.clip, x, y):
            return
        if box.scroll_view is not None and _contains_point(box.rect, x, y):
            result.append((box.scroll_view, depth))
        for child in box.children:
            visit(child, depth + 1)

    visit(frame.root, 0)
    result.sort(key=lambda item: item[1], reverse=True)
    return [entry for entry, _depth in result]


# ─────────────────────────────────────────────────────────────────────────────
# Kitty 图像元数据注册表 — 对应 terminal-image.ts 的注册机制
# ─────────────────────────────────────────────────────────────────────────────


class KittyImageMetadata(NamedTuple):
    """Kitty 图像元数据，对应 terminal-image.ts 的 KittyImageMetadata 接口。"""

    image_id: int
    columns: int
    rows: int
    width_px: int
    height_px: int


_kitty_image_registry: dict[int, KittyImageMetadata] = {}

_KITTY_ESCAPE_RE = re.compile(r"\x1b_G([^;]*);")
_IMAGE_ID_RE = re.compile(r"(?:^|,)i=(\d+)(?:,|$)")


def register_kitty_image_metadata(metadata: KittyImageMetadata) -> None:
    """注册 Kitty 图像元数据（按 image_id，容量上限 1000）。

    对应 terminal-image.ts 的 registerKittyImageMetadata()。
    """
    _kitty_image_registry[metadata.image_id] = metadata
    if len(_kitty_image_registry) > 1000:
        oldest_id = next(iter(_kitty_image_registry))
        _kitty_image_registry.pop(oldest_id, None)


def get_kitty_image_metadata(line: str) -> KittyImageMetadata | None:
    """从图像行解析注册表中的元数据；未注册返回 None。

    对应 terminal-image.ts 的 getKittyImageMetadata()。
    """
    match = _KITTY_ESCAPE_RE.search(line)
    if match is None:
        return None
    id_match = _IMAGE_ID_RE.search(match.group(1))
    if id_match is None:
        return None
    return _kitty_image_registry.get(int(id_match.group(1)))


def crop_kitty_image_line(line: str, hidden_rows: int, visible_rows: int) -> str:
    """裁剪 Kitty 图像行：调整 y/h/r 控制参数。

    对应 terminal-image.ts 的 cropKittyImageLine()。
    """
    metadata = get_kitty_image_metadata(line)
    match = _KITTY_ESCAPE_RE.search(line)
    if (
        metadata is None
        or match is None
        or hidden_rows < 0
        or hidden_rows >= metadata.rows
        or visible_rows <= 0
    ):
        return line
    cropped_rows = min(visible_rows, metadata.rows - hidden_rows)
    if hidden_rows == 0 and cropped_rows == metadata.rows:
        return line
    source_y = math.floor((metadata.height_px * hidden_rows) / metadata.rows)
    source_end = math.ceil(
        (metadata.height_px * (hidden_rows + cropped_rows)) / metadata.rows
    )
    source_height = max(1, min(metadata.height_px, source_end) - source_y)
    controls = [
        control
        for control in match.group(1).split(",")
        if not re.match(r"^[yhr]=", control)
    ]
    controls.append(f"y={source_y}")
    controls.append(f"h={source_height}")
    controls.append(f"r={cropped_rows}")
    return line[: match.start()] + f"\x1b_G{','.join(controls)};" + line[match.end() :]
