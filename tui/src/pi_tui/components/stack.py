"""Stack 抽象基类与弹性布局尺寸分配 — mirrors components/stack.ts

对应 TypeScript 源文件：packages/tui/src/components/stack.ts。
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeGuard, cast

from ..layout_node import (
    LayoutViewport,
    StackEntryOptions,
    StackLayoutEntry,
    StackLayoutNode,
)
from ..tui import Component, Container

AlignValue = Literal["stretch", "start", "center", "end"]

__all__ = [
    "AlignValue",
    "Stack",
    "StackChild",
    "StackEntry",
    "StackEntryOptions",
    "StackOptions",
    "allocate_stack_sizes",
    "clamp_size",
    "is_stack_entry",
    "normalize_size",
    "visible_stack_entries",
]

#: Stack 子项（裸组件或带布局选项的条目），对应 stack.ts 的 StackEntry（StackEntryOptions + component）
StackEntry = StackLayoutEntry
#: Stack 子项联合类型，对应 stack.ts 的 StackChild
StackChild = Component | StackEntry


@dataclass
class StackOptions:
    """Stack 容器选项，对应 stack.ts 的 StackOptions 接口。"""

    gap: int | None = None
    align: AlignValue | None = None


def is_stack_entry(child: StackChild) -> TypeGuard[StackEntry]:
    """判断子项是否为 StackEntry（含布局选项）而非裸组件。

    对应 stack.ts 的 isStackEntry()：TS 用 ``"render" in child`` 判断，
    Python 端用 ``hasattr(child, "render")`` 等价判断。
    """
    return not hasattr(child, "render")


def normalize_size(value: int | None, fallback: int) -> int:
    """规范化尺寸：None 时回退 fallback，否则取 ≥0 的整数（向下取整）。

    对应 stack.ts 的 normalizeSize()。TS 用 Number.isFinite 防御 NaN，
    Python 的 int 恒有限，故省略该判断。
    """
    if value is None:
        return fallback
    return max(0, math.floor(value))


def clamp_size(size: int, entry: StackLayoutEntry) -> int:
    """将尺寸限制在条目的 min_size / max_size 范围内。

    对应 stack.ts 的 clampSize()。
    """
    minimum = max(0, math.floor(entry.min_size if entry.min_size is not None else 0))
    maximum = max(
        minimum,
        math.floor(entry.max_size if entry.max_size is not None else sys.maxsize),
    )
    return max(minimum, min(maximum, max(0, math.floor(size))))


GrowShrinkMode = Literal["grow", "shrink"]


def _entry_can_change(entry: StackLayoutEntry, size: int, mode: GrowShrinkMode) -> bool:
    """判断条目在 grow/shrink 模式下是否还能继续变化（受权重与 min/max 约束）。"""
    if mode == "grow":
        return (entry.grow if entry.grow is not None else 0) > 0 and size < (
            entry.max_size if entry.max_size is not None else sys.maxsize
        )
    return (entry.shrink if entry.shrink is not None else 1) > 0 and size > (
        entry.min_size if entry.min_size is not None else 0
    )


def _entry_weight(entry: StackLayoutEntry, size: int, mode: GrowShrinkMode) -> int:
    """条目在 grow/shrink 分配中的权重（shrink 时与当前尺寸成正比）。"""
    if mode == "grow":
        return entry.grow if entry.grow is not None else 0
    return (entry.shrink if entry.shrink is not None else 1) * max(1, size)


def _distribute(
    sizes: list[int],
    entries: Sequence[StackLayoutEntry],
    amount: int,
    mode: GrowShrinkMode,
) -> None:
    """按权重把 amount 分配到可变化的条目上（grow 增加 / shrink 减少）。

    对应 stack.ts 的 distribute()：逐轮按比例分配，直到无可分配空间为止。
    """
    remaining = amount
    while remaining > 0:
        candidates = [
            (index, entry)
            for index, entry in enumerate(entries)
            if _entry_can_change(entry, sizes[index], mode)
        ]
        if not candidates:
            return
        total_weight = sum(
            _entry_weight(entry, sizes[index], mode) for index, entry in candidates
        )
        distributed = 0
        for index, entry in candidates:
            if remaining <= 0:
                break
            weight = _entry_weight(entry, sizes[index], mode)
            proposed = max(1, math.floor((remaining * weight) / total_weight))
            if mode == "grow":
                capacity = (
                    entry.max_size if entry.max_size is not None else sys.maxsize
                ) - sizes[index]
                delta = min(remaining, proposed, capacity)
                sizes[index] = sizes[index] + delta
            else:
                capacity = sizes[index] - (
                    entry.min_size if entry.min_size is not None else 0
                )
                delta = min(remaining, proposed, capacity)
                sizes[index] = sizes[index] - delta
            remaining -= delta
            distributed += delta
        if distributed == 0:
            return


def allocate_stack_sizes(
    entries: Sequence[StackLayoutEntry],
    intrinsic_sizes: Sequence[int],
    available_size: int | None,
    gap: int,
) -> list[int]:
    """计算各条目分配的尺寸（含 basis、grow/shrink 分配与 min/max 约束）。

    对应 stack.ts 的 allocateStackSizes()。
    """
    sizes: list[int] = []
    for index, entry in enumerate(entries):
        basis = entry.basis
        if isinstance(basis, int):
            raw = basis
        else:
            # basis 为 None 或 "auto"：取固有尺寸
            raw = intrinsic_sizes[index]
        sizes.append(clamp_size(raw, entry))
    if available_size is None:
        return sizes
    content_size = max(0, math.floor(available_size) - max(0, len(entries) - 1) * gap)
    total = sum(sizes)
    if total < content_size:
        _distribute(sizes, entries, content_size - total, "grow")
    elif total > content_size:
        _distribute(sizes, entries, total - content_size, "shrink")
    return sizes


def visible_stack_entries(
    entries: Sequence[StackLayoutEntry],
    viewport: LayoutViewport,
) -> list[StackLayoutEntry]:
    """按条目的 visible 回调过滤条目（未提供 visible 时默认可见）。

    对应 stack.ts 的 visibleStackEntries()。
    """
    return [
        entry for entry in entries if entry.visible is None or entry.visible(viewport)
    ]


class Stack(Container):
    """弹性布局容器的抽象基类（VStack / HStack 的公共逻辑）。

    Mirrors Stack in components/stack.ts。
    """

    layout_type: Literal["vstack", "hstack"]

    def __init__(
        self,
        children: Sequence[StackChild] = (),
        options: StackOptions | None = None,
    ) -> None:
        super().__init__()
        opt = options if options is not None else StackOptions()
        self.gap = normalize_size(opt.gap, 0)
        self.align = opt.align or "stretch"
        self.entries: list[StackLayoutEntry] = []
        for child in children:
            if is_stack_entry(child):
                self.add_child(child.component, child)
            else:
                self.add_child(child)

    def add_child(
        self,
        component: object,
        options: StackEntryOptions | StackLayoutEntry | None = None,
    ) -> None:
        """添加子组件，可选弹性布局选项。

        对应 stack.ts 的 addChild()：grow/shrink/min/max 按 TS 语义归一化
        （grow 默认 0、shrink 默认 1、min 默认 0、max 默认 sys.maxsize）。
        options 接受 StackEntryOptions 或 StackLayoutEntry（后者由构造函数
        传入整个子项，二者字段一致）。
        """
        comp = cast(Component, component)
        super().add_child(comp)
        entry = StackLayoutEntry(component=comp)
        if options is not None:
            if options.basis is not None:
                entry.basis = options.basis
            if options.grow is not None:
                entry.grow = normalize_size(options.grow, 0)
            if options.shrink is not None:
                entry.shrink = normalize_size(options.shrink, 1)
            if options.min_size is not None:
                entry.min_size = normalize_size(options.min_size, 0)
            if options.max_size is not None:
                entry.max_size = normalize_size(options.max_size, sys.maxsize)
            if options.visible is not None:
                entry.visible = options.visible
        self.entries.append(entry)

    def remove_child(self, component: object) -> None:
        """移除子组件及其布局条目。对应 stack.ts 的 removeChild()。"""
        super().remove_child(component)
        for index, entry in enumerate(self.entries):
            if entry.component is component:
                del self.entries[index]
                break

    def clear(self) -> None:
        """清空所有子组件与布局条目。对应 stack.ts 的 clear()。"""
        super().clear()
        self.entries = []

    def layout_node(self) -> StackLayoutNode:
        """返回本组件的布局节点。对应 stack.ts 的 [LAYOUT_NODE]()。"""
        return StackLayoutNode(
            type=self.layout_type,
            entries=list(self.entries),
            gap=self.gap,
            align=self.align,
        )
