"""VStack 组件（垂直弹性布局）— mirrors components/v-stack.ts

对应 TypeScript 源文件：packages/tui/src/components/v-stack.ts。
"""

from __future__ import annotations

import sys
from typing import Literal

from ..layout_node import LayoutViewport
from .stack import (
    Stack,
    StackChild,
    StackEntry,
    StackEntryOptions,
    StackOptions,
    allocate_stack_sizes,
    visible_stack_entries,
)

__all__ = [
    "Stack",
    "StackChild",
    "StackEntry",
    "StackEntryOptions",
    "StackOptions",
    "VStack",
]


class VStack(Stack):
    """垂直排列的弹性布局容器。

    Mirrors VStack in components/v-stack.ts。
    """

    layout_type: Literal["vstack", "hstack"] = "vstack"

    def render(self, width: int) -> list[str]:
        """把子组件按垂直方向拼接渲染（含 gap 与 basis 高度填充）。

        对应 v-stack.ts 的 render()：无固定高度（available_size=None），
        因此渲染阶段不做 grow/shrink，弹性高度分配由 layout.py 布局引擎完成。
        """
        viewport = LayoutViewport(max(1, width), sys.maxsize)
        entries = visible_stack_entries(self.entries, viewport)
        rendered = [entry.component.render(viewport.width) for entry in entries]
        sizes = allocate_stack_sizes(
            entries,
            [len(lines) for lines in rendered],
            None,
            self.gap,
        )
        lines: list[str] = []
        for index, _entry in enumerate(entries):
            if index > 0:
                lines.extend([""] * self.gap)
            child_lines = rendered[index][: sizes[index]]
            lines.extend(child_lines)
            lines.extend([""] * (sizes[index] - len(child_lines)))
        return lines
