"""HStack 组件（水平弹性布局）— mirrors components/h-stack.ts

对应 TypeScript 源文件：packages/tui/src/components/h-stack.ts。
"""

from __future__ import annotations

import sys
from typing import Literal

from ..layout import composite_tui_line
from ..layout_node import LayoutViewport
from ..utils import visible_width
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
    "HStack",
]


class HStack(Stack):
    """水平排列的弹性布局容器。

    Mirrors HStack in components/h-stack.ts。
    """

    layout_type: Literal["vstack", "hstack"] = "hstack"

    def render(self, width: int) -> list[str]:
        """把子组件按水平方向拼接渲染（含 gap、宽度分配与垂直对齐）。

        对应 h-stack.ts 的 render()：用 compositeTuiLine 逐行合成，
        输出行的可见宽度不超过 width。
        """
        safe_width = max(1, width)
        viewport = LayoutViewport(safe_width, sys.maxsize)
        entries = visible_stack_entries(self.entries, viewport)
        if not entries:
            return []

        intrinsic_widths = [
            max(
                (visible_width(line) for line in entry.component.render(safe_width)),
                default=0,
            )
            for entry in entries
        ]
        widths = allocate_stack_sizes(entries, intrinsic_widths, safe_width, self.gap)
        rendered = [
            [] if widths[index] == 0 else entry.component.render(widths[index])
            for index, entry in enumerate(entries)
        ]
        height = max((len(lines) for lines in rendered), default=0)
        result: list[str] = [""] * height
        x = 0
        for index, lines in enumerate(rendered):
            child_width = widths[index]
            offset = 0
            if self.align == "center":
                offset = (height - len(lines)) // 2
            elif self.align == "end":
                offset = height - len(lines)
            for row, line in enumerate(lines):
                target = row + offset
                if target < 0 or target >= len(result):
                    continue
                result[target] = composite_tui_line(
                    result[target], line, x, child_width, safe_width
                )
            x += child_width + self.gap
        return result
