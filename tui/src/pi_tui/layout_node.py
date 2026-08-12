"""布局节点协议 — mirrors layout-node.ts

对应 TypeScript 源文件：packages/tui/src/layout-node.ts。

TS 用 Symbol.for("@earendil-works/pi-tui/layout-node") 作为组件上的方法键，
Python 端用固定方法名 ``layout_node()`` 等价表示（见 LAYOUT_NODE 常量）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, NamedTuple, Protocol, runtime_checkable

from .tui import Component

#: 组件实现布局节点的固定方法名（等价 TS 的 LAYOUT_NODE Symbol 键）
LAYOUT_NODE = "layout_node"

#: 弹性布局对齐方式，对应 layout-node.ts 的 align 联合类型
AlignValue = Literal["stretch", "start", "center", "end"]


class LayoutViewport(NamedTuple):
    """布局视口尺寸（宽、高），对应 layout-node.ts 的 LayoutViewport 接口。"""

    width: int
    height: int


@dataclass
class StackEntryOptions:
    """Stack 子项的弹性布局选项，对应 stack.ts 的 StackEntryOptions 接口。

    说明：TS 中该类型定义在 stack.ts；为避免 stack.py 与 layout_node.py 之间循环导入，
    这里与 StackLayoutEntry 一并定义（字段一致）。
    """

    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackLayoutEntry:
    """Stack 布局条目（组件 + 弹性选项），对应 layout-node.ts 的 StackLayoutEntry 接口。

    dataclass 字段顺序：component 无默认值置前，选项字段带默认值置后。
    """

    component: Component
    basis: int | Literal["auto"] | None = None
    grow: int | None = None
    shrink: int | None = None
    min_size: int | None = None
    max_size: int | None = None
    visible: Callable[[LayoutViewport], bool] | None = None


@dataclass
class StackLayoutNode:
    """Stack 布局节点，对应 layout-node.ts 的 StackLayoutNode 接口。"""

    type: Literal["vstack", "hstack"]
    entries: list[StackLayoutEntry]
    gap: int
    align: AlignValue


@runtime_checkable
class ScrollLayoutState(Protocol):
    """滚动布局状态接口，对应 layout-node.ts 的 ScrollLayoutState 接口。

    由 ScrollView 结构实现。除 TS 接口成员外，额外包含布局引擎绘制滚动条
    所需的只读成员（is_scrollbar_visible / scrollbar_style）。
    TS 中这些成员均为只读，因此协议端用只读 property 声明。
    """

    @property
    def scroll_top(self) -> int: ...

    @property
    def primary(self) -> bool: ...

    @property
    def overscroll(self) -> Literal["chain", "contain"]: ...

    @property
    def viewport_height(self) -> int: ...

    @property
    def is_scrollbar_visible(self) -> bool: ...

    @property
    def scrollbar_style(self) -> Callable[[str], str]: ...

    def get_content_width(self, width: int) -> int:
        """返回内容宽度（always 滚动条时预留一列）。"""
        ...

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        """在布局阶段更新内容高度 / 视口高度并夹紧滚动偏移。"""
        ...


@dataclass
class ScrollLayoutNode:
    """滚动布局节点，对应 layout-node.ts 的 ScrollLayoutNode 接口。"""

    type: Literal["scroll"]
    component: Component
    state: ScrollLayoutState


#: 布局节点联合类型（判别联合），对应 layout-node.ts 的 LayoutNode
LayoutNode = StackLayoutNode | ScrollLayoutNode


@runtime_checkable
class LayoutComponent(Protocol):
    """实现了 layout_node() 方法的组件，对应 layout-node.ts 的 LayoutComponent 接口。"""

    def layout_node(self) -> LayoutNode:
        """返回组件的布局节点。"""
        ...


def get_layout_node(component: object) -> LayoutNode | None:
    """返回组件的布局节点；组件未实现布局协议时返回 None。

    对应 layout-node.ts 的 getLayoutNode()。
    """
    if isinstance(component, LayoutComponent):
        return component.layout_node()
    return None
