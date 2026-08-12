"""ScrollView 组件 — mirrors components/scroll-view.ts

对应 TypeScript 源文件：packages/tui/src/components/scroll-view.ts。

滚动状态（scrollTop / follow-end / 滚动条可见性）由本组件维护，
视口裁剪与滚动条绘制由 layout.py 的布局引擎在布局/绘制阶段完成，
与 TS 的分工一致。
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from ..layout_node import ScrollLayoutNode
from ..tui import Component, Container

#: 滚动条显示模式，对应 scroll-view.ts 的 ScrollViewScrollbar
ScrollViewScrollbar = Literal["hidden", "auto", "always"]


@dataclass
class ScrollViewOptions:
    """ScrollView 选项，对应 scroll-view.ts 的 ScrollViewOptions 接口。"""

    axis: Literal["vertical"] | None = None
    follow: Literal["none", "end"] | None = None
    primary: bool = False
    overscroll: Literal["chain", "contain"] | None = None
    scrollbar: ScrollViewScrollbar | None = None
    scrollbar_style: Callable[[str], str] | None = None
    scrollbar_hide_delay_ms: int | None = None


@dataclass
class ScrollViewScrollToOptions:
    """scroll_to 选项，对应 scroll-view.ts 的 ScrollViewScrollToOptions 接口。"""

    #: 目标为当前内容末尾时也禁用 follow-end（即使内容末尾就是目标）
    disable_follow: bool = False


def _default_scrollbar_style(text: str) -> str:
    """默认滚动条样式：灰色背景。对应 scroll-view.ts 的默认 scrollbarStyle。"""
    return f"\x1b[100m{text}\x1b[49m"


class ScrollView(Container):
    """单子组件滚动容器，支持 follow-end 与滚动条。

    Mirrors ScrollView in components/scroll-view.ts。
    """

    def __init__(
        self,
        component: Component,
        options: ScrollViewOptions | None = None,
    ) -> None:
        super().__init__()
        opt = options if options is not None else ScrollViewOptions()
        if opt.axis is not None and opt.axis != "vertical":
            raise ValueError(f"Unsupported ScrollView axis: {opt.axis}")
        self.child = component
        self.children.append(component)
        self.follow_end = (opt.follow or "none") == "end"
        self.following_end = self.follow_end
        self._primary = opt.primary
        self._overscroll: Literal["chain", "contain"] = opt.overscroll or "chain"
        self.current_scrollbar: ScrollViewScrollbar = opt.scrollbar or "hidden"
        self._scrollbar_style = (
            opt.scrollbar_style
            if opt.scrollbar_style is not None
            else _default_scrollbar_style
        )
        self.scrollbar_hide_delay_ms = max(
            0, math.floor(opt.scrollbar_hide_delay_ms or 1000)
        )
        self.current_scroll_top = 0
        self.content_height = 0
        self.current_viewport_height = 0
        self.follow_suppressed_at_end = False
        self.request_render_callback: Callable[[], None] | None = None
        self.transient_scrollbar_visible = False
        self.scrollbar_active = False
        self._scrollbar_hide_timer: asyncio.TimerHandle | None = None
        # render 缓存（按 width 键控）；invalidate() 时清除
        self._cache_width: int | None = None
        self._cache_lines: list[str] | None = None

    @property
    def scroll_top(self) -> int:
        """当前滚动偏移（行）。"""
        return self.current_scroll_top

    @property
    def primary(self) -> bool:
        """是否为主滚动视图（对应 scroll-view.ts 的 readonly primary）。"""
        return self._primary

    @property
    def overscroll(self) -> Literal["chain", "contain"]:
        """到达边界时的滚动行为（对应 scroll-view.ts 的 readonly overscroll）。"""
        return self._overscroll

    @property
    def scrollbar_style(self) -> Callable[[str], str]:
        """滚动条单元样式函数（对应 scroll-view.ts 的 readonly scrollbarStyle）。"""
        return self._scrollbar_style

    @property
    def is_following_end(self) -> bool:
        """是否处于跟随末尾状态。"""
        return self.following_end

    @property
    def viewport_height(self) -> int:
        """最近一次布局的视口高度。"""
        return self.current_viewport_height

    @property
    def scrollbar(self) -> ScrollViewScrollbar:
        """当前滚动条显示模式。"""
        return self.current_scrollbar

    @property
    def is_scrollbar_visible(self) -> bool:
        """滚动条当前是否可见。对应 scroll-view.ts 的 isScrollbarVisible。"""
        if self.current_scrollbar == "always":
            return self.current_viewport_height > 0
        return (
            self.current_scrollbar == "auto"
            and self.content_height > self.current_viewport_height
            and self.transient_scrollbar_visible
        )

    def set_scrollbar(self, scrollbar: ScrollViewScrollbar) -> None:
        """切换滚动条显示模式。对应 scroll-view.ts 的 setScrollbar()。"""
        if scrollbar == self.current_scrollbar:
            return
        self.current_scrollbar = scrollbar
        if scrollbar != "auto":
            self._hide_transient_scrollbar()
        elif self.scrollbar_active:
            self._mark_scrollbar_activity()
        if self.request_render_callback is not None:
            self.request_render_callback()

    def get_content_width(self, width: int) -> int:
        """返回内容宽度：always 滚动条时预留最后一列。

        对应 scroll-view.ts 的 getContentWidth()。
        """
        return width - 1 if self.current_scrollbar == "always" and width > 1 else width

    def _mark_scrollbar_activity(self) -> None:
        """标记滚动活动：auto 模式下使临时滚动条可见并重置隐藏定时器。

        对应 scroll-view.ts 的 markScrollbarActivity()。TS 用 setTimeout().unref()；
        Python 端在存在运行中事件循环时用 loop.call_later 调度隐藏，
        无运行循环时保持临时可见（由下一次活动或 hideTransientScrollbar 复位）。
        """
        if (
            self.current_scrollbar != "auto"
            or self.content_height <= self.current_viewport_height
        ):
            return
        self.transient_scrollbar_visible = True
        if self._scrollbar_hide_timer is not None:
            self._scrollbar_hide_timer.cancel()
            self._scrollbar_hide_timer = None
        if self.scrollbar_active:
            return
        delay = self.scrollbar_hide_delay_ms / 1000.0
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return
        self._scrollbar_hide_timer = loop.call_later(
            delay, self._on_scrollbar_hide_timeout
        )

    def _on_scrollbar_hide_timeout(self) -> None:
        """滚动条隐藏定时器回调。"""
        self._scrollbar_hide_timer = None
        self.transient_scrollbar_visible = False
        if self.request_render_callback is not None:
            self.request_render_callback()

    def _hide_transient_scrollbar(self) -> None:
        """立即隐藏临时滚动条并取消定时器。对应 scroll-view.ts 的 hideTransientScrollbar()。"""
        self.transient_scrollbar_visible = False
        if self._scrollbar_hide_timer is None:
            return
        self._scrollbar_hide_timer.cancel()
        self._scrollbar_hide_timer = None

    def set_scrollbar_active(self, active: bool) -> None:
        """设置滚动条激活状态（如拖动/按键交互期间）。

        对应 scroll-view.ts 的 setScrollbarActive()。
        """
        if active == self.scrollbar_active:
            return
        self.scrollbar_active = active
        self._mark_scrollbar_activity()

    def scroll_to(
        self,
        scroll_top: int,
        options: ScrollViewScrollToOptions | None = None,
    ) -> None:
        """滚动到指定偏移（行），并维护 follow-end 状态。

        对应 scroll-view.ts 的 scrollTo()。
        """
        opt = options if options is not None else ScrollViewScrollToOptions()
        requested = math.trunc(scroll_top)
        max_scroll_top = max(0, self.content_height - self.current_viewport_height)
        next_scroll = max(0, min(max_scroll_top, requested))
        next_follow_suppressed_at_end = (
            opt.disable_follow and next_scroll == max_scroll_top
        )
        next_following_end = (
            not next_follow_suppressed_at_end
            and self.follow_end
            and next_scroll == max_scroll_top
        )
        if (
            next_scroll == self.current_scroll_top
            and next_following_end == self.following_end
            and next_follow_suppressed_at_end == self.follow_suppressed_at_end
        ):
            return
        moved = next_scroll != self.current_scroll_top
        self.current_scroll_top = next_scroll
        self.following_end = next_following_end
        self.follow_suppressed_at_end = next_follow_suppressed_at_end
        if moved:
            self._mark_scrollbar_activity()
        if self.request_render_callback is not None:
            self.request_render_callback()

    def scroll_by(self, lines: int) -> int:
        """相对滚动，返回未消耗的滚动量（到边界后的溢出）。

        对应 scroll-view.ts 的 scrollBy()。
        """
        requested = math.trunc(lines)
        if requested == 0:
            return 0
        max_scroll_top = max(0, self.content_height - self.current_viewport_height)
        start = max_scroll_top if self.following_end else self.current_scroll_top
        next_scroll = max(0, min(max_scroll_top, start + requested))
        moved = next_scroll - start
        was_following_end = self.following_end
        self.current_scroll_top = next_scroll
        self.following_end = self.follow_end and next_scroll == max_scroll_top
        self.follow_suppressed_at_end = False
        if moved != 0:
            self._mark_scrollbar_activity()
        if moved != 0 or self.following_end != was_following_end:
            if self.request_render_callback is not None:
                self.request_render_callback()
        return requested - moved

    def scroll_to_start(self) -> None:
        """滚动到顶部并解除 follow-end。

        对应 scroll-view.ts 的 scrollToStart()。
        """
        changed = self.current_scroll_top != 0 or self.following_end != (
            self.follow_end and self.content_height <= self.current_viewport_height
        )
        self.current_scroll_top = 0
        self.following_end = (
            self.follow_end and self.content_height <= self.current_viewport_height
        )
        self.follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self.request_render_callback is not None:
                self.request_render_callback()

    def scroll_to_end(self) -> None:
        """滚动到底部（重新启用 follow-end）。

        对应 scroll-view.ts 的 scrollToEnd()。
        """
        next_scroll = max(0, self.content_height - self.current_viewport_height)
        changed = (
            self.current_scroll_top != next_scroll
            or self.following_end != self.follow_end
        )
        self.current_scroll_top = next_scroll
        self.following_end = self.follow_end
        self.follow_suppressed_at_end = False
        if changed:
            self._mark_scrollbar_activity()
            if self.request_render_callback is not None:
                self.request_render_callback()

    def update_layout(
        self,
        content_height: int,
        viewport_height: int,
        request_render: Callable[[], None],
    ) -> None:
        """布局阶段更新内容高度 / 视口高度，并夹紧滚动偏移、维护 follow-end。

        对应 scroll-view.ts 的 updateLayout()。
        """
        self.content_height = max(0, math.floor(content_height))
        self.current_viewport_height = max(0, math.floor(viewport_height))
        self.request_render_callback = request_render
        max_scroll_top = max(0, self.content_height - self.current_viewport_height)
        if self.following_end:
            self.current_scroll_top = max_scroll_top
        else:
            self.current_scroll_top = max(
                0, min(self.current_scroll_top, max_scroll_top)
            )
        if self.current_scroll_top < max_scroll_top:
            self.follow_suppressed_at_end = False
        if (
            self.follow_end
            and self.current_scroll_top == max_scroll_top
            and not self.follow_suppressed_at_end
        ):
            self.following_end = True
        if self.content_height <= self.current_viewport_height:
            self._hide_transient_scrollbar()

    def add_child(self, _component: object) -> None:
        """ScrollView 只允许一个子组件。对应 scroll-view.ts 的 addChild()。"""
        raise RuntimeError("ScrollView has exactly one child")

    def remove_child(self, _component: object) -> None:
        """ScrollView 子组件不可移除。对应 scroll-view.ts 的 removeChild()。"""
        raise RuntimeError("ScrollView child cannot be removed")

    def clear(self) -> None:
        """ScrollView 子组件不可清空。对应 scroll-view.ts 的 clear()。"""
        raise RuntimeError("ScrollView child cannot be cleared")

    def invalidate(self) -> None:
        """使 render 缓存失效并传递给子组件。

        缓存按 render 的宽度键控，子组件内容变化后需调用本方法刷新输出。
        """
        self._cache_width = None
        self._cache_lines = None
        self.child.invalidate()

    def render(self, width: int) -> list[str]:
        """渲染子组件内容（预留滚动条列时在行尾补空格）。

        对应 scroll-view.ts 的 render()；视口裁剪由 layout.py 布局引擎完成。
        """
        if self._cache_width == width and self._cache_lines is not None:
            return self._cache_lines
        content_width = self.get_content_width(width)
        lines = self.child.render(content_width)
        if content_width != width:
            lines = [f"{line} " for line in lines]
        self._cache_width = width
        self._cache_lines = lines
        return lines

    def layout_node(self) -> ScrollLayoutNode:
        """返回本组件的布局节点。对应 scroll-view.ts 的 [LAYOUT_NODE]()。"""
        return ScrollLayoutNode(type="scroll", component=self.child, state=self)
