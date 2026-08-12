"""
TUI — Terminal UI with differential rendering.
Mirrors packages/tui/src/tui.ts exactly.

Provides:
- Component: protocol for renderable+interactive components
- Focusable: protocol for components that can receive hardware cursor focus
- Container: component that holds children
- TUI: main class with differential rendering, overlay compositing
- CURSOR_MARKER: APC sequence for cursor positioning
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .keys import is_key_release, matches_key
from .terminal import Terminal
from .terminal_colors import (
    RgbColor,
    TerminalColorScheme,
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from .terminal_image import (
    CellDimensions,
    get_capabilities,
    is_image_line,
    set_cell_dimensions,
)
from .utils import extract_segments, slice_by_column, slice_with_width, visible_width

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Component protocol
# ─────────────────────────────────────────────────────────────────────────────


@runtime_checkable
class Component(Protocol):
    """
    All components must implement this protocol.
    Mirrors the Component interface in tui.ts.
    """

    def render(self, width: int) -> list[str]:
        """Render component to lines. Each line ≤ width visible chars."""
        ...

    def invalidate(self) -> None:
        """Invalidate cached rendering state."""
        ...

    def handle_input(self, data: str) -> None:
        """Handle keyboard input when component has focus."""
        ...


@runtime_checkable
class Focusable(Protocol):
    """
    Components that can receive focus and display a hardware cursor.
    When focused, emit CURSOR_MARKER at the cursor position in render output.
    Mirrors the Focusable interface in tui.ts.
    """

    focused: bool


def is_focusable(component: object) -> bool:
    """Check if a component implements Focusable."""
    return component is not None and hasattr(component, "focused")


# Cursor position marker — APC sequence (zero-width, terminals ignore it).
# Components emit this at their cursor position when focused.
# TUI finds, strips it, and positions the hardware cursor there.
CURSOR_MARKER = "\x1b_pi:c\x07"

# ─────────────────────────────────────────────────────────────────────────────
# Overlay types
# ─────────────────────────────────────────────────────────────────────────────

OverlayAnchor = str  # "center" | "top-left" | ... (string alias)


@dataclass
class OverlayMargin:
    top: int = 0
    right: int = 0
    bottom: int = 0
    left: int = 0


# SizeValue: int (absolute) or str like "50%"
SizeValue = int | str


def _parse_size_value(value: SizeValue | None, ref: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    import re

    m = re.match(r"^(\d+(?:\.\d+)?)%$", value)
    if m:
        return int(ref * float(m.group(1)) / 100)
    return None


@dataclass
class OverlayOptions:
    width: SizeValue | None = None
    min_width: int | None = None
    max_height: SizeValue | None = None
    anchor: str = "center"
    offset_x: int = 0
    offset_y: int = 0
    row: SizeValue | None = None
    col: SizeValue | None = None
    margin: OverlayMargin | int | None = None
    visible: Callable[[int, int], bool] | None = None
    non_capturing: bool = False  # NEW: If true, don't capture keyboard focus when shown


@dataclass
class OverlayEntry:
    component: object  # Component
    options: OverlayOptions | None
    pre_focus: object | None  # Component | None
    hidden: bool = False
    focus_order: int = 0  # NEW: For sorting overlays by focus order


class OverlayHandle:
    """Handle returned by show_overlay(). Controls overlay visibility and focus."""

    def __init__(
        self,
        hide_fn: Callable[[], None],
        set_hidden_fn: Callable[[bool], None],
        is_hidden_fn: Callable[[], bool],
        entry: OverlayEntry,
        tui: TUI,
    ) -> None:
        self._hide = hide_fn
        self._set_hidden = set_hidden_fn
        self._is_hidden = is_hidden_fn
        self._entry = entry
        self._tui = tui

    def hide(self) -> None:
        self._hide()

    def set_hidden(self, hidden: bool) -> None:
        self._set_hidden(hidden)

    def is_hidden(self) -> bool:
        return self._is_hidden()

    def focus(self) -> None:
        """Set focus to this overlay."""
        self._tui._focused_overlay = self._entry
        self._entry.focus_order = self._tui._next_focus_order
        self._tui._next_focus_order += 1
        self._tui.request_render()

    def unfocus(self) -> None:
        """Remove focus from this overlay."""
        if self._tui._focused_overlay == self._entry:
            self._tui._focused_overlay = None
            self._tui.request_render()

    def is_focused(self) -> bool:
        """Check if this overlay has focus."""
        return self._tui._focused_overlay == self._entry


# ─────────────────────────────────────────────────────────────────────────────
# Container
# ─────────────────────────────────────────────────────────────────────────────


class Container:
    """
    A component that contains other components.
    Mirrors Container in tui.ts.
    """

    def __init__(self) -> None:
        self.children: list[object] = []  # list[Component]

    def add_child(self, component: object) -> None:
        self.children.append(component)

    def remove_child(self, component: object) -> None:
        try:
            self.children.remove(component)
        except ValueError:
            pass

    def clear(self) -> None:
        self.children = []

    def invalidate(self) -> None:
        for child in self.children:
            if hasattr(child, "invalidate"):
                child.invalidate()

    def render(self, width: int) -> list[str]:
        lines: list[str] = []
        for child in self.children:
            if hasattr(child, "render"):
                lines.extend(child.render(width))
        return lines

    def handle_input(self, data: str) -> None:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# TUI
# ─────────────────────────────────────────────────────────────────────────────

InputListenerResult = dict | None  # type: ignore[type-arg]  # {"consume": bool, "data": str}
InputListener = Callable[[str], InputListenerResult]

_SEGMENT_RESET = "\x1b[0m\x1b]8;;\x07"


class TUI(Container):
    """
    Main class for managing terminal UI with differential rendering.
    Mirrors TUI in tui.ts exactly — same algorithm, same edge cases.
    """

    # 最小渲染间隔（毫秒），对齐 TS TuiBase.MIN_RENDER_INTERVAL_MS = 16（60fps 上限）。
    # 流式 token 级事件合并到 16ms 一帧，避免高频全屏 diff 重绘造成闪烁。
    MIN_RENDER_INTERVAL_MS = 16.0

    def __init__(
        self, terminal: Terminal, show_hardware_cursor: bool | None = None
    ) -> None:
        super().__init__()
        self.terminal = terminal

        env_cursor = os.environ.get("PI_HARDWARE_CURSOR", "0") == "1"
        self._show_hardware_cursor = (
            show_hardware_cursor if show_hardware_cursor is not None else env_cursor
        )
        self._clear_on_shrink = os.environ.get("PI_CLEAR_ON_SHRINK", "0") == "1"

        self._previous_lines: list[str] = []
        self._previous_width: int = 0
        self._previous_height: int = 0
        self._focused_component: object | None = None
        self._input_listeners: list[InputListener] = []

        self.on_debug: Callable[[], None] | None = None

        self._render_requested = False
        self._render_timer: asyncio.TimerHandle | None = None
        self._last_render_at = 0.0
        self._cursor_row = 0
        self._hardware_cursor_row = 0
        self._input_buffer = ""
        self._cell_size_query_pending = False
        self._max_lines_rendered = 0
        self._previous_viewport_top = 0
        self._full_redraw_count = 0
        self._stopped = False

        # OSC 11 背景色查询：待处理的 future 队列（支持并发查询）
        self._pending_osc11_queries: deque[asyncio.Future[RgbColor | None]] = deque()
        # DSR 996 配色方案查询：等待报告的 future 监听器
        self._terminal_color_scheme_listeners: list[
            asyncio.Future[TerminalColorScheme | None]
        ] = []

        self._overlay_stack: list[OverlayEntry] = []
        self._focused_overlay: OverlayEntry | None = None  # NEW: Track focused overlay
        self._next_focus_order: int = 0  # NEW: For focus ordering
        self._main_loop: asyncio.AbstractEventLoop | None = None

        # AltScreen 模式标志：由 enter_alt_screen()/exit_alt_screen() 切换。
        # 完整的备用屏幕渲染管线由独立渲染器 pi_tui.tui_alt_screen.TuiAltScreen
        # （对标 TS TuiAltScreen，继承本类）提供。
        self._alt_screen_active = False

    @property
    def stopped(self) -> bool:
        return self._stopped

    @property
    def full_redraws(self) -> int:
        return self._full_redraw_count

    def get_show_hardware_cursor(self) -> bool:
        return self._show_hardware_cursor

    def set_show_hardware_cursor(self, enabled: bool) -> None:
        if self._show_hardware_cursor == enabled:
            return
        self._show_hardware_cursor = enabled
        if not enabled:
            self.terminal.hide_cursor()
        self.request_render()

    def get_clear_on_shrink(self) -> bool:
        return self._clear_on_shrink

    def set_clear_on_shrink(self, enabled: bool) -> None:
        self._clear_on_shrink = enabled

    def set_focus(self, component: object | None) -> None:
        if is_focusable(self._focused_component):
            self._focused_component.focused = False  # type: ignore
        self._focused_component = component
        if is_focusable(component):
            component.focused = True  # type: ignore

    def show_overlay(
        self, component: object, options: OverlayOptions | None = None
    ) -> OverlayHandle:
        """Show a component as an overlay. Returns a handle to control it."""
        entry = OverlayEntry(
            component=component,
            options=options,
            pre_focus=self._focused_component,
            hidden=False,
            focus_order=self._next_focus_order,
        )
        self._overlay_stack.append(entry)

        # Only set focus if not non-capturing and overlay is visible
        is_non_capturing = options and options.non_capturing
        if not is_non_capturing and self._is_overlay_visible(entry):
            self.set_focus(component)
            self._focused_overlay = entry
            self._next_focus_order += 1

        self.terminal.hide_cursor()
        self.request_render()

        def _hide() -> None:
            try:
                self._overlay_stack.remove(entry)
            except ValueError:
                return
            if self._focused_component is component:
                top = self._get_topmost_visible_overlay()
                self.set_focus(top.component if top else entry.pre_focus)
            if self._focused_overlay == entry:
                self._focused_overlay = None
            if not self._overlay_stack:
                self.terminal.hide_cursor()
            self.request_render()

        def _set_hidden(hidden: bool) -> None:
            if entry.hidden == hidden:
                return
            entry.hidden = hidden
            if hidden:
                if self._focused_component is component:
                    top = self._get_topmost_visible_overlay()
                    self.set_focus(top.component if top else entry.pre_focus)
                if self._focused_overlay == entry:
                    self._focused_overlay = None
            else:
                is_non_capturing = options and options.non_capturing
                if not is_non_capturing and self._is_overlay_visible(entry):
                    self.set_focus(component)
                    self._focused_overlay = entry
                    entry.focus_order = self._next_focus_order
                    self._next_focus_order += 1
            self.request_render()

        def _is_hidden() -> bool:
            return entry.hidden

        return OverlayHandle(_hide, _set_hidden, _is_hidden, entry, self)

    def hide_overlay(self) -> None:
        """Hide the topmost overlay and restore previous focus."""
        if not self._overlay_stack:
            return
        overlay = self._overlay_stack.pop()
        top = self._get_topmost_visible_overlay()
        self.set_focus(top.component if top else overlay.pre_focus)
        if not self._overlay_stack:
            self.terminal.hide_cursor()
        self.request_render()

    def has_overlay(self) -> bool:
        return any(self._is_overlay_visible(o) for o in self._overlay_stack)

    def _is_overlay_visible(self, entry: OverlayEntry) -> bool:
        if entry.hidden:
            return False
        if entry.options and entry.options.visible:
            return entry.options.visible(self.terminal.columns, self.terminal.rows)
        return True

    def _get_topmost_visible_overlay(self) -> OverlayEntry | None:
        """Find the topmost visible capturing overlay, if any."""
        for i in range(len(self._overlay_stack) - 1, -1, -1):
            entry = self._overlay_stack[i]
            # Skip non-capturing overlays
            if entry.options and entry.options.non_capturing:
                continue
            if self._is_overlay_visible(entry):
                return entry
        return None

    def invalidate(self) -> None:
        super().invalidate()
        for overlay in self._overlay_stack:
            if hasattr(overlay.component, "invalidate"):
                overlay.component.invalidate()  # type: ignore

    def start(self) -> None:
        self._stopped = False
        # Store the running event loop so request_render() can safely
        # schedule renders from background threads via call_soon_threadsafe.
        import asyncio as _asyncio

        try:
            self._main_loop = _asyncio.get_running_loop()
        except RuntimeError:
            try:
                self._main_loop = _asyncio.get_event_loop()
            except RuntimeError:
                self._main_loop = None
        self.terminal.start(
            lambda data: self._handle_input(data),
            lambda: self.request_render(),
        )
        self.terminal.hide_cursor()
        self._query_cell_size()
        self.request_render()

    def add_input_listener(self, listener: InputListener) -> Callable[[], None]:
        self._input_listeners.append(listener)

        def _remove() -> None:
            try:
                self._input_listeners.remove(listener)
            except ValueError:
                pass

        return _remove

    def remove_input_listener(self, listener: InputListener) -> None:
        try:
            self._input_listeners.remove(listener)
        except ValueError:
            pass

    def _query_cell_size(self) -> None:
        if not get_capabilities().images:
            return
        self._cell_size_query_pending = True
        self.terminal.write("\x1b[16t")

    def stop(self) -> None:
        self._stopped = True
        # 取消挂起的节流渲染定时器，避免 stop 后仍触发 _do_render
        if self._render_timer is not None:
            self._render_timer.cancel()
            self._render_timer = None
        if self._previous_lines:
            target_row = len(self._previous_lines)
            line_diff = target_row - self._hardware_cursor_row
            if line_diff > 0:
                self.terminal.write(f"\x1b[{line_diff}B")
            elif line_diff < 0:
                self.terminal.write(f"\x1b[{-line_diff}A")
            self.terminal.write("\r\n")
        self.terminal.show_cursor()
        self.terminal.stop()

    # ─────────────────────────────────────────────────────────────────────────
    # AltScreen 接入（备用屏幕模式）
    # ─────────────────────────────────────────────────────────────────────────

    def enter_alt_screen(self) -> None:
        """进入备用屏幕模式：终端切换至备用屏幕缓冲区并隐藏光标。

        AltScreen 的完整渲染管线（差分渲染进备用缓冲区、鼠标选区/滚动、搜索）
        由独立渲染器 :class:`pi_tui.tui_alt_screen.TuiAltScreen` 提供
        （对标 TS TuiAltScreen，继承本类，其 start()/stop() 会自行完成进入/退出
        序列）；此方法仅供需要直接切换缓冲区或由外部组合使用。
        """
        self.terminal.write("\x1b[?1049h\x1b[?7l\x1b[2J\x1b[H\x1b[?25l")
        self._alt_screen_active = True

    def exit_alt_screen(self) -> None:
        """退出备用屏幕模式：恢复主屏幕缓冲区、自动换行与光标显示。"""
        self.terminal.write("\x1b[?7h\x1b[?1049l\x1b[?25h")
        self._alt_screen_active = False

    def request_render(self, force: bool = False) -> None:
        if force:
            self._previous_lines = []
            self._previous_width = -1
            self._previous_height = -1
            self._cursor_row = 0
            self._hardware_cursor_row = 0
            self._max_lines_rendered = 0
            self._previous_viewport_top = 0
        if self._render_requested:
            return
        self._render_requested = True
        # 节流调度：合并高频 request（流式 token 级事件），对齐 TS
        # MIN_RENDER_INTERVAL_MS = 16（60fps 上限），避免每 token 全屏 diff 重绘闪烁。
        import asyncio

        loop = self._main_loop
        if loop is not None and loop.is_running():
            try:
                running = asyncio.get_running_loop()
            except RuntimeError:
                running = None

            if running is loop:
                loop.call_soon(self._schedule_render)  # type: ignore[union-attr]
            else:
                loop.call_soon_threadsafe(self._schedule_render)  # type: ignore[union-attr]
            return
        # Fallback: try the current thread's loop
        try:
            cur = asyncio.get_running_loop()
            if cur.is_running():
                cur.call_soon(self._schedule_render)
                return
        except RuntimeError:
            pass
        # Synchronous fallback
        self._render_requested = False
        try:
            self._do_render()
        except Exception:
            logger.exception("Synchronous _do_render fallback failed")
            raise

    def _schedule_render(self) -> None:
        """按最小渲染间隔（16ms）调度 _render_tick，对齐 TS scheduleRender()。"""
        if (
            self._stopped
            or self._render_timer is not None
            or not self._render_requested
        ):
            return
        elapsed = time.monotonic() - self._last_render_at
        delay = max(0.0, self.MIN_RENDER_INTERVAL_MS / 1000.0 - elapsed)
        loop = self._main_loop
        if loop is not None and loop.is_running():
            self._render_timer = loop.call_later(delay, self._render_tick)  # type: ignore[union-attr]
        else:
            # 无运行中的事件循环：同步渲染
            self._render_requested = False
            self._last_render_at = time.monotonic()
            try:
                self._do_render()
            except Exception:
                logger.exception("Synchronous _do_render fallback failed")
                raise
            if self._render_requested:
                self._schedule_render()

    def _render_tick(self) -> None:
        self._render_timer = None
        if self._stopped or not self._render_requested:
            return
        self._render_requested = False
        self._last_render_at = time.monotonic()
        try:
            self._do_render()
        except Exception:
            logger.exception("_do_render raised an exception")
            # Re-raise so the event loop's exception handler also sees it
            raise
        if self._render_requested:
            self._schedule_render()

    def _handle_input(self, data: str) -> None:
        # 终端颜色查询响应消费（OSC 11 背景色 / DSR 996 配色方案），
        # 对齐 origin_pi tui.ts 的 handleTerminalInput 顺序。
        if self._consume_osc11_background_response(data):
            return
        if self._consume_terminal_color_scheme_report(data):
            return

        if self._input_listeners:
            current = data
            for listener in list(self._input_listeners):
                result = listener(current)
                if result and result.get("consume"):
                    return
                if result and result.get("data") is not None:
                    current = result["data"]
            if not current:
                return
            data = current

        # Cell size response buffering
        if self._cell_size_query_pending:
            self._input_buffer += data
            filtered = self._parse_cell_size_response()
            if not filtered:
                return
            data = filtered

        # Debug key
        if matches_key(data, "shift+ctrl+d") and self.on_debug:
            self.on_debug()
            return

        # Verify focused overlay is still visible
        focused_overlay = next(
            (o for o in self._overlay_stack if o.component is self._focused_component),
            None,
        )
        if focused_overlay and not self._is_overlay_visible(focused_overlay):
            top = self._get_topmost_visible_overlay()
            if top:
                self.set_focus(top.component)
            else:
                self.set_focus(focused_overlay.pre_focus)

        # Forward to focused component
        if self._focused_component and hasattr(self._focused_component, "handle_input"):
            wants_release = getattr(self._focused_component, "wants_key_release", False)
            if is_key_release(data) and not wants_release:
                return
            self._focused_component.handle_input(data)  # type: ignore
            self.request_render()

    def _parse_cell_size_response(self) -> str:
        import re

        pattern = re.compile(r"\x1b\[6;(\d+);(\d+)t")
        m = pattern.search(self._input_buffer)
        if m:
            h_px = int(m.group(1))
            w_px = int(m.group(2))
            if h_px > 0 and w_px > 0:
                set_cell_dimensions(CellDimensions(width_px=w_px, height_px=h_px))
                self.invalidate()
                self.request_render()
            self._input_buffer = (
                self._input_buffer[: m.start()] + self._input_buffer[m.end() :]
            )
            self._cell_size_query_pending = False

        partial = re.compile(r"\x1b(\[6?;?[\d;]*)?$")
        if partial.search(self._input_buffer):
            last = self._input_buffer[-1] if self._input_buffer else ""
            if not (last and last.isalpha() or last in ("~",)):
                return ""

        result = self._input_buffer
        self._input_buffer = ""
        self._cell_size_query_pending = False
        return result

    # ─────────────────────────────────────────────────────────────────────────
    # 终端颜色查询（OSC 11 背景色 / DSR 996 配色方案）— mirrors tui.ts
    # ─────────────────────────────────────────────────────────────────────────

    def _consume_osc11_background_response(self, data: str) -> bool:
        """消费 OSC 11 背景色响应并结算最旧的查询 future。"""
        if not self._pending_osc11_queries:
            return False
        if not is_osc11_background_color_response(data):
            return False
        rgb = parse_osc11_background_color(data)
        query = self._pending_osc11_queries.popleft()
        if not query.done():
            query.set_result(rgb)
        return True

    def _consume_terminal_color_scheme_report(self, data: str) -> bool:
        """消费 DSR 996 配色方案报告并广播给所有等待的监听器。"""
        scheme = parse_terminal_color_scheme_report(data)
        if scheme is None:
            return False
        for listener in list(self._terminal_color_scheme_listeners):
            if not listener.done():
                listener.set_result(scheme)
        return True

    async def query_terminal_background_color(self, timeout_ms: int) -> RgbColor | None:
        """查询终端默认背景色（OSC 11，``ESC ] 11 ; ? BEL``）。

        Args:
            timeout_ms: 查询超时毫秒数。

        Returns:
            解析后的 RGB 颜色；超时或解析失败返回 None。
        """
        fut: asyncio.Future[RgbColor | None] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_osc11_queries.append(fut)
        self.terminal.write("\x1b]11;?\x07")
        try:
            return await asyncio.wait_for(fut, timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            try:
                self._pending_osc11_queries.remove(fut)
            except ValueError:
                pass
            return None

    async def query_terminal_color_scheme(
        self, timeout_ms: int
    ) -> TerminalColorScheme | None:
        """查询终端配色方案偏好（DSR 996，``CSI ? 996 n``）。

        支持颜色调色板通知协议的终端会回复 ``CSI ? 997 ; 1 n``（dark）
        或 ``CSI ? 997 ; 2 n``（light）。

        Args:
            timeout_ms: 查询超时毫秒数。

        Returns:
            "dark" / "light"；超时或终端不支持时返回 None。
        """
        fut: asyncio.Future[TerminalColorScheme | None] = (
            asyncio.get_running_loop().create_future()
        )
        # 清理已完成的监听器，避免列表膨胀
        self._terminal_color_scheme_listeners = [
            f for f in self._terminal_color_scheme_listeners if not f.done()
        ]
        self._terminal_color_scheme_listeners.append(fut)
        self.terminal.write("\x1b[?996n")
        try:
            return await asyncio.wait_for(fut, timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            return None
        finally:
            try:
                self._terminal_color_scheme_listeners.remove(fut)
            except ValueError:
                pass

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay layout resolution — mirrors resolveOverlayLayout in tui.ts
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_overlay_layout(
        self,
        options: OverlayOptions | None,
        overlay_height: int,
        term_width: int,
        term_height: int,
    ) -> tuple[int, int, int, int | None]:
        """Returns (width, row, col, max_height)."""
        opt = options or OverlayOptions()

        if isinstance(opt.margin, int):
            m_top = m_right = m_bottom = m_left = max(0, opt.margin)
        elif isinstance(opt.margin, OverlayMargin):
            m_top = max(0, opt.margin.top)
            m_right = max(0, opt.margin.right)
            m_bottom = max(0, opt.margin.bottom)
            m_left = max(0, opt.margin.left)
        else:
            m_top = m_right = m_bottom = m_left = 0

        avail_w = max(1, term_width - m_left - m_right)
        avail_h = max(1, term_height - m_top - m_bottom)

        parsed_w = _parse_size_value(opt.width, term_width)
        width = parsed_w if parsed_w is not None else min(80, avail_w)
        if opt.min_width is not None:
            width = max(width, opt.min_width)
        width = max(1, min(width, avail_w))

        max_height = _parse_size_value(opt.max_height, term_height)
        if max_height is not None:
            max_height = max(1, min(max_height, avail_h))

        eff_h = (
            min(overlay_height, max_height)
            if max_height is not None
            else overlay_height
        )

        # Row
        if opt.row is not None:
            if isinstance(opt.row, str) and opt.row.endswith("%"):
                max_row = max(0, avail_h - eff_h)
                pct = float(opt.row[:-1]) / 100
                row = m_top + int(max_row * pct)
            elif isinstance(opt.row, int):
                row = opt.row
            else:
                row = self._anchor_row(opt.anchor, eff_h, avail_h, m_top)
        else:
            row = self._anchor_row(opt.anchor, eff_h, avail_h, m_top)

        # Col
        if opt.col is not None:
            if isinstance(opt.col, str) and opt.col.endswith("%"):
                max_col = max(0, avail_w - width)
                pct = float(opt.col[:-1]) / 100
                col = m_left + int(max_col * pct)
            elif isinstance(opt.col, int):
                col = opt.col
            else:
                col = self._anchor_col(opt.anchor, width, avail_w, m_left)
        else:
            col = self._anchor_col(opt.anchor, width, avail_w, m_left)

        row += opt.offset_y
        col += opt.offset_x

        row = max(m_top, min(row, term_height - m_bottom - eff_h))
        col = max(m_left, min(col, term_width - m_right - width))

        return width, row, col, max_height

    def _anchor_row(self, anchor: str, height: int, avail_h: int, m_top: int) -> int:
        if anchor in ("top-left", "top-center", "top-right"):
            return m_top
        if anchor in ("bottom-left", "bottom-center", "bottom-right"):
            return m_top + avail_h - height
        return m_top + (avail_h - height) // 2  # center / left-center / right-center

    def _anchor_col(self, anchor: str, width: int, avail_w: int, m_left: int) -> int:
        if anchor in ("top-left", "left-center", "bottom-left"):
            return m_left
        if anchor in ("top-right", "right-center", "bottom-right"):
            return m_left + avail_w - width
        return m_left + (avail_w - width) // 2  # center / top-center / bottom-center

    # ─────────────────────────────────────────────────────────────────────────
    # Overlay compositing — mirrors compositeOverlays in tui.ts
    # ─────────────────────────────────────────────────────────────────────────

    def _composite_overlays(
        self,
        lines: list[str],
        term_width: int,
        term_height: int,
    ) -> list[str]:
        if not self._overlay_stack:
            return lines
        result = list(lines)

        # Get visible overlays and sort by focus_order
        visible_entries = [
            e for e in self._overlay_stack if self._is_overlay_visible(e)
        ]
        visible_entries.sort(key=lambda e: e.focus_order)  # NEW: Sort by focus order

        rendered: list[
            tuple[list[str], int, int, int]
        ] = []  # (overlay_lines, row, col, w)
        min_lines_needed = len(result)

        for entry in visible_entries:
            w, _, _, max_h = self._resolve_overlay_layout(
                entry.options, 0, term_width, term_height
            )
            comp = entry.component
            overlay_lines: list[str] = comp.render(w) if hasattr(comp, "render") else []  # type: ignore
            if max_h is not None and len(overlay_lines) > max_h:
                overlay_lines = overlay_lines[:max_h]
            _, row, col, _ = self._resolve_overlay_layout(
                entry.options, len(overlay_lines), term_width, term_height
            )
            rendered.append((overlay_lines, row, col, w))
            min_lines_needed = max(min_lines_needed, row + len(overlay_lines))

        working_h = max(self._max_lines_rendered, min_lines_needed)
        while len(result) < working_h:
            result.append("")

        viewport_start = max(0, working_h - term_height)
        modified: set[int] = set()

        for overlay_lines, row, col, w in rendered:
            for i, ol in enumerate(overlay_lines):
                idx = viewport_start + row + i
                if 0 <= idx < len(result):
                    trunc = (
                        slice_by_column(ol, 0, w, True) if visible_width(ol) > w else ol
                    )
                    result[idx] = self._composite_line_at(
                        result[idx], trunc, col, w, term_width
                    )
                    modified.add(idx)

        for idx in modified:
            lw = visible_width(result[idx])
            if lw > term_width:
                result[idx] = slice_by_column(result[idx], 0, term_width, True)

        return result

    def _apply_line_resets(self, lines: list[str]) -> list[str]:
        for i, line in enumerate(lines):
            if not is_image_line(line):
                lines[i] = line + _SEGMENT_RESET
        return lines

    def _composite_line_at(
        self,
        base_line: str,
        overlay_line: str,
        start_col: int,
        overlay_width: int,
        total_width: int,
    ) -> str:
        if is_image_line(base_line):
            return base_line

        after_start = start_col + overlay_width
        base = extract_segments(
            base_line, start_col, after_start, total_width - after_start, True
        )
        overlay = slice_with_width(overlay_line, 0, overlay_width, True)

        before_pad = max(0, start_col - base.before_width)
        overlay_pad = max(0, overlay_width - overlay.width)
        actual_before_w = max(start_col, base.before_width)
        actual_overlay_w = max(overlay_width, overlay.width)
        after_target = max(0, total_width - actual_before_w - actual_overlay_w)
        after_pad = max(0, after_target - base.after_width)

        r = _SEGMENT_RESET
        result = (
            base.before
            + " " * before_pad
            + r
            + overlay.text
            + " " * overlay_pad
            + r
            + base.after
            + " " * after_pad
        )

        if visible_width(result) <= total_width:
            return result
        return slice_by_column(result, 0, total_width, True)

    # ─────────────────────────────────────────────────────────────────────────
    # Cursor marker extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_cursor_position(
        self, lines: list[str], height: int
    ) -> tuple[int, int] | None:
        """Find CURSOR_MARKER, strip it, return (row, col) or None."""
        viewport_top = max(0, len(lines) - height)
        for row in range(len(lines) - 1, viewport_top - 1, -1):
            line = lines[row]
            idx = line.find(CURSOR_MARKER)
            if idx != -1:
                before = line[:idx]
                col = visible_width(before)
                lines[row] = before + line[idx + len(CURSOR_MARKER) :]
                return row, col
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Main render loop — mirrors doRender() in tui.ts
    # ─────────────────────────────────────────────────────────────────────────

    def _do_render(self) -> None:
        if self._stopped:
            return

        width = self.terminal.columns
        height = self.terminal.rows

        viewport_top = max(0, self._max_lines_rendered - height)
        prev_viewport_top = self._previous_viewport_top
        hardware_cursor_row = self._hardware_cursor_row

        def compute_line_diff(target_row: int) -> int:
            cur_screen = hardware_cursor_row - prev_viewport_top
            tgt_screen = target_row - viewport_top
            return tgt_screen - cur_screen

        new_lines = self.render(width)

        if self._overlay_stack:
            new_lines = self._composite_overlays(new_lines, width, height)

        cursor_pos = self._extract_cursor_position(new_lines, height)
        new_lines = self._apply_line_resets(new_lines)

        width_changed = self._previous_width != 0 and self._previous_width != width
        height_changed = self._previous_height != 0 and self._previous_height != height

        debug_redraw = os.environ.get("PI_DEBUG_REDRAW") == "1"

        def log_redraw(reason: str) -> None:
            if not debug_redraw:
                return
            try:
                from datetime import datetime

                log_path = os.path.join(
                    os.path.expanduser("~"), ".pi", "agent", "pi-debug.log"
                )
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                msg = (
                    f"[{datetime.now().isoformat()}] fullRender: {reason} "
                    f"(prev={len(self._previous_lines)}, new={len(new_lines)}, height={height})\n"
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(msg)
            except Exception:
                pass

        def full_render(clear: bool) -> None:
            nonlocal hardware_cursor_row, viewport_top, prev_viewport_top
            self._full_redraw_count += 1
            buf = "\x1b[?2026h"
            if clear:
                buf += "\x1b[3J\x1b[2J\x1b[H"
            for i, ln in enumerate(new_lines):
                if i > 0:
                    buf += "\r\n"
                buf += ln
            buf += "\x1b[?2026l"
            self.terminal.write(buf)
            self._cursor_row = max(0, len(new_lines) - 1)
            self._hardware_cursor_row = self._cursor_row
            if clear:
                self._max_lines_rendered = len(new_lines)
            else:
                self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
            self._previous_viewport_top = max(0, self._max_lines_rendered - height)
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_width = width
            self._previous_height = height

        if not self._previous_lines and not width_changed and not height_changed:
            log_redraw("first render")
            full_render(False)
            return

        if width_changed or height_changed:
            reason = f"terminal size changed ({self._previous_width}x{self._previous_height} -> {width}x{height})"
            log_redraw(reason)
            full_render(True)
            return

        if (
            self._clear_on_shrink
            and len(new_lines) < self._max_lines_rendered
            and not self._overlay_stack
        ):
            log_redraw(f"clearOnShrink (maxLinesRendered={self._max_lines_rendered})")
            full_render(True)
            return

        first_changed = -1
        last_changed = -1
        max_l = max(len(new_lines), len(self._previous_lines))
        for i in range(max_l):
            old = self._previous_lines[i] if i < len(self._previous_lines) else ""
            new = new_lines[i] if i < len(new_lines) else ""
            if old != new:
                if first_changed == -1:
                    first_changed = i
                last_changed = i

        appended = len(new_lines) > len(self._previous_lines)
        if appended:
            if first_changed == -1:
                first_changed = len(self._previous_lines)
            last_changed = len(new_lines) - 1

        append_start = (
            appended
            and first_changed == len(self._previous_lines)
            and first_changed > 0
        )

        if first_changed == -1:
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_viewport_top = max(0, self._max_lines_rendered - height)
            return

        if first_changed >= len(new_lines):
            if len(self._previous_lines) > len(new_lines):
                buf = "\x1b[?2026h"
                target_row = max(0, len(new_lines) - 1)
                ld = compute_line_diff(target_row)
                if ld > 0:
                    buf += f"\x1b[{ld}B"
                elif ld < 0:
                    buf += f"\x1b[{-ld}A"
                buf += "\r"
                extra = len(self._previous_lines) - len(new_lines)
                if extra > height:
                    full_render(True)
                    return
                if extra > 0:
                    buf += "\x1b[1B"
                for i in range(extra):
                    buf += "\r\x1b[2K"
                    if i < extra - 1:
                        buf += "\x1b[1B"
                if extra > 0:
                    buf += f"\x1b[{extra}A"
                buf += "\x1b[?2026l"
                self.terminal.write(buf)
                self._cursor_row = target_row
                self._hardware_cursor_row = target_row
            self._position_hardware_cursor(cursor_pos, len(new_lines))
            self._previous_lines = new_lines
            self._previous_width = width
            self._previous_height = height
            self._previous_viewport_top = max(0, self._max_lines_rendered - height)
            return

        prev_content_viewport_top = max(0, len(self._previous_lines) - height)
        if first_changed < prev_content_viewport_top:
            full_render(True)
            return

        buf = "\x1b[?2026h"
        prev_viewport_bottom = prev_viewport_top + height - 1
        move_target_row = first_changed - 1 if append_start else first_changed

        if move_target_row > prev_viewport_bottom:
            cur_screen = max(
                0, min(height - 1, hardware_cursor_row - prev_viewport_top)
            )
            move_to_bottom = height - 1 - cur_screen
            if move_to_bottom > 0:
                buf += f"\x1b[{move_to_bottom}B"
            scroll = move_target_row - prev_viewport_bottom
            buf += "\r\n" * scroll
            prev_viewport_top += scroll
            viewport_top += scroll
            hardware_cursor_row = move_target_row

        ld = compute_line_diff(move_target_row)
        if ld > 0:
            buf += f"\x1b[{ld}B"
        elif ld < 0:
            buf += f"\x1b[{-ld}A"

        buf += "\r\n" if append_start else "\r"

        render_end = min(last_changed, len(new_lines) - 1)
        for i in range(first_changed, render_end + 1):
            if i > first_changed:
                buf += "\r\n"
            buf += "\x1b[2K"
            line = new_lines[i]
            if not is_image_line(line) and visible_width(line) > width:
                # Safety: truncate and log
                logger.warning(
                    "Line %d exceeds terminal width (%d > %d)",
                    i,
                    visible_width(line),
                    width,
                )
                line = slice_by_column(line, 0, width, True)
            buf += line

        final_cursor_row = render_end

        if len(self._previous_lines) > len(new_lines):
            if render_end < len(new_lines) - 1:
                move_down = len(new_lines) - 1 - render_end
                buf += f"\x1b[{move_down}B"
                final_cursor_row = len(new_lines) - 1
            extra = len(self._previous_lines) - len(new_lines)
            for _ in range(extra):
                buf += "\r\n\x1b[2K"
            buf += f"\x1b[{extra}A"

        buf += "\x1b[?2026l"
        self.terminal.write(buf)

        self._cursor_row = max(0, len(new_lines) - 1)
        self._hardware_cursor_row = final_cursor_row
        self._max_lines_rendered = max(self._max_lines_rendered, len(new_lines))
        self._previous_viewport_top = max(0, self._max_lines_rendered - height)

        self._position_hardware_cursor(cursor_pos, len(new_lines))
        self._previous_lines = new_lines
        self._previous_width = width
        self._previous_height = height

    # ─────────────────────────────────────────────────────────────────────────
    # Hardware cursor positioning
    # ─────────────────────────────────────────────────────────────────────────

    def _position_hardware_cursor(
        self,
        cursor_pos: tuple[int, int] | None,
        total_lines: int,
    ) -> None:
        if cursor_pos is None or total_lines <= 0:
            self.terminal.hide_cursor()
            return

        target_row = max(0, min(cursor_pos[0], total_lines - 1))
        target_col = max(0, cursor_pos[1])

        row_delta = target_row - self._hardware_cursor_row
        buf = ""
        if row_delta > 0:
            buf += f"\x1b[{row_delta}B"
        elif row_delta < 0:
            buf += f"\x1b[{-row_delta}A"
        buf += f"\x1b[{target_col + 1}G"

        if buf:
            self.terminal.write(buf)

        self._hardware_cursor_row = target_row

        if self._show_hardware_cursor:
            self.terminal.show_cursor()
        else:
            self.terminal.hide_cursor()
