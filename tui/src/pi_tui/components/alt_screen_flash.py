"""
备用屏幕闪烁消息 — mirrors packages/tui/src/components/alt-screen-flash.ts

AltScreenFlashContainer: 由备用屏幕渲染器合成的瞬时消息堆栈。
消息默认持续 1000ms 后自动移除并触发重新渲染，渲染为反显（inverse video）行。
"""

from __future__ import annotations

import threading
from collections.abc import Callable

from ..utils import truncate_to_width

DEFAULT_DURATION_MS = 1000


class FlashEntry:
    """单条闪烁消息及其过期定时器。"""

    __slots__ = ("id", "message", "timer")

    def __init__(self, entry_id: int, message: str, timer: threading.Timer) -> None:
        self.id = entry_id
        self.message = message
        self.timer = timer


class AltScreenFlashContainer:
    """瞬时消息堆栈（对标 TS AltScreenFlashContainer）。"""

    def __init__(self, request_render: Callable[[], None]) -> None:
        self._request_render = request_render
        self._entries: list[FlashEntry] = []
        self._next_id = 0
        self._lock = threading.Lock()

    def flash(self, message: str, duration_ms: int | None = None) -> None:
        """显示一条消息，duration_ms 后自动移除。

        Args:
            message: 消息文本。
            duration_ms: 持续毫秒数，缺省 1000ms。
        """
        duration = max(
            0, duration_ms if duration_ms is not None else DEFAULT_DURATION_MS
        )
        entry_id = self._next_id
        self._next_id += 1
        timer = threading.Timer(duration / 1000.0, self._expire, args=[entry_id])
        timer.daemon = True  # 对齐 TS timer.unref()：不阻止进程退出
        timer.start()
        with self._lock:
            self._entries.append(FlashEntry(entry_id, message, timer))
        self._request_render()

    def _expire(self, entry_id: int) -> None:
        removed = False
        with self._lock:
            for index, entry in enumerate(self._entries):
                if entry.id == entry_id:
                    del self._entries[index]
                    removed = True
                    break
        if removed:
            self._request_render()

    def dispose(self) -> None:
        """取消所有定时器并清空消息。"""
        with self._lock:
            for entry in self._entries:
                entry.timer.cancel()
            self._entries.clear()

    def invalidate(self) -> None:
        pass

    def handle_input(self, _data: str) -> None:
        pass

    def render(self, width: int) -> list[str]:
        """将每条消息渲染为反显行，超宽按 width 截断。"""
        with self._lock:
            entries = list(self._entries)
        return [
            f"\x1b[7m{truncate_to_width(f' {entry.message} ', width, '')}\x1b[27m"
            for entry in entries
        ]
