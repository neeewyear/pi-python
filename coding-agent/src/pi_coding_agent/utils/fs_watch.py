"""文件系统监控工具（对应 TS ``utils/fs_watch.ts``）。

提供基于 ``asyncio`` 轮询的文件/目录变化监控。
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

FS_WATCH_RETRY_DELAY_MS: int = 5000
"""文件监控重试延迟（毫秒）。"""

_DEFAULT_POLL_INTERVAL: float = 1.0
"""默认轮询间隔（秒）。"""

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

WatchCallback = Callable[[str, str | None], Any]
"""监控回调签名：``(event_type, filename) -> None``。"""

ErrorHandler = Callable[[], Any]
"""错误处理回调签名：``() -> None``。"""


# ---------------------------------------------------------------------------
# 监控器实现
# ---------------------------------------------------------------------------


@dataclass
class _FileWatcher:
    """文件监控器内部状态。"""

    path: str
    """被监控的文件或目录路径。"""
    callback: WatchCallback
    """变化回调。"""
    error_handler: ErrorHandler
    """错误处理回调。"""
    poll_interval: float = _DEFAULT_POLL_INTERVAL
    """轮询间隔（秒）。"""
    _task: asyncio.Task[None] | None = field(default=None, init=False, repr=False)
    """后台轮询任务。"""
    _last_mtime: float = field(default=0.0, init=False, repr=False)
    """上次检查时的修改时间。"""
    _closed: bool = field(default=False, init=False, repr=False)
    """是否已关闭。"""

    async def _poll_loop(self) -> None:
        """后台轮询循环。"""
        try:
            # 初始检查
            if os.path.isfile(self.path):
                try:
                    self._last_mtime = os.path.getmtime(self.path)
                except OSError:
                    pass
            elif os.path.isdir(self.path):
                self._last_mtime = time.time()

            while not self._closed:
                await asyncio.sleep(self.poll_interval)
                if self._closed:
                    break

                try:
                    if os.path.isfile(self.path):
                        new_mtime = os.path.getmtime(self.path)
                        if new_mtime != self._last_mtime:
                            self._last_mtime = new_mtime
                            self.callback("modified", os.path.basename(self.path))
                    elif os.path.isdir(self.path):
                        self._poll_directory()
                except OSError:
                    self.error_handler()
                    return
        except asyncio.CancelledError:
            pass
        except Exception:
            self.error_handler()

    def _poll_directory(self) -> None:
        """轮询目录变化。"""
        try:
            entries = os.listdir(self.path)
        except OSError:
            self.error_handler()
            return

        for entry in entries:
            entry_path = os.path.join(self.path, entry)
            try:
                new_mtime = os.path.getmtime(entry_path)
            except OSError:
                continue
            # 简单检查：如果文件被修改则触发回调
            # 注意：此实现不跟踪每个文件的 mtime，仅用于触发刷新
            self.callback("modified", entry)

    def start(self) -> None:
        """启动后台轮询任务。"""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._poll_loop())

    def close(self) -> None:
        """关闭监控器。"""
        self._closed = True
        if self._task is not None:
            self._task.cancel()
            self._task = None


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def watch_with_error_handler(
    path: str,
    callback: WatchCallback,
    error_handler: ErrorHandler,
    poll_interval: float | None = None,
) -> _FileWatcher | None:
    """监控文件或目录变化，出错时调用 error_handler。

    Args:
        path: 要监控的文件或目录路径。
        callback: 变化回调 ``(event_type, filename)``。
        error_handler: 错误处理回调。
        poll_interval: 轮询间隔（秒），默认 1.0。

    Returns:
        ``_FileWatcher`` 实例，或 ``None``（路径无效时）。
    """
    if not os.path.exists(path):
        return None

    watcher = _FileWatcher(
        path=path,
        callback=callback,
        error_handler=error_handler,
        poll_interval=poll_interval or _DEFAULT_POLL_INTERVAL,
    )
    watcher.start()
    return watcher


def close_watcher(watcher: Any) -> None:
    """关闭文件监控器。

    Args:
        watcher: ``watch_with_error_handler`` 返回的监控器实例。
    """
    if watcher is None:
        return
    if hasattr(watcher, "close"):
        watcher.close()
