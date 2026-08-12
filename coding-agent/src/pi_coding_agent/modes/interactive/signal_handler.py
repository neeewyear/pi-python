"""信号处理模块（SIGINT/SIGTERM/SIGHUP 优雅关闭）。

使用 ``asyncio`` 事件循环的 ``add_signal_handler`` 注册信号处理器，
确保回调在事件循环中安全执行，而非在信号中断上下文中。

Textual 的 ``app.run_async()`` 使用 asyncio 事件循环运行，
``loop.add_signal_handler`` 与之兼容，不会像 ``signal.signal`` 那样
被 Textual 内部覆盖。
"""

from __future__ import annotations

import asyncio
import signal
import sys
from collections.abc import Awaitable, Callable


def register_signal_handlers(
    dispose_runtime: Callable[[], Awaitable[None]],
    app_exit: Callable[[], None] | None = None,
) -> list[Callable[[], None]]:
    """注册信号处理器，用于优雅关闭。

    使用 ``loop.add_signal_handler`` 在事件循环中安全地处理信号。
    信号处理器**只负责退出应用**，资源清理由 ``finally`` 块负责。
    不在信号处理器中调度 ``dispose_runtime``，避免干扰 Textual 的
    终端状态清理流程。

    Args:
        dispose_runtime: 异步清理函数（仅在 ``finally`` 块中使用）。
        app_exit: 同步退出函数（如 Textual 的 ``app.exit()``）。

    Returns:
        清理处理器列表，用于恢复原始信号处理。
    """
    _ = dispose_runtime  # 仅在 finally 块中使用
    cleanup_handlers: list[Callable[[], None]] = []
    signals: list[int] = [signal.SIGTERM, signal.SIGINT]
    if sys.platform != "win32":
        signals.append(signal.SIGHUP)

    loop = asyncio.get_event_loop()

    for sig in signals:

        def callback(sig_value: int = sig) -> None:
            # 只退出应用 -> Textual 清理终端 -> 事件循环退出 -> finally 块清理
            if app_exit is not None:
                app_exit()

        try:
            loop.add_signal_handler(sig, callback)
        except NotImplementedError:
            # Windows 不支持 add_signal_handler，回退到 signal.signal
            original_handler = signal.getsignal(sig)

            def handler(
                signum: int = sig,
                _frame: object = None,
            ) -> None:
                if app_exit is not None:
                    app_exit()

            signal.signal(sig, handler)

            def cleanup(
                sig_value: int = sig,
                original_handler: object = original_handler,
            ) -> None:
                signal.signal(sig_value, original_handler)  # type: ignore[arg-type]

            cleanup_handlers.append(cleanup)
        else:

            def cleanup(sig_value: int = sig, original_handler: object = None) -> None:
                try:
                    loop.remove_signal_handler(sig_value)
                except (ValueError, NotImplementedError):
                    pass

            cleanup_handlers.append(cleanup)

    return cleanup_handlers


def restore_signal_handlers(
    cleanup_handlers: list[Callable[[], None]],
) -> None:
    """恢复原始信号处理器。

    Args:
        cleanup_handlers: ``register_signal_handlers`` 返回的清理处理器列表。
    """
    for cleanup in cleanup_handlers:
        try:
            cleanup()
        except Exception:
            pass
