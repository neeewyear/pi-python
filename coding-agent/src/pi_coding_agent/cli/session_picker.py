"""会话选择器。

非 TUI 版本：使用 stdin 文本列表选择会话。
由于 TUI 组件尚未移植到 Python，此版本提供简单的文本选择界面。
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Coroutine
from typing import Any

# 会话加载器类型签名
SessionsLoader = Callable[..., Coroutine[Any, Any, list[Any]]]
"""会话加载器，返回 ``SessionInfo`` 列表的异步函数。"""


async def pick_session(
    current_sessions_loader: SessionsLoader | None = None,
    all_sessions_loader: SessionsLoader | None = None,
    settings_manager: Any = None,
) -> str | None:
    """交互式会话选择（非 TUI 版本）。

    使用 stdin 文本列表让用户选择会话。
    完整 TUI 实现需要 ``@earendil-works/pi-tui`` 支持。

    Args:
        current_sessions_loader: 当前会话加载器。
        all_sessions_loader: 所有会话加载器。
        settings_manager: 设置管理器。

    Returns:
        选择的会话路径，取消返回 None。
    """
    # 优先使用 all_sessions_loader，回退到 current_sessions_loader
    loader = all_sessions_loader or current_sessions_loader
    if loader is None:
        print("No session loader available.", file=sys.stderr)
        return None

    sessions = await loader()
    if not sessions:
        print("No sessions found.", file=sys.stderr)
        return None

    # 打印会话列表
    print(file=sys.stderr)
    print("Available sessions:", file=sys.stderr)
    print(file=sys.stderr)

    for i, session in enumerate(sessions, start=1):
        # 尝试获取会话名称和路径
        name = getattr(session, "name", None) or getattr(session, "id", f"session-{i}")
        path = getattr(session, "path", "")
        created = getattr(session, "created", "")
        msg_count = getattr(session, "message_count", 0)
        first_msg = getattr(session, "first_message", "")

        # 显示前 60 个字符的第一条消息
        preview = first_msg[:60] + "..." if len(first_msg) > 60 else first_msg
        print(f"  {i}. {name}", file=sys.stderr)
        if path:
            print(f"     Path: {path}", file=sys.stderr)
        if msg_count:
            print(f"     Messages: {msg_count} | Created: {created}", file=sys.stderr)
        if preview:
            print(f"     Preview: {preview}", file=sys.stderr)
        print(file=sys.stderr)

    print("  q. Cancel", file=sys.stderr)
    print(file=sys.stderr)

    # 非交互模式：返回第一个会话
    if not sys.stdin.isatty():
        print("(non-interactive mode, selecting first session)", file=sys.stderr)
        return getattr(sessions[0], "path", None)

    # 交互模式：让用户选择
    while True:
        choice = (
            input(f"Select session [1-{len(sessions)}] (default: 1): ").strip().lower()
        )
        if choice == "q" or choice == "":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(sessions):
                return getattr(sessions[idx], "path", None)
        except ValueError:
            pass
        print(
            f"Invalid choice. Enter 1-{len(sessions)} or 'q' to cancel.",
            file=sys.stderr,
        )


__all__ = [
    "pick_session",
]
