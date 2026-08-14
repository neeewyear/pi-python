"""会话资源管理。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

SessionResourceCleanup = Callable[[str | None], None]
"""会话资源清理函数类型。"""

_session_resource_cleanups: set[SessionResourceCleanup] = set()


def register_session_resource_cleanup(cleanup: SessionResourceCleanup) -> Callable[[], None]:
    """注册会话资源清理函数。返回取消注册函数。"""
    _session_resource_cleanups.add(cleanup)
    return lambda: _session_resource_cleanups.discard(cleanup)


def cleanup_session_resources(session_id: str | None = None) -> None:
    """清理所有已注册的会话资源。"""
    errors: list[Any] = []
    for cleanup in _session_resource_cleanups:
        try:
            cleanup(session_id)
        except Exception as error:
            errors.append(error)
    if errors:
        msg = "; ".join(str(e) for e in errors)
        raise ExceptionGroup("Failed to cleanup session resources", errors)