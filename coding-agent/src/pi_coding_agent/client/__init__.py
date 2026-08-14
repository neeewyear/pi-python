"""客户端模块。

导出远程会话和会话记录相关类型与函数。
"""

from __future__ import annotations

from .remote_session import (
    CreateRemoteSessionOptions,
    RemoteSession,
    RemoteSessionLifecycle,
    RemoteSessionOperation,
    RemoteSessionOptions,
    RemoteSessionState,
)
from .transcript import (
    JsonValue,
    TranscriptState,
    apply_transcript_progress,
    apply_transcript_snapshot,
    create_transcript_state,
    select_transcript,
)

__all__ = [
    # remote_session
    "CreateRemoteSessionOptions",
    "RemoteSession",
    "RemoteSessionLifecycle",
    "RemoteSessionOperation",
    "RemoteSessionOptions",
    "RemoteSessionState",
    # transcript
    "JsonValue",
    "TranscriptState",
    "apply_transcript_progress",
    "apply_transcript_snapshot",
    "create_transcript_state",
    "select_transcript",
]