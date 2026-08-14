"""会话记录。

管理会话记录状态，包括快照、进度条目和工具调用缓冲区。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, TypeAlias

from pydantic import BaseModel

# ============================================================================
# Types
# ============================================================================

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
"""JSON 值类型。"""


class TranscriptState(BaseModel):
    """会话记录状态。

    管理会话快照和流式进度条目。
    """

    snapshot: dict[str, Any]
    """会话快照（SessionSnapshot）。"""
    progress_items: dict[str, Any]
    """进度条目映射（id -> TranscriptItem）。"""
    progress_order: list[str]
    """进度条目顺序。"""
    tool_call_buffers: dict[str, str]
    """工具调用缓冲区映射。"""

    model_config = {"arbitrary_types_allowed": True}


# ============================================================================
# Helper Functions
# ============================================================================


def _is_json_value(value: object) -> bool:
    """检查值是否为有效的 JSON 值。"""
    if value is None or isinstance(value, (bool, str)):
        return True
    if isinstance(value, (int, float)):
        import math

        return math.isfinite(value)
    if isinstance(value, list):
        return all(_is_json_value(item) for item in value)
    if isinstance(value, dict):
        return all(_is_json_value(k) and _is_json_value(v) for k, v in value.items())
    return False


def _parse_partial_tool_input(value: str) -> JsonValue:
    """解析部分工具输入。"""
    try:
        import json

        parsed: object = json.loads(value)
        if _is_json_value(parsed):
            return parsed  # type: ignore[return-value]
    except Exception:
        pass
    return value


# ============================================================================
# State Management Functions
# ============================================================================


def create_transcript_state(snapshot: dict[str, Any]) -> TranscriptState:
    """创建初始会话记录状态。"""
    return TranscriptState(
        snapshot=deepcopy(snapshot),
        progress_items={},
        progress_order=[],
        tool_call_buffers={},
    )


def apply_transcript_snapshot(
    state: TranscriptState, snapshot: dict[str, Any]
) -> TranscriptState:
    """应用会话快照更新。"""    
    if state.snapshot.get("id") == snapshot.get("id") and snapshot.get(
        "revision", -1
    ) < state.snapshot.get("revision", 0):
        return state
    return create_transcript_state(snapshot)


def _set_progress_item(state: TranscriptState, item: dict[str, Any]) -> TranscriptState:
    """设置进度条目。"""
    progress_items = dict(state.progress_items)
    progress_order = (
        list(state.progress_order)
        if item.get("id") in progress_items
        else [*state.progress_order, item["id"]]
    )
    progress_items[item["id"]] = deepcopy(item)
    return TranscriptState(
        snapshot=state.snapshot,
        progress_items=progress_items,
        progress_order=progress_order,
        tool_call_buffers=state.tool_call_buffers,
    )


def apply_transcript_progress(
    state: TranscriptState, progress: dict[str, Any]
) -> TranscriptState:
    """应用进度更新。"""
    progress_type = progress.get("type")

    if progress_type in ("item_started", "item_updated"):
        return _set_progress_item(state, progress["item"])

    if progress_type == "item_finished":
        tool_call_buffers = dict(state.tool_call_buffers)
        for key in list(tool_call_buffers.keys()):
            if key.startswith(f"{progress['item']['id']}:"):
                del tool_call_buffers[key]
        return _set_progress_item(
            TranscriptState(
                snapshot=state.snapshot,
                progress_items=state.progress_items,
                progress_order=state.progress_order,
                tool_call_buffers=tool_call_buffers,
            ),
            progress["item"],
        )

    message_id = progress.get("messageId")
    if not isinstance(message_id, str):
        return state
    item = state.progress_items.get(message_id)
    if item is None:
        for t in state.snapshot.get("transcript", []):
            if t.get("id") == message_id:
                item = t
                break
    if not item or item.get("role") != "assistant":
        return state

    tool_call_buffers = dict(state.tool_call_buffers)
    content_index = progress.get("contentIndex", 0)
    content = list(item.get("content", []))

    for idx, part in enumerate(content):
        if idx != content_index:
            continue
        if progress.get("kind") == "text" and part.get("type") == "text":
            content[idx] = {
                **part,
                "text": part.get("text", "") + progress.get("delta", ""),
            }
        elif progress.get("kind") == "thinking" and part.get("type") == "thinking":
            content[idx] = {
                **part,
                "thinking": part.get("thinking", "") + progress.get("delta", ""),
            }
        elif progress.get("kind") == "toolCall" and part.get("type") == "toolCall":
            key = f"{message_id}:{content_index}"
            existing = tool_call_buffers.get(key, "")
            if isinstance(part.get("input"), str):
                existing = part["input"]
            buffer = existing + progress.get("delta", "")
            tool_call_buffers = {**tool_call_buffers, key: buffer}
            content[idx] = {**part, "input": _parse_partial_tool_input(buffer)}

    return _set_progress_item(
        TranscriptState(
            snapshot=state.snapshot,
            progress_items=state.progress_items,
            progress_order=state.progress_order,
            tool_call_buffers=tool_call_buffers,
        ),
        {**item, "content": content},
    )


def select_transcript(
    state: TranscriptState,
) -> list[dict[str, Any]]:
    """选择最终转录条目。"""
    transcript: list[dict[str, Any]] = [
        state.progress_items.get(item["id"], item)
        for item in state.snapshot.get("transcript", [])
    ]
    ids = set(item["id"] for item in transcript)

    for item_id in state.progress_order:
        if item_id in ids:
            continue
        item = state.progress_items.get(item_id)
        if item:
            transcript.append(item)
            ids.add(item_id)

    for item in state.snapshot.get("queuedSteer", []):
        if item.get("id") in ids:
            continue
        transcript.append(item)
        ids.add(item["id"])

    return transcript


__all__ = [
    "JsonValue",
    "TranscriptState",
    "apply_transcript_progress",
    "apply_transcript_snapshot",
    "create_transcript_state",
    "select_transcript",
]
