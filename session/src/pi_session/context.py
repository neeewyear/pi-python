"""会话 → 模型上下文消息。

从路径条目推导模型上下文：压缩边界、分支摘要、自定义条目投影、
以及 model / thinking / active tools 状态推导。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeAlias

from pydantic import BaseModel

from pi_agent.harness.messages import (
    create_branch_summary_message,
    create_compaction_summary_message,
)
from pi_agent.types import AgentMessage, AssistantMessage
from .types import (
    ActiveToolsEntry,
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    Entry,
    MessageEntry,
    ModelChangeEntry,
    ThinkingLevelEntry,
)


class SessionContext(BaseModel):
    """会话上下文。"""

    messages: list[AgentMessage]
    thinking_level: str
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None


ContextEntryTransform: TypeAlias = Callable[[list[Entry]], list[Entry]]
"""条目变换函数：``(entries) -> entries``。"""

CustomEntryContextMessageProjector: TypeAlias = Callable[
    [CustomEntry, int, list[Entry]], list[AgentMessage] | None
]
"""自定义条目投影函数：``(entry, index, entries) -> AgentMessage[] | None``。"""


class SessionContextBuildOptions(BaseModel):
    """构建会话上下文的选项。"""

    entry_transforms: list[ContextEntryTransform] | None = None
    entry_projectors: dict[str, CustomEntryContextMessageProjector] | None = None


def _derive_session_context_state(path_entries: list[Entry]) -> dict[str, object]:
    """从路径条目推导 thinking_level / model / active_tool_names。"""
    thinking_level = "off"
    model: dict[str, str] | None = None
    active_tool_names: list[str] | None = None
    for entry in path_entries:
        if isinstance(entry, ThinkingLevelEntry):
            thinking_level = entry.thinking_level
        elif isinstance(entry, ModelChangeEntry):
            model = {"provider": entry.provider, "modelId": entry.model_id}
        elif isinstance(entry, MessageEntry) and isinstance(
            entry.message, AssistantMessage
        ):
            model = {"provider": entry.message.provider, "modelId": entry.message.model}
        elif isinstance(entry, ActiveToolsEntry):
            active_tool_names = list(entry.active_tool_names)
    return {
        "thinking_level": thinking_level,
        "model": model,
        "active_tool_names": active_tool_names,
    }


def default_context_entry_transform(path_entries: list[Entry]) -> list[Entry]:
    """默认变换：保留最近一次压缩条目及其之后的内容，丢弃更早的历史。"""
    compaction_index = -1
    for index in range(len(path_entries) - 1, -1, -1):
        if isinstance(path_entries[index], CompactionEntry):
            compaction_index = index
            break
    if compaction_index == -1:
        return list(path_entries)
    return [path_entries[compaction_index], *path_entries[compaction_index + 1 :]]


def build_context_entries(
    path_entries: list[Entry], options: SessionContextBuildOptions | None = None
) -> list[Entry]:
    """按顺序应用条目变换。"""
    options = options or SessionContextBuildOptions()
    entries = default_context_entry_transform(path_entries)
    for transform in options.entry_transforms or []:
        entries = list(transform(entries))
    return entries


def session_entry_to_context_messages(
    entry: Entry,
    index: int,
    entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> list[AgentMessage]:
    """把单条会话条目转为上下文消息。"""
    options = options or SessionContextBuildOptions()
    if isinstance(entry, MessageEntry):
        if (
            isinstance(entry.message, AssistantMessage)
            and entry.message.stop_reason == "deferred"
        ):
            return []
        return [entry.message]
    if isinstance(entry, CompactionEntry):
        return [
            create_compaction_summary_message(
                entry.summary, entry.tokens_before, entry.timestamp
            ),
            *entry.retained_tail,
        ]
    if isinstance(entry, BranchSummaryEntry) and entry.summary:
        return [
            create_branch_summary_message(entry.summary, entry.from_id, entry.timestamp)
        ]
    if isinstance(entry, CustomEntry):
        projector = (options.entry_projectors or {}).get(entry.custom_type)
        if projector is None:
            return []
        projected = projector(entry, index, entries)
        return list(projected) if projected is not None else []
    return []


def build_session_context(
    path_entries: list[Entry],
    options: SessionContextBuildOptions | None = None,
) -> SessionContext:
    """构建会话上下文。"""
    options = options or SessionContextBuildOptions()
    state = _derive_session_context_state(path_entries)
    context_entries = build_context_entries(path_entries, options)
    messages: list[AgentMessage] = []
    for index, entry in enumerate(context_entries):
        messages.extend(
            session_entry_to_context_messages(entry, index, context_entries, options)
        )
    return SessionContext(
        messages=messages,
        thinking_level=str(state["thinking_level"]),
        model=state["model"] if isinstance(state["model"], dict) else None,
        active_tool_names=state["active_tool_names"]
        if isinstance(state["active_tool_names"], list)
        else None,
    )
