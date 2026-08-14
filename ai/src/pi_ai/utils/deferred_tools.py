"""延迟工具加载。"""

from __future__ import annotations

from typing import Callable

from ..types import Context, Tool


def split_deferred_tools(
    context: Context,
    enabled: bool,
    normalize_name: Callable[[str], str] | None = None,
) -> tuple[list[Tool], dict[str, Tool]]:
    """将当前工具拆分为前缀和对话加载的定义。"""
    norm = normalize_name or (lambda name: name)
    unique_tools: dict[str, Tool] = {}
    for tool in context.tools or []:
        unique_tools[norm(tool.name)] = tool

    if not enabled:
        return list(unique_tools.values()), {}

    deferred_names: set[str] = set()
    used_names: set[str] = set()
    for message in context.messages:
        if message.role == "assistant":
            for block in message.content:
                if block.type == "toolCall":
                    used_names.add(norm(block.name))
        elif message.role == "toolResult":
            for name in message.added_tool_names or []:
                normalized_name = norm(name)
                if normalized_name not in used_names:
                    deferred_names.add(normalized_name)

    immediate: list[Tool] = []
    deferred: dict[str, Tool] = {}
    for name, tool in unique_tools.items():
        if name in deferred_names:
            deferred[name] = tool
        else:
            immediate.append(tool)

    return immediate, deferred