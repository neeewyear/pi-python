"""工具包装器。

这些包装器仅适配工具执行，使扩展工具接收运行器上下文。
工具调用和工具结果拦截由 AgentSession 通过 agent-core 钩子处理。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_agent.types import (
        AgentTool,
        AgentToolResult,
        CancellationToken,
    )

    from .runner import ExtensionRunner
    from .types import RegisteredTool


def wrap_registered_tool(
    registered_tool: RegisteredTool, runner: ExtensionRunner
) -> AgentTool:
    """将 RegisteredTool 包装为 AgentTool。

    使用运行器的 create_context() 在工具和事件处理器之间提供一致的上下文。

    Args:
        registered_tool: 已注册的工具。
        runner: 扩展运行器。

    Returns:
        包装后的 AgentTool。
    """
    from pi_agent.types import AgentTool

    from ..tools.tool_definition_wrapper import (
        create_tool_definition_from_agent_tool,
        wrap_tool_definition,
    )

    definition = registered_tool.definition
    # 创建基础的 AgentTool（不包含扩展上下文）
    base_tool = wrap_tool_definition(
        create_tool_definition_from_agent_tool(
            AgentTool(
                name=definition.name,
                label=definition.label,
                description=definition.description,
                parameters=definition.parameters,
                execute=definition.execute,  # type: ignore[arg-type]
                prepare_arguments=definition.prepare_arguments,
                execution_mode=definition.execution_mode,
            )
        )
    )

    original_execute = base_tool.execute

    async def execute_with_tracking(
        tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None = None,
        on_update: Callable[[AgentToolResult], None] | None = None,
    ) -> AgentToolResult:
        active_before = runner.get_active_tools()
        # Use the callback directly as AgentToolUpdateCallback
        result = await original_execute(tool_call_id, params, signal, on_update)
        active_after = runner.get_active_tools()

        # 如果活跃工具列表被减少了，直接返回结果
        if not all(name in active_after for name in active_before):
            return result

        before_names = set(active_before)
        added_tool_names = [name for name in active_after if name not in before_names]
        if not added_tool_names:
            return result

        existing_added = (
            set(result.added_tool_names) if result.added_tool_names else set()
        )
        all_added = list(existing_added | set(added_tool_names))

        return AgentToolResult(
            content=result.content,
            details=result.details,
            usage=result.usage,
            added_tool_names=all_added,
        )

    return AgentTool(
        name=base_tool.name,
        label=base_tool.label,
        description=base_tool.description,
        parameters=base_tool.parameters,
        execute=execute_with_tracking,
        prepare_arguments=base_tool.prepare_arguments,
        execution_mode=base_tool.execution_mode,
    )


def wrap_registered_tools(
    registered_tools: list[RegisteredTool],
    runner: ExtensionRunner,
) -> list[AgentTool]:
    """将所有已注册工具包装为 AgentTools。

    使用运行器的 create_context() 在工具和事件处理器之间提供一致的上下文。

    Args:
        registered_tools: 已注册的工具列表。
        runner: 扩展运行器。

    Returns:
        包装后的 AgentTool 列表。
    """
    return [wrap_registered_tool(tool, runner) for tool in registered_tools]
