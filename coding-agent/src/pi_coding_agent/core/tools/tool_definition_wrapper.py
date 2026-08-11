"""Tool definition wrapper utilities.

Provides functions to convert between ToolDefinition and AgentTool.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from pi_agent.types import AgentTool, AgentToolResult, AgentToolUpdateCallback, CancellationToken, ToolExecutionMode

TDetails = TypeVar("TDetails")


class ToolDefinition(BaseModel, Generic[TDetails]):
    """Tool definition (corresponds to TS ``ToolDefinition``).

    Captures the full definition of a tool including its parameters schema,
    execute function, and optional metadata.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    label: str
    description: str
    parameters: dict[str, object]
    execute: Callable[
        [str, dict[str, object], CancellationToken | None, AgentToolUpdateCallback | None],
        Awaitable[AgentToolResult],
    ]
    prepare_arguments: Callable[[dict[str, object]], dict[str, object]] | None = None
    execution_mode: ToolExecutionMode | None = None
    prompt_snippet: str | None = None
    """用于工具提示的代码片段。"""
    prompt_guidelines: list[str] | None = None
    """用于工具提示的指南列表。"""


def wrap_tool_definition(
    definition: ToolDefinition[TDetails],
) -> AgentTool:
    """Wrap a ToolDefinition into an AgentTool for the core runtime.

    Corresponds to TS ``wrapToolDefinition``.
    """
    return AgentTool(
        name=definition.name,
        label=definition.label,
        description=definition.description,
        parameters=definition.parameters,
        execute=definition.execute,
        prepare_arguments=definition.prepare_arguments,
        execution_mode=definition.execution_mode,
    )


def wrap_tool_definitions(
    definitions: list[ToolDefinition[object]],
) -> list[AgentTool]:
    """Wrap multiple ToolDefinitions into AgentTools.

    Corresponds to TS ``wrapToolDefinitions``.
    """
    return [wrap_tool_definition(d) for d in definitions]


def create_tool_definition_from_agent_tool(tool: AgentTool) -> ToolDefinition[object]:
    """Synthesize a minimal ToolDefinition from an AgentTool.

    Corresponds to TS ``createToolDefinitionFromAgentTool``.
    """
    return ToolDefinition(
        name=tool.name,
        label=tool.label,
        description=tool.description,
        parameters=tool.parameters,
        execute=tool.execute,
        prepare_arguments=tool.prepare_arguments,
        execution_mode=tool.execution_mode,
    )