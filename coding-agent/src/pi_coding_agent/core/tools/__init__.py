"""Tools system for the pi-coding-agent package.

Exports all tool types, factory functions, and utility modules.
Adapted from ``pi/packages/coding-agent/src/core/tools/index.ts``.
"""

from __future__ import annotations

from typing import Any, Literal

from .bash import (
    BashSpawnContext,
    BashSpawnHook,
    BashToolDetails,
    BashToolInput,
    BashToolOptions,
    create_bash_tool,
    create_bash_tool_definition,
)
from .edit import (
    EditToolDetails,
    EditToolInput,
    EditToolOptions,
    create_edit_tool,
    create_edit_tool_definition,
)
from .edit_diff import Edit, EditDiffError, EditDiffResult
from .file_mutation_queue import with_file_mutation_queue
from .find import (
    FindToolDetails,
    FindToolInput,
    FindToolOptions,
    create_find_tool,
    create_find_tool_definition,
)
from .grep import (
    GrepToolDetails,
    GrepToolInput,
    GrepToolOptions,
    create_grep_tool,
    create_grep_tool_definition,
)
from .ls import (
    LsToolDetails,
    LsToolInput,
    LsToolOptions,
    create_ls_tool,
    create_ls_tool_definition,
)
from .read import (
    ReadToolDetails,
    ReadToolInput,
    ReadToolOptions,
    create_read_tool,
    create_read_tool_definition,
)
from .tool_definition_wrapper import (
    ToolDefinition,
    create_tool_definition_from_agent_tool,
    wrap_tool_definition,
    wrap_tool_definitions,
)
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationOptions,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
    truncate_tail,
)
from .write import (
    WriteToolInput,
    WriteToolOptions,
    create_write_tool,
    create_write_tool_definition,
)

# ---------------------------------------------------------------------------
# Tool name type
# ---------------------------------------------------------------------------

ToolName = Literal["read", "bash", "edit", "write", "grep", "find", "ls"]
"""Type alias for tool names."""

ALL_TOOL_NAMES: set[ToolName] = {"read", "bash", "edit", "write", "grep", "find", "ls"}
"""Set of all available tool names."""


# ---------------------------------------------------------------------------
# Tool options
# ---------------------------------------------------------------------------


class ToolsOptions:
    """Options for all tools."""

    def __init__(
        self,
        read: ReadToolOptions | None = None,
        bash: BashToolOptions | None = None,
        write: WriteToolOptions | None = None,
        edit: EditToolOptions | None = None,
        grep: GrepToolOptions | None = None,
        find: FindToolOptions | None = None,
        ls: LsToolOptions | None = None,
    ) -> None:
        self.read = read
        self.bash = bash
        self.write = write
        self.edit = edit
        self.grep = grep
        self.find = find
        self.ls = ls


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_tool_definition(
    tool_name: ToolName, cwd: str, options: ToolsOptions | None = None
) -> ToolDefinition[Any]:
    """Create a tool definition by name."""
    tool_options = options or ToolsOptions()
    factories: dict[ToolName, ToolDefinition[Any]] = {
        "read": create_read_tool_definition(cwd, tool_options.read),
        "bash": create_bash_tool_definition(cwd, tool_options.bash),
        "edit": create_edit_tool_definition(cwd, tool_options.edit),
        "write": create_write_tool_definition(cwd, tool_options.write),
        "grep": create_grep_tool_definition(cwd, tool_options.grep),
        "find": create_find_tool_definition(cwd, tool_options.find),
        "ls": create_ls_tool_definition(cwd, tool_options.ls),
    }
    factory = factories.get(tool_name)
    if factory is None:
        raise ValueError(f"Unknown tool name: {tool_name}")
    return factory


def create_coding_tool_definitions(
    cwd: str, options: ToolsOptions | None = None
) -> list[ToolDefinition[Any]]:
    """Create coding tool definitions (read, bash, edit, write).""" 
    tool_options = options or ToolsOptions()
    return [
        create_read_tool_definition(cwd, tool_options.read),
        create_bash_tool_definition(cwd, tool_options.bash),
        create_edit_tool_definition(cwd, tool_options.edit),
        create_write_tool_definition(cwd, tool_options.write),
    ]


def create_read_only_tool_definitions(
    cwd: str, options: ToolsOptions | None = None
) -> list[ToolDefinition[Any]]:
    """Create read-only tool definitions (read, grep, find, ls)."""
    tool_options = options or ToolsOptions()
    return [
        create_read_tool_definition(cwd, tool_options.read),
        create_grep_tool_definition(cwd, tool_options.grep),
        create_find_tool_definition(cwd, tool_options.find),
        create_ls_tool_definition(cwd, tool_options.ls),
    ]


def create_all_tool_definitions(
    cwd: str, options: ToolsOptions | None = None
) -> dict[ToolName, ToolDefinition[Any]]:
    """Create all tool definitions as a dict keyed by tool name."""
    tool_options = options or ToolsOptions()
    return {
        "read": create_read_tool_definition(cwd, tool_options.read),
        "bash": create_bash_tool_definition(cwd, tool_options.bash),
        "edit": create_edit_tool_definition(cwd, tool_options.edit),
        "write": create_write_tool_definition(cwd, tool_options.write),
        "grep": create_grep_tool_definition(cwd, tool_options.grep),
        "find": create_find_tool_definition(cwd, tool_options.find),
        "ls": create_ls_tool_definition(cwd, tool_options.ls),
    }


# Import AgentTool for type hints
from pi_agent.types import AgentTool


def create_coding_tools(
    cwd: str, options: ToolsOptions | None = None
) -> list[AgentTool]:
    """Create coding tools (read, bash, edit, write)."""
    from .tool_definition_wrapper import wrap_tool_definition

    tool_options = options or ToolsOptions()
    return [
        wrap_tool_definition(create_read_tool_definition(cwd, tool_options.read)),
        wrap_tool_definition(create_bash_tool_definition(cwd, tool_options.bash)),
        wrap_tool_definition(create_edit_tool_definition(cwd, tool_options.edit)),
        wrap_tool_definition(create_write_tool_definition(cwd, tool_options.write)),
    ]


def create_read_only_tools(
    cwd: str, options: ToolsOptions | None = None
) -> list[AgentTool]:
    """Create read-only tools (read, grep, find, ls)."""
    from .tool_definition_wrapper import wrap_tool_definition

    tool_options = options or ToolsOptions()
    return [
        wrap_tool_definition(create_read_tool_definition(cwd, tool_options.read)),
        wrap_tool_definition(create_grep_tool_definition(cwd, tool_options.grep)),
        wrap_tool_definition(create_find_tool_definition(cwd, tool_options.find)),
        wrap_tool_definition(create_ls_tool_definition(cwd, tool_options.ls)),
    ]


def create_all_tools(
    cwd: str, options: ToolsOptions | None = None
) -> dict[ToolName, AgentTool]:
    """Create all tools as a dict keyed by tool name."""
    from .tool_definition_wrapper import wrap_tool_definition

    tool_options = options or ToolsOptions()
    return {
        "read": wrap_tool_definition(
            create_read_tool_definition(cwd, tool_options.read)
        ),
        "bash": wrap_tool_definition(
            create_bash_tool_definition(cwd, tool_options.bash)
        ),
        "edit": wrap_tool_definition(
            create_edit_tool_definition(cwd, tool_options.edit)
        ),
        "write": wrap_tool_definition(
            create_write_tool_definition(cwd, tool_options.write)
        ),
        "grep": wrap_tool_definition(
            create_grep_tool_definition(cwd, tool_options.grep)
        ),
        "find": wrap_tool_definition(
            create_find_tool_definition(cwd, tool_options.find)
        ),
        "ls": wrap_tool_definition(create_ls_tool_definition(cwd, tool_options.ls)),
    }


__all__ = [
    # Tool names
    "ToolName",
    "ALL_TOOL_NAMES",
    # Options
    "ToolsOptions",
    # Read
    "ReadToolInput",
    "ReadToolDetails",
    "ReadToolOptions",
    "create_read_tool",
    "create_read_tool_definition",
    # Bash
    "BashToolInput",
    "BashToolDetails",
    "BashToolOptions",
    "BashSpawnContext",
    "BashSpawnHook",
    "create_bash_tool",
    "create_bash_tool_definition",
    # Edit
    "EditToolInput",
    "EditToolDetails",
    "EditToolOptions",
    "Edit",
    "EditDiffResult",
    "EditDiffError",
    "create_edit_tool",
    "create_edit_tool_definition",
    # Write
    "WriteToolInput",
    "WriteToolOptions",
    "create_write_tool",
    "create_write_tool_definition",
    # Grep
    "GrepToolInput",
    "GrepToolDetails",
    "GrepToolOptions",
    "create_grep_tool",
    "create_grep_tool_definition",
    # Find
    "FindToolInput",
    "FindToolDetails",
    "FindToolOptions",
    "create_find_tool",
    "create_find_tool_definition",
    # Ls
    "LsToolInput",
    "LsToolDetails",
    "LsToolOptions",
    "create_ls_tool",
    "create_ls_tool_definition",
    # Truncation
    "DEFAULT_MAX_LINES",
    "DEFAULT_MAX_BYTES",
    "TruncationOptions",
    "TruncationResult",
    "format_size",
    "truncate_head",
    "truncate_tail",
    "truncate_line",
    # Tool definition wrapper
    "ToolDefinition",
    "wrap_tool_definition",
    "wrap_tool_definitions",
    "create_tool_definition_from_agent_tool",
    # File mutation queue
    "with_file_mutation_queue",
    # Factory functions
    "create_tool_definition",
    "create_coding_tool_definitions",
    "create_read_only_tool_definitions",
    "create_all_tool_definitions",
    "create_coding_tools",
    "create_read_only_tools",
    "create_all_tools",
]
