"""Write tool - writes content to files.

Uses aiofiles for file writing. Creates parent directories automatically.
"""

from __future__ import annotations

import os

from pi_agent.types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CancellationToken,
    TextContent,
)
from pydantic import BaseModel, ConfigDict

from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_path
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class WriteToolInput(BaseModel):
    """Write tool input parameters."""

    path: str
    content: str


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class WriteOperations:
    """Pluggable operations for the write tool.

    Override these to delegate file writing to remote systems.

    """

    async def write_file(self, absolute_path: str, content: str) -> None:
        """Write content to a file."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="w", encoding="utf-8") as f:
            await f.write(content)

    async def mkdir(self, directory: str) -> None:
        """Create directory recursively."""
        os.makedirs(directory, exist_ok=True)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class WriteToolOptions(BaseModel):
    """Write tool options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: WriteOperations | None = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_write_tool_definition(
    cwd: str,
    options: WriteToolOptions | None = None,
) -> ToolDefinition[None]:
    """Create a write tool definition."""
    ops = (
        options.operations
        if options is not None and options.operations is not None
        else WriteOperations()
    )

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        path = str(params["path"])
        content = str(params["content"])

        absolute_path = resolve_path(path, cwd)
        directory = os.path.dirname(absolute_path)

        async def _write() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            # Create parent directories if needed
            await ops.mkdir(directory)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            # Write the file contents
            await ops.write_file(absolute_path, content)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully wrote {len(content)} bytes to {path}",
                    )
                ],
                details=None,
            )

        return await with_file_mutation_queue(absolute_path, _write)

    return ToolDefinition(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to write (relative or absolute)",
                },
                "content": {
                    "type": "string",
                    "description": "Content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
        execute=_execute,
    )


def create_write_tool(
    cwd: str,
    options: WriteToolOptions | None = None,
) -> AgentTool:
    """Create a write tool (AgentTool).
    """
    return wrap_tool_definition(create_write_tool_definition(cwd, options))
