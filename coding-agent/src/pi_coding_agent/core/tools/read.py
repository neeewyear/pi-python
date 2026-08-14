"""Read tool - reads file contents.

Supports text files and images. Uses aiofiles for async file reading.
"""

from __future__ import annotations

from pi_agent.types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CancellationToken,
    TextContent,
)
from pydantic import BaseModel, ConfigDict

from .path_utils import resolve_path
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ReadToolInput(BaseModel):
    """Read tool input parameters."""

    path: str
    offset: int | None = None
    limit: int | None = None


class ReadToolDetails(BaseModel):
    """Read tool output details."""

    truncation: TruncationResult | None = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class ReadOperations:
    """Pluggable operations for the read tool.

    Override these to delegate file reading to remote systems.

    """

    async def read_file(self, absolute_path: str) -> bytes:
        """Read file contents as bytes."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="rb") as f:
            return await f.read()

    async def access(self, absolute_path: str) -> None:
        """Check if file is readable (throw if not)."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="rb") as f:
            await f.read(1)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class ReadToolOptions(BaseModel):
    """Read tool options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: ReadOperations | None = None


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_read_tool_definition(
    cwd: str,
    options: ReadToolOptions | None = None,
) -> ToolDefinition[ReadToolDetails | None]:
    """Create a read tool definition.


    """
    ops = (
        options.operations
        if options is not None and options.operations is not None
        else ReadOperations()
    )

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        path = str(params["path"])
        offset_val = params.get("offset")
        offset: int | None = None
        if offset_val is not None and isinstance(offset_val, (int, float)):
            offset = int(offset_val)
        limit_val = params.get("limit")
        limit: int | None = None
        if limit_val is not None and isinstance(limit_val, (int, float)):
            limit = int(limit_val)

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        absolute_path = resolve_path(path, cwd)

        # Check if file exists and is readable
        await ops.access(absolute_path)
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        # Read text content
        buffer = await ops.read_file(absolute_path)
        text_content = buffer.decode("utf-8")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)

        # Apply offset if specified
        start_line = max(0, offset - 1) if offset is not None else 0
        start_line_display = start_line + 1

        if start_line >= len(all_lines):
            raise RuntimeError(
                f"Offset {offset} is beyond end of file ({all_lines} lines total)"
            )

        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        # Apply truncation
        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, "
                f"exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | "
                f"head -c {DEFAULT_MAX_BYTES}]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines}. Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < len(
            all_lines
        ):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = f"{truncation.content}\n\n[{remaining} more lines in file. Use offset={next_offset} to continue.]"
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details=details,
        )

    return ToolDefinition(
        name="read",
        label="read",
        description=(
            f"Read the contents of a file. Supports text files. "
            f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Use offset/limit for large files."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to read (relative or absolute)",
                },
                "offset": {
                    "type": "number",
                    "description": "Line number to start reading from (1-indexed)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of lines to read",
                },
            },
            "required": ["path"],
        },
        execute=_execute,
    )


def create_read_tool(
    cwd: str,
    options: ReadToolOptions | None = None,
) -> AgentTool:
    """Create a read tool (AgentTool).

    """
    return wrap_tool_definition(create_read_tool_definition(cwd, options))
