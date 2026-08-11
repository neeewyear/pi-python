"""Ls tool - list directory contents.

Uses aiofiles for directory listing.
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

from .path_utils import resolve_path
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition
from .truncate import (
    DEFAULT_MAX_BYTES,
    TruncationOptions,
    TruncationResult,
    format_size,
    truncate_head,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class LsToolInput(BaseModel):
    """Ls tool input parameters (corresponds to TS ``LsToolInput``)."""

    path: str | None = None
    limit: int | None = None


class LsToolDetails(BaseModel):
    """Ls tool output details (corresponds to TS ``LsToolDetails``)."""

    truncation: TruncationResult | None = None
    entry_limit_reached: int | None = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class LsOperations:
    """Pluggable operations for the ls tool.

    Override these to delegate directory listing to remote systems.
    Corresponds to TS ``LsOperations``.
    """

    async def exists(self, absolute_path: str) -> bool:
        """Check if path exists."""
        return os.path.exists(absolute_path)

    async def stat(self, absolute_path: str) -> os.stat_result:
        """Get file or directory stats."""
        return os.stat(absolute_path)

    async def readdir(self, absolute_path: str) -> list[str]:
        """Read directory entries."""
        return os.listdir(absolute_path)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class LsToolOptions(BaseModel):
    """Ls tool options (corresponds to TS ``LsToolOptions``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: LsOperations | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 500


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_ls_tool_definition(
    cwd: str,
    options: LsToolOptions | None = None,
) -> ToolDefinition[LsToolDetails | None]:
    """Create an ls tool definition.

    Corresponds to TS ``createLsToolDefinition``.
    """
    ops = (
        options.operations
        if options is not None and options.operations is not None
        else LsOperations()
    )

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        path_val = params.get("path")
        dir_path_str: str = str(path_val) if path_val is not None else "."
        limit_val = params.get("limit", DEFAULT_LIMIT)
        effective_limit = max(
            1, int(limit_val) if isinstance(limit_val, (int, float)) else DEFAULT_LIMIT
        )

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        dir_path = resolve_path(dir_path_str, cwd)

        # Check if path exists
        if not await ops.exists(dir_path):
            raise RuntimeError(f"Path not found: {dir_path}")

        # Check if path is a directory
        stat = await ops.stat(dir_path)
        if not stat.st_mode & 0o40000:  # S_IFDIR
            raise RuntimeError(f"Not a directory: {dir_path}")

        # Read directory entries
        try:
            entries = await ops.readdir(dir_path)
        except OSError as e:
            raise RuntimeError(f"Cannot read directory: {e}") from e

        # Sort alphabetically, case-insensitive
        entries.sort(key=lambda x: x.lower())

        # Format entries with directory indicators
        results: list[str] = []
        entry_limit_reached = False
        for entry in entries:
            if len(results) >= effective_limit:
                entry_limit_reached = True
                break

            full_path = os.path.join(dir_path, entry)
            suffix = ""
            try:
                entry_stat = await ops.stat(full_path)
                if entry_stat.st_mode & 0o40000:  # S_IFDIR
                    suffix = "/"
            except OSError:
                continue
            results.append(entry + suffix)

        if not results:
            return AgentToolResult(
                content=[TextContent(type="text", text="(empty directory)")],
                details=None,
            )

        raw_output = "\n".join(results)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=1000000))
        output = truncation.content
        details: LsToolDetails = LsToolDetails()
        notices: list[str] = []
        if entry_limit_reached:
            notices.append(
                f"{effective_limit} entries limit reached. Use limit={effective_limit * 2} for more"
            )
            details.entry_limit_reached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details=details
            if (
                details.entry_limit_reached is not None
                or details.truncation is not None
            )
            else None,
        )

    return ToolDefinition(
        name="ls",
        label="ls",
        description=(
            f"List directory contents. Returns entries sorted alphabetically, "
            f"with '/' suffix for directories. Includes dotfiles. Output is truncated "
            f"to {DEFAULT_LIMIT} entries or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory to list (default: current directory)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of entries to return (default: 500)",
                },
            },
        },
        execute=_execute,
    )


def create_ls_tool(
    cwd: str,
    options: LsToolOptions | None = None,
) -> AgentTool:
    """Create an ls tool (AgentTool).

    Corresponds to TS ``createLsTool``.
    """
    return wrap_tool_definition(create_ls_tool_definition(cwd, options))
