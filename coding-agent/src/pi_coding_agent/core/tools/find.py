"""Find tool - search for files by glob pattern.

Uses asyncio.create_subprocess_exec to run fd.
"""

from __future__ import annotations

import asyncio
import os

from pi_agent.types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CancellationToken,
    TextContent,
)
from pydantic import BaseModel, ConfigDict

from .path_utils import path_exists, resolve_path
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


class FindToolInput(BaseModel):
    """Find tool input parameters (corresponds to TS ``FindToolInput``)."""

    pattern: str
    path: str | None = None
    limit: int | None = None


class FindToolDetails(BaseModel):
    """Find tool output details (corresponds to TS ``FindToolDetails``)."""

    truncation: TruncationResult | None = None
    result_limit_reached: int | None = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class FindOperations:
    """Pluggable operations for the find tool.

    Override these to delegate file search to remote systems.
    Corresponds to TS ``FindOperations``.
    """

    async def exists(self, absolute_path: str) -> bool:
        """Check if path exists."""
        return path_exists(absolute_path)

    async def glob(
        self, pattern: str, cwd: str, options: dict[str, object]
    ) -> list[str]:
        """Find files matching glob pattern. Returns relative or absolute paths."""
        _ = pattern
        _ = cwd
        _ = options
        return []


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class FindToolOptions(BaseModel):
    """Find tool options (corresponds to TS ``FindToolOptions``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: FindOperations | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 1000


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def relativize_find_result_path(result_path: str, search_path: str) -> str:
    """Relativize a find result against the search root and normalize to posix separators.

    Corresponds to TS ``relativizeFindResultPath``.
    """
    had_trailing_separator = result_path.endswith(os.sep) or (
        os.sep == "\\" and result_path.endswith("/")
    )
    relative_path = (
        os.path.relpath(result_path, search_path)
        if os.path.isabs(result_path)
        else result_path
    )
    posix_path = relative_path.replace("\\", "/")
    if had_trailing_separator and not posix_path.endswith("/"):
        return f"{posix_path}/"
    return posix_path


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_find_tool_definition(
    cwd: str,
    options: FindToolOptions | None = None,
) -> ToolDefinition[FindToolDetails | None]:
    """Create a find tool definition.

    Corresponds to TS ``createFindToolDefinition``.
    """
    custom_ops = options.operations if options is not None else None

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        pattern = str(params["pattern"])
        search_dir_val = params.get("path")
        search_dir: str | None = (
            str(search_dir_val) if search_dir_val is not None else None
        )
        limit_val = params.get("limit", DEFAULT_LIMIT)
        effective_limit = max(
            1, int(limit_val) if isinstance(limit_val, (int, float)) else DEFAULT_LIMIT
        )

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        search_path = resolve_path(search_dir or ".", cwd)

        ops = custom_ops or FindOperations()

        if not await ops.exists(search_path):
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=f"Path not found: {search_path}")
                ],
                details=None,
            )

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        # Use custom glob if provided
        if custom_ops is not None and type(custom_ops).glob is not FindOperations.glob:
            results = await ops.glob(
                pattern,
                search_path,
                {
                    "ignore": ["**/node_modules/**", "**/.git/**"],
                    "limit": effective_limit,
                },
            )
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            if not results:
                return AgentToolResult(
                    content=[
                        TextContent(type="text", text="No files found matching pattern")
                    ],
                    details=None,
                )

            relativized = [relativize_find_result_path(p, search_path) for p in results]
            result_limit_reached = len(relativized) >= effective_limit
            raw_output = "\n".join(relativized)
            truncation = truncate_head(raw_output, TruncationOptions(max_lines=1000000))
            result_output = truncation.content
            details: FindToolDetails = FindToolDetails()
            notices: list[str] = []
            if result_limit_reached:
                notices.append(f"{effective_limit} results limit reached")
                details.result_limit_reached = effective_limit
            if truncation.truncated:
                notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
                details.truncation = truncation
            if notices:
                result_output += f"\n\n[{' '.join(notices)}]"
            return AgentToolResult(
                content=[TextContent(type="text", text=result_output)],
                details=details
                if (
                    details.result_limit_reached is not None
                    or details.truncation is not None
                )
                else None,
            )

        # Default implementation uses fd
        fd_args: list[str] = ["--glob", "--color=never", "--hidden"]

        # Check if inside git repo
        inside_git_repo = False
        current = search_path
        while True:
            if os.path.isdir(os.path.join(current, ".git")):
                inside_git_repo = True
                break
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent

        if not inside_git_repo:
            fd_args.append("--no-require-git")
        fd_args.extend(["--max-results", str(effective_limit)])

        # Handle path-containing patterns
        effective_pattern = pattern
        if "/" in pattern:
            fd_args.append("--full-path")
            if (
                not pattern.startswith("/")
                and not pattern.startswith("**/")
                and pattern != "**"
            ):
                effective_pattern = f"**/{pattern}"

        fd_args.extend(["--", effective_pattern, search_path])

        try:
            process = await asyncio.create_subprocess_exec(
                "fd",
                *fd_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="fd is not available. Please install it first.",
                    )
                ],
                details=None,
            )

        if signal is not None and signal.aborted:
            process.kill()
            raise RuntimeError("Operation aborted")

        stdout_bytes, stderr_bytes = await process.communicate()

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if process.returncode != 0:
            stderr = stderr_bytes.decode("utf-8", errors="replace").strip()
            if stderr:
                return AgentToolResult(
                    content=[TextContent(type="text", text=stderr)],
                    details=None,
                )

        output = stdout_bytes.decode("utf-8", errors="replace").strip()
        if not output:
            return AgentToolResult(
                content=[
                    TextContent(type="text", text="No files found matching pattern")
                ],
                details=None,
            )

        lines = [line.strip() for line in output.split("\n") if line.strip()]
        relativized = [relativize_find_result_path(line, search_path) for line in lines]

        result_limit_reached = len(relativized) >= effective_limit
        raw_output = "\n".join(relativized)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=1000000))
        result_output = truncation.content
        details = FindToolDetails()
        notices = []
        if result_limit_reached:
            notices.append(
                f"{effective_limit} results limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.result_limit_reached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if notices:
            result_output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=result_output)],
            details=details
            if (
                details.result_limit_reached is not None
                or details.truncation is not None
            )
            else None,
        )

    return ToolDefinition(
        name="find",
        label="find",
        description=(
            f"Search for files by glob pattern. Returns matching file paths relative to "
            f"the search directory. Respects .gitignore. Output is truncated to "
            f"{DEFAULT_LIMIT} results or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files, e.g. '*.py' or 'src/**/*.ts'",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (default: current directory)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of results (default: 1000)",
                },
            },
            "required": ["pattern"],
        },
        execute=_execute,
    )


def create_find_tool(
    cwd: str,
    options: FindToolOptions | None = None,
) -> AgentTool:
    """Create a find tool (AgentTool).

    Corresponds to TS ``createFindTool``.
    """
    return wrap_tool_definition(create_find_tool_definition(cwd, options))
