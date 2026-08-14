"""Grep tool - search file contents for patterns.

Uses asyncio.create_subprocess_exec to run ripgrep (rg).
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

from .path_utils import resolve_path
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition
from .truncate import (
    DEFAULT_MAX_BYTES,
    GREP_MAX_LINE_LENGTH,
    TruncationOptions,
    TruncationResult,
    format_size,
    truncate_head,
    truncate_line,
)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class GrepToolInput(BaseModel):
    """Grep tool input parameters."""

    pattern: str
    path: str | None = None
    glob: str | None = None
    ignore_case: bool | None = None
    literal: bool | None = None
    context: int | None = None
    limit: int | None = None


class GrepToolDetails(BaseModel):
    """Grep tool output details."""

    truncation: TruncationResult | None = None
    match_limit_reached: int | None = None
    lines_truncated: bool | None = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class GrepOperations:
    """Pluggable operations for the grep tool.

    Override these to delegate search to remote systems.

    """

    async def is_directory(self, absolute_path: str) -> bool:
        """Check if path is a directory."""
        return os.path.isdir(absolute_path)

    async def read_file(self, absolute_path: str) -> str:
        """Read file contents for context lines."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="r", encoding="utf-8") as f:
            return await f.read()


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class GrepToolOptions(BaseModel):
    """Grep tool options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: GrepOperations | None = None


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_LIMIT = 100


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_grep_tool_definition(
    cwd: str,
    options: GrepToolOptions | None = None,
) -> ToolDefinition[GrepToolDetails | None]:
    """Create a grep tool definition.


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
        glob_val = params.get("glob")
        glob_pattern: str | None = str(glob_val) if glob_val is not None else None
        ignore_case = params.get("ignore_case", False)
        literal = params.get("literal", False)
        context_val = params.get("context", 0)
        context = (
            int(context_val)
            if isinstance(context_val, (int, float)) and context_val is not None
            else 0
        )
        limit_val = params.get("limit", DEFAULT_LIMIT)
        effective_limit = max(
            1, int(limit_val) if isinstance(limit_val, (int, float)) else DEFAULT_LIMIT
        )

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        search_path = resolve_path(search_dir or ".", cwd)
        ops = custom_ops or GrepOperations()

        # Check if path exists
        if not os.path.exists(search_path):
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=f"Path not found: {search_path}")
                ],
                details=None,
            )

        is_directory = await ops.is_directory(search_path)
        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        # Build rg args
        rg_args: list[str] = ["--json", "--line-number", "--color=never", "--hidden"]
        if ignore_case:
            rg_args.append("--ignore-case")
        if literal:
            rg_args.append("--fixed-strings")
        if glob_pattern:
            rg_args.extend(["--glob", glob_pattern])
        rg_args.extend(["--", pattern, search_path])

        try:
            process = await asyncio.create_subprocess_exec(
                "rg",
                *rg_args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except FileNotFoundError:
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text="ripgrep (rg) is not available. Please install it first.",
                    )
                ],
                details=None,
            )

        if signal is not None and signal.aborted:
            process.kill()
            raise RuntimeError("Operation aborted")

        # Collect matches
        match_count = 0
        match_limit_reached = False
        lines_truncated = False
        matches: list[dict[str, object]] = []
        killed_due_to_limit = False

        async def _read_stdout() -> None:
            nonlocal match_count, match_limit_reached, killed_due_to_limit
            assert process.stdout is not None
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").strip()
                if not text or match_count >= effective_limit:
                    continue
                try:
                    import json

                    event = json.loads(text)
                except (json.JSONDecodeError, ValueError):
                    continue
                if event.get("type") == "match":
                    match_count += 1
                    data = event.get("data", {})
                    file_path = data.get("path", {}).get("text")
                    line_number = data.get("line_number")
                    line_text = data.get("lines", {}).get("text")
                    if file_path and isinstance(line_number, (int, float)):
                        matches.append(
                            {
                                "filePath": file_path,
                                "lineNumber": int(line_number),
                                "lineText": line_text,
                            }
                        )
                    if match_count >= effective_limit:
                        match_limit_reached = True
                        killed_due_to_limit = True
                        process.kill()
                        break

        stderr_lines: list[str] = []

        async def _read_stderr() -> None:
            assert process.stderr is not None
            while True:
                line = await process.stderr.readline()
                if not line:
                    break
                stderr_lines.append(line.decode("utf-8", errors="replace"))

        await asyncio.gather(_read_stdout(), _read_stderr())
        await process.wait()

        if signal is not None and signal.aborted:
            raise RuntimeError("Operation aborted")

        if (
            not killed_due_to_limit
            and process.returncode is not None
            and process.returncode not in (0, 1)
        ):
            error_msg = (
                "".join(stderr_lines).strip()
                or f"ripgrep exited with code {process.returncode}"
            )
            return AgentToolResult(
                content=[TextContent(type="text", text=error_msg)],
                details=None,
            )

        if match_count == 0:
            return AgentToolResult(
                content=[TextContent(type="text", text="No matches found")],
                details=None,
            )

        # Format matches
        output_lines: list[str] = []
        file_cache: dict[str, list[str]] = {}

        async def _get_file_lines(file_path: str) -> list[str]:
            if file_path in file_cache:
                return file_cache[file_path]
            try:
                content = await ops.read_file(file_path)
                lines = content.replace("\r\n", "\n").replace("\r", "\n").split("\n")
            except Exception:
                lines = []
            file_cache[file_path] = lines
            return lines

        def _format_path(file_path: str) -> str:
            if is_directory:
                rel = os.path.relpath(file_path, search_path)
                if rel and not rel.startswith(".."):
                    return rel.replace("\\", "/")
            return os.path.basename(file_path)

        for match in matches:
            fp = str(match.get("filePath", ""))
            ln_val = match.get("lineNumber", 0)
            ln = int(ln_val) if isinstance(ln_val, (int, float)) else 0
            lt = match.get("lineText")

            if context == 0 and lt is not None:
                relative_path = _format_path(fp)
                sanitized = (
                    str(lt).replace("\r\n", "\n").replace("\r", "").replace("\n$", "")
                )
                truncated, was_truncated = truncate_line(sanitized)
                if was_truncated:
                    lines_truncated = True
                output_lines.append(f"{relative_path}:{ln}: {truncated}")
            else:
                relative_path = _format_path(fp)
                file_lines = await _get_file_lines(fp)
                if not file_lines:
                    output_lines.append(f"{relative_path}:{ln}: (unable to read file)")
                    continue
                start = max(1, ln - context) if context > 0 else ln
                end = min(len(file_lines), ln + context) if context > 0 else ln
                for current in range(start, end + 1):
                    line_text = (
                        file_lines[current - 1] if current <= len(file_lines) else ""
                    )
                    sanitized = line_text.replace("\r", "")
                    if current == ln:
                        truncated, was_truncated = truncate_line(sanitized)
                        if was_truncated:
                            lines_truncated = True
                        output_lines.append(f"{relative_path}:{current}: {truncated}")
                    else:
                        output_lines.append(f"{relative_path}-{current}- {sanitized}")

        raw_output = "\n".join(output_lines)
        truncation = truncate_head(raw_output, TruncationOptions(max_lines=1000000))
        output = truncation.content
        details: GrepToolDetails = GrepToolDetails()
        notices: list[str] = []

        if match_limit_reached:
            notices.append(
                f"{effective_limit} matches limit reached. Use limit={effective_limit * 2} for more, or refine pattern"
            )
            details.match_limit_reached = effective_limit
        if truncation.truncated:
            notices.append(f"{format_size(DEFAULT_MAX_BYTES)} limit reached")
            details.truncation = truncation
        if lines_truncated:
            notices.append(
                f"Some lines truncated to {GREP_MAX_LINE_LENGTH} chars. Use read tool to see full lines"
            )
            details.lines_truncated = True

        if notices:
            output += f"\n\n[{' '.join(notices)}]"

        return AgentToolResult(
            content=[TextContent(type="text", text=output)],
            details=details
            if (
                details.match_limit_reached is not None
                or details.truncation is not None
                or details.lines_truncated
            )
            else None,
        )

    return ToolDefinition(
        name="grep",
        label="grep",
        description=(
            f"Search file contents for a pattern. Returns matching lines with file paths "
            f"and line numbers. Respects .gitignore. Output is truncated to {DEFAULT_LIMIT} "
            f"matches or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Long lines are truncated to {GREP_MAX_LINE_LENGTH} chars."
        ),
        parameters={
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Search pattern (regex or literal string)",
                },
                "path": {
                    "type": "string",
                    "description": "Directory or file to search (default: current directory)",
                },
                "glob": {
                    "type": "string",
                    "description": "Filter files by glob pattern, e.g. '*.py'",
                },
                "ignoreCase": {
                    "type": "boolean",
                    "description": "Case-insensitive search (default: false)",
                },
                "literal": {
                    "type": "boolean",
                    "description": "Treat pattern as literal string instead of regex (default: false)",
                },
                "context": {
                    "type": "number",
                    "description": "Number of lines to show before and after each match (default: 0)",
                },
                "limit": {
                    "type": "number",
                    "description": "Maximum number of matches to return (default: 100)",
                },
            },
            "required": ["pattern"],
        },
        execute=_execute,
    )


def create_grep_tool(
    cwd: str,
    options: GrepToolOptions | None = None,
) -> AgentTool:
    """Create a grep tool (AgentTool).

    """
    return wrap_tool_definition(create_grep_tool_definition(cwd, options))
