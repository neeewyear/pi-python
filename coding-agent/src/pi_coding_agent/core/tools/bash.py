"""Bash tool - executes bash commands.

Uses asyncio.create_subprocess_exec for command execution.
Supports timeout control, throttled progress updates, output truncation,
and full output persistence to temp files.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from collections.abc import Callable

from pi_agent.types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CancellationToken,
    TextContent,
)
from pydantic import BaseModel, ConfigDict

from .output_accumulator import (
    OutputAccumulator,
    OutputAccumulatorOptions,
    OutputSnapshot,
)
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition
from .truncate import DEFAULT_MAX_BYTES, DEFAULT_MAX_LINES, format_size

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_UPDATE_THROTTLE_MS = 100


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class BashToolInput(BaseModel):
    """Bash tool input parameters."""

    command: str
    timeout: int | None = None


class BashToolDetails(BaseModel):
    """Bash tool output details."""

    truncation: object | None = None
    full_output_path: str | None = None


class BashSpawnContext(BaseModel):
    """Bash spawn context."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command: str
    cwd: str
    env: dict[str, str]


BashSpawnHook = Callable[[BashSpawnContext], BashSpawnContext]


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class BashExecResult:
    """Result of a bash execution."""

    def __init__(self, exit_code: int | None) -> None:
        self.exit_code = exit_code


class BashExecOptions:
    """Options for bash execution."""

    def __init__(
        self,
        on_data: Callable[[bytes], None],
        signal: CancellationToken | None = None,
        timeout: int | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.on_data = on_data
        self.signal = signal
        self.timeout = timeout
        self.env = env


class BashOperations:
    """Pluggable operations for the bash tool.

    Override these to delegate command execution to remote systems.

    """

    async def exec(
        self,
        command: str,
        cwd: str,
        options: BashExecOptions,
    ) -> BashExecResult:
        """Execute a command and stream output."""
        shell = os.environ.get("SHELL", "/bin/bash")

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("aborted")

        # Check working directory exists
        if not os.path.isdir(cwd):
            raise RuntimeError(
                f"Working directory does not exist: {cwd}\nCannot execute bash commands."
            )

        process = await asyncio.create_subprocess_exec(
            shell,
            "-c",
            command,
            cwd=cwd,
            env=options.env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        timed_out = False

        async def _read_stream(stream: asyncio.StreamReader) -> None:
            while True:
                chunk = await stream.read(4096)
                if not chunk:
                    break
                options.on_data(chunk)

        async def _run_with_timeout() -> int:
            try:
                timeout_ctx = (
                    asyncio.timeout(options.timeout)
                    if options.timeout is not None
                    else contextlib.nullcontext()
                )
                async with timeout_ctx:
                    stdout_stream = process.stdout
                    stderr_stream = process.stderr
                    stdout_task = asyncio.ensure_future(_read_stream(stdout_stream))  # type: ignore[arg-type]
                    stderr_task = asyncio.ensure_future(_read_stream(stderr_stream))  # type: ignore[arg-type]
                    await asyncio.gather(stdout_task, stderr_task)
                    return await process.wait()
            except asyncio.TimeoutError:
                nonlocal timed_out
                timed_out = True
                process.kill()
                return -1

        exit_code = await _run_with_timeout()

        if timed_out:
            raise RuntimeError(f"timeout:{options.timeout}")

        if options.signal is not None and options.signal.aborted:
            raise RuntimeError("aborted")

        return BashExecResult(exit_code=exit_code)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class BashToolOptions(BaseModel):
    """Bash tool options."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: BashOperations | None = None
    command_prefix: str | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _validate_timeout(timeout: int | None) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_bash_tool_definition(
    cwd: str,
    options: BashToolOptions | None = None,
) -> ToolDefinition[BashToolDetails | None]:
    """Create a bash tool definition.

    """
    ops = (
        options.operations
        if options is not None and options.operations is not None
        else BashOperations()
    )
    command_prefix = options.command_prefix if options is not None else None

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        command = str(params["command"])
        timeout_val = params.get("timeout")
        timeout: int | None = None
        if timeout_val is not None and isinstance(timeout_val, (int, float)):
            timeout = int(timeout_val)
        _validate_timeout(timeout)

        resolved_command = f"{command_prefix}\n{command}" if command_prefix else command

        output = OutputAccumulator(OutputAccumulatorOptions(temp_file_prefix="pi-bash"))
        accepting_output = True
        update_timer: asyncio.Task[None] | None = None
        update_dirty = False
        last_update_at = 0.0

        def _emit_output_update() -> None:
            nonlocal update_dirty, last_update_at
            if on_update is None or not update_dirty:
                return
            update_dirty = False
            last_update_at = time.monotonic() * 1000
            snapshot = output.snapshot(persist_if_truncated=True)
            on_update(
                AgentToolResult(
                    content=[TextContent(type="text", text=snapshot.content or "")],
                    details=BashToolDetails(
                        truncation=snapshot.truncation
                        if snapshot.truncation.truncated
                        else None,
                        full_output_path=snapshot.full_output_path,
                    ),
                )
            )

        def _schedule_output_update() -> None:
            nonlocal update_timer
            if on_update is None:
                return
            update_dirty = True
            delay = BASH_UPDATE_THROTTLE_MS - (time.monotonic() * 1000 - last_update_at)
            if delay <= 0:
                if update_timer is not None:
                    update_timer.cancel()
                    update_timer = None
                _emit_output_update()
                return
            if update_timer is None:

                async def _delayed() -> None:
                    nonlocal update_timer
                    await asyncio.sleep(delay / 1000.0)
                    update_timer = None
                    _emit_output_update()

                update_timer = asyncio.ensure_future(_delayed())

        def _clear_update_timer() -> None:
            nonlocal update_timer
            if update_timer is not None:
                update_timer.cancel()
                update_timer = None

        def _handle_data(data: bytes) -> None:
            if not accepting_output:
                return
            output.append(data)
            _schedule_output_update()

        async def _finish_output() -> OutputSnapshot:
            nonlocal accepting_output
            accepting_output = False
            output.finish()
            _clear_update_timer()
            _emit_output_update()
            snapshot = output.snapshot(persist_if_truncated=True)
            await output.close_temp_file()
            return snapshot

        def _format_output(
            snapshot: OutputSnapshot, empty_text: str = "(no output)"
        ) -> tuple[str, BashToolDetails | None]:
            truncation = snapshot.truncation
            text = snapshot.content or empty_text
            details: BashToolDetails | None = None
            if truncation.truncated:
                details = BashToolDetails(
                    truncation=truncation,
                    full_output_path=snapshot.full_output_path,
                )
                start_line = truncation.total_lines - truncation.output_lines + 1
                end_line = truncation.total_lines
                if truncation.last_line_partial:
                    last_line_size = format_size(output.get_last_line_bytes())
                    text += (
                        f"\n\n[Showing last {format_size(truncation.output_bytes)} "
                        f"of line {end_line} (line is {last_line_size}). "
                        f"Full output: {snapshot.full_output_path}]"
                    )
                elif truncation.truncated_by == "lines":
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} "
                        f"of {truncation.total_lines}. "
                        f"Full output: {snapshot.full_output_path}]"
                    )
                else:
                    text += (
                        f"\n\n[Showing lines {start_line}-{end_line} "
                        f"of {truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). "
                        f"Full output: {snapshot.full_output_path}]"
                    )
            return text, details

        def _append_status(text: str, status: str) -> str:
            return f"{text}\n\n{status}" if text else status

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        try:
            exit_code: int | None
            try:
                result = await ops.exec(
                    resolved_command,
                    cwd,
                    BashExecOptions(
                        on_data=_handle_data,
                        signal=signal,
                        timeout=timeout,
                    ),
                )
                exit_code = result.exit_code
            except RuntimeError as err:
                snapshot = await _finish_output()
                text, _details = _format_output(snapshot, "")
                msg = str(err)
                if msg == "aborted":
                    raise RuntimeError(_append_status(text, "Command aborted")) from err
                if msg.startswith("timeout:"):
                    timeout_secs = msg.split(":")[1]
                    raise RuntimeError(
                        _append_status(
                            text, f"Command timed out after {timeout_secs} seconds"
                        )
                    ) from err
                raise

            snapshot = await _finish_output()
            output_text, details = _format_output(snapshot)
            if exit_code is not None and exit_code != 0:
                raise RuntimeError(
                    _append_status(output_text, f"Command exited with code {exit_code}")
                )

            return AgentToolResult(
                content=[TextContent(type="text", text=output_text)],
                details=details,
            )
        finally:
            _clear_update_timer()

    return ToolDefinition(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. "
            f"Returns stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            f"(whichever is hit first). If truncated, full output is saved "
            f"to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters={
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Bash command to execute"},
                "timeout": {
                    "type": "number",
                    "description": "Timeout in seconds (optional, no default timeout)",
                },
            },
            "required": ["command"],
        },
        execute=_execute,
    )


def create_bash_tool(
    cwd: str,
    options: BashToolOptions | None = None,
) -> AgentTool:
    """Create a bash tool (AgentTool).


    """
    return wrap_tool_definition(create_bash_tool_definition(cwd, options))
