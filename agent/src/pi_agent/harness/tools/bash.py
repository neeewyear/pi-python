"""Bash 工具。

提供 ``createBashTool`` 工厂函数，生成执行 shell 命令的工具定义。
支持超时控制、100ms 节流进度更新、输出截断与完整落盘。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from ...cancellation import CancellationToken
from ...result import get_or_throw
from ...types import AgentToolResult, AgentToolUpdateCallback, TextContent
from ..types import AgentHarnessTool
from ..utils.shell_output import (
    ShellCaptureOptions,
    ShellCaptureProgress,
    execute_shell_with_capture,
)
from ..utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
)
from .tool_context import ExecutionToolContext

MAX_TIMEOUT_SECONDS = 2_147_483_647 / 1000
BASH_UPDATE_THROTTLE_MS = 100


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class BashToolInput(BaseModel):
    """Bash 工具输入参数。"""

    command: str
    timeout: int | None = None


class BashToolDetails(BaseModel):
    """Bash 工具输出详情。"""

    truncation: TruncationResult | None = None
    full_output_path: str | None = None


class BashExecution(BaseModel):
    """Bash 执行配置。"""

    command: str
    cwd: str
    env: dict[str, str]
    inherit_env: bool = True


BashPrepare = Callable[
    [BashExecution, ExecutionToolContext, CancellationToken | None],
    None | Awaitable[None],
]


class BashToolOptions(BaseModel):
    """Bash 工具选项。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    command_prefix: str | None = None
    prepare: BashPrepare | None = None


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def _validate_timeout(timeout: int | None) -> None:
    if timeout is None:
        return
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        raise ValueError("Invalid timeout: must be a finite number of seconds")
    if timeout > MAX_TIMEOUT_SECONDS:
        raise ValueError(f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds")


def create_bash_tool(
    options: BashToolOptions | None = None,
) -> AgentHarnessTool:
    """创建 Bash 工具。"""
    opts = options or BashToolOptions()

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
        context: object,
    ) -> AgentToolResult:
        command = str(params["command"])
        timeout_val = params.get("timeout")
        timeout: int | None = None
        if timeout_val is not None:
            if isinstance(timeout_val, (int, float)) or isinstance(timeout_val, str):
                timeout = int(timeout_val)
        _validate_timeout(timeout)

        ctx = context
        if not isinstance(ctx, ExecutionToolContext):
            raise TypeError("context must be ExecutionToolContext")
        env = ctx.env

        execution = BashExecution(
            command=f"{opts.command_prefix}\n{command}"
            if opts.command_prefix
            else command,
            cwd=env.cwd,
            env={},
            inherit_env=True,
        )

        if opts.prepare is not None:
            result = opts.prepare(execution, ctx, signal)
            if result is not None:
                await result

        # 进度更新节流状态
        get_latest_progress: Callable[[], ShellCaptureProgress] | None = None
        update_timer: asyncio.Task[None] | None = None
        update_dirty = False
        last_update_at = 0.0

        def _emit_output_update() -> None:
            nonlocal update_dirty, last_update_at
            if on_update is None or not update_dirty or get_latest_progress is None:
                return
            update_dirty = False
            last_update_at = time.monotonic() * 1000
            progress = get_latest_progress()
            on_update(
                AgentToolResult(
                    content=[TextContent(type="text", text=progress.output)],
                    details=BashToolDetails(
                        truncation=(
                            progress.truncation
                            if progress.truncation.truncated
                            else None
                        ),
                        full_output_path=progress.full_output_path,
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

        def _on_chunk(
            chunk: str, get_progress: Callable[[], ShellCaptureProgress]
        ) -> None:
            nonlocal get_latest_progress
            get_latest_progress = get_progress
            _schedule_output_update()

        if on_update is not None:
            on_update(AgentToolResult(content=[], details=None))

        try:
            capture = get_or_throw(
                await execute_shell_with_capture(
                    env,
                    execution.command,
                    ShellCaptureOptions(
                        cwd=execution.cwd,
                        env=execution.env,
                        inherit_env=execution.inherit_env,
                        timeout=timeout,
                        abort_signal=signal,
                        return_execution_errors=True,
                        on_chunk=_on_chunk,
                    ),
                )
            )

            _clear_update_timer()
            get_latest_progress = lambda: capture
            update_dirty = True
            _emit_output_update()

            output_text = capture.output
            details: BashToolDetails | None = None

            if capture.truncation.truncated:
                details = BashToolDetails(
                    truncation=capture.truncation,
                    full_output_path=capture.full_output_path,
                )
                start_line = (
                    capture.truncation.total_lines - capture.truncation.output_lines + 1
                )
                end_line = capture.truncation.total_lines
                if capture.truncation.last_line_partial:
                    last_line_size = format_size(capture.last_line_bytes)
                    output_text += (
                        f"\n\n[Showing last {format_size(capture.truncation.output_bytes)} "
                        f"of line {end_line} (line is {last_line_size}). "
                        f"Full output: {capture.full_output_path}]"
                    )
                elif capture.truncation.truncated_by == "lines":
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} "
                        f"of {capture.truncation.total_lines}. "
                        f"Full output: {capture.full_output_path}]"
                    )
                else:
                    output_text += (
                        f"\n\n[Showing lines {start_line}-{end_line} "
                        f"of {capture.truncation.total_lines} "
                        f"({format_size(DEFAULT_MAX_BYTES)} limit). "
                        f"Full output: {capture.full_output_path}]"
                    )

            def _status(status: str) -> str:
                return f"{output_text}\n\n{status}" if output_text else status

            if capture.cancelled:
                raise RuntimeError(_status("Command aborted"))
            if (
                capture.execution_error is not None
                and capture.execution_error.code == "timeout"
            ):
                raise RuntimeError(
                    _status(f"Command timed out after {timeout} seconds")
                )
            if capture.execution_error is not None:
                raise capture.execution_error
            if capture.exit_code is not None and capture.exit_code != 0:
                raise RuntimeError(
                    _status(f"Command exited with code {capture.exit_code}")
                )

            return AgentToolResult(
                content=[TextContent(type="text", text=output_text or "(no output)")],
                details=details,
            )
        finally:
            _clear_update_timer()

    return AgentHarnessTool(
        name="bash",
        label="bash",
        description=(
            f"Execute a bash command in the current working directory. "
            f"Returns stdout and stderr. Output is truncated to last "
            f"{DEFAULT_MAX_LINES} lines or {DEFAULT_MAX_BYTES // 1024}KB "
            f"(whichever is hit first). If truncated, full output is saved "
            f"to a temp file. Optionally provide a timeout in seconds."
        ),
        parameters={"type": "object", "properties": {}},
        execute=_execute,
    )
