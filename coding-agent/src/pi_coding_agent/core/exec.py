from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExecOptions:
    """Options for executing shell commands."""
    signal: Optional[asyncio.Event] = None
    """AbortSignal to cancel the command."""
    timeout: Optional[float] = None
    """Timeout in milliseconds."""
    cwd: Optional[str] = None
    """Working directory."""


@dataclass
class ExecResult:
    """Result of executing a shell command."""
    stdout: str = ""
    stderr: str = ""
    code: int = 0
    killed: bool = False


async def exec_command(
    command: str,
    args: list[str],
    cwd: str,
    options: Optional[ExecOptions] = None,
) -> ExecResult:
    """Execute a shell command and return stdout/stderr/code.
    Supports timeout and abort signal.
    """
    proc = await asyncio.create_subprocess_exec(
        command,
        *args,
        cwd=cwd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )

    stdout = ""
    stderr = ""
    killed = False

    async def kill_process() -> None:
        nonlocal killed
        if not killed:
            killed = True
            try:
                proc.terminate()
                await asyncio.sleep(5)
                if proc.returncode is None:
                    proc.kill()
            except ProcessLookupError:
                pass

    timeout_handle: Optional[asyncio.TimerHandle] = None

    async def read_streams() -> tuple[str, str]:
        nonlocal stdout, stderr
        stdout_data, stderr_data = await proc.communicate()
        stdout = stdout_data.decode("utf-8", errors="replace") if stdout_data else ""
        stderr = stderr_data.decode("utf-8", errors="replace") if stderr_data else ""
        return stdout, stderr

    try:
        if options and options.signal and options.signal.is_set():
            await kill_process()

        if options and options.timeout and options.timeout > 0:
            async def timeout_handler() -> None:
                await kill_process()

            loop = asyncio.get_event_loop()
            timeout_handle = loop.call_later(options.timeout / 1000, lambda: asyncio.ensure_future(timeout_handler()))

        await read_streams()
    finally:
        if timeout_handle:
            timeout_handle.cancel()

    return ExecResult(
        stdout=stdout,
        stderr=stderr,
        code=proc.returncode or 0,
        killed=killed,
    )