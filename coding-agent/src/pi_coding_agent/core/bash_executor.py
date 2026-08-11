from __future__ import annotations

import asyncio
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from pi_agent.types import CancellationToken

from ..utils.ansi import strip_ansi
from ..utils.shell import sanitize_binary_output
from .tools.bash import BashExecOptions, BashOperations
from .tools.truncate import DEFAULT_MAX_BYTES, truncate_tail


@dataclass
class BashExecutorOptions:
    """Callback for streaming output chunks (already sanitized)."""

    on_chunk: Callable[[str], None] | None = None
    """AbortSignal for cancellation."""
    signal: CancellationToken | None = None


@dataclass
class BashResult:
    """Combined stdout + stderr output (sanitized, possibly truncated)."""

    output: str
    """Process exit code (None if killed/cancelled)."""
    exit_code: int | None = None
    """Whether the command was cancelled via signal."""
    cancelled: bool = False
    """Whether the output was truncated."""
    truncated: bool = False
    """Path to temp file containing full output (if output exceeded truncation threshold)."""
    full_output_path: str | None = None


async def execute_bash_with_operations(
    command: str,
    cwd: str,
    operations: BashOperations,
    options: BashExecutorOptions | None = None,
) -> BashResult:
    """Execute a bash command using custom BashOperations.
    Used for remote execution (SSH, containers, etc.).
    """
    output_chunks: list[str] = []
    output_bytes = 0
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    temp_file_path: str | None = None
    temp_file_handle = None
    total_bytes = 0

    async def ensure_temp_file() -> None:
        nonlocal temp_file_path, temp_file_handle
        if temp_file_path:
            return
        import secrets

        import aiofiles

        id_ = secrets.token_hex(8)
        temp_file_path = str(Path(tempfile.gettempdir(), f"pi-bash-{id_}.log"))
        temp_file_handle = await aiofiles.open(temp_file_path, mode="w")
        for chunk in output_chunks:
            await temp_file_handle.write(chunk)

    cancelled = False

    def on_data(data: bytes) -> None:
        nonlocal total_bytes, output_bytes, output_chunks
        total_bytes += len(data)

        text = sanitize_binary_output(
            strip_ansi(data.decode("utf-8", errors="replace"))
        ).replace("\r", "")

        if total_bytes > DEFAULT_MAX_BYTES:
            asyncio.ensure_future(ensure_temp_file())

        if temp_file_handle:
            asyncio.ensure_future(temp_file_handle.write(text))

        output_chunks.append(text)
        output_bytes += len(text)
        while output_bytes > max_output_bytes and len(output_chunks) > 1:
            removed = output_chunks.pop(0)
            output_bytes -= len(removed)

        if options and options.on_chunk:
            options.on_chunk(text)

    try:
        result = await operations.exec(
            command,
            cwd,
            BashExecOptions(
                on_data=on_data,
                signal=options.signal if options else None,
            ),
        )

        full_output = "".join(output_chunks)
        truncation_result = truncate_tail(full_output)
        if truncation_result.truncated:
            await ensure_temp_file()
        if temp_file_handle:
            await temp_file_handle.close()

        cancelled = (options and options.signal and options.signal.aborted) or False

        return BashResult(
            output=truncation_result.content
            if truncation_result.truncated
            else full_output,
            exit_code=None if cancelled else (result.exit_code or None),
            cancelled=cancelled,
            truncated=truncation_result.truncated,
            full_output_path=temp_file_path,
        )
    except Exception:
        if options and options.signal and options.signal.aborted:
            full_output = "".join(output_chunks)
            truncation_result = truncate_tail(full_output)
            if truncation_result.truncated:
                await ensure_temp_file()
            if temp_file_handle:
                await temp_file_handle.close()
            return BashResult(
                output=truncation_result.content
                if truncation_result.truncated
                else full_output,
                exit_code=None,
                cancelled=True,
                truncated=truncation_result.truncated,
                full_output_path=temp_file_path,
            )

        if temp_file_handle:
            await temp_file_handle.close()
        raise
