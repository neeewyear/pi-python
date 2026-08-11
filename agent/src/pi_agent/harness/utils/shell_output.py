"""Shell 输出捕获（对应 ``harness/utils/shell-output.ts``）。

``executeShellWithCapture`` 执行 shell 命令并捕获输出，支持：
- 实时进度回调（``on_chunk``）
- 超限时自动落盘完整输出到临时文件
- 二进制输出净化
- 取消 / 错误编码进 ``ShellCaptureResult``
"""

from __future__ import annotations

import asyncio
from typing import Callable

from pydantic import BaseModel, ConfigDict

from ..types import (
    ExecutionEnv,
    ExecutionError,
    ShellExecOptions,
    ShellExecResult,
)
from ...cancellation import CancellationToken
from ...result import Result, err, ok, to_error
from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    truncate_tail,
)


class ShellCaptureProgress(BaseModel):
    """进度快照（对应 TS ``ShellCaptureProgress``）。"""

    output: str
    """当前截断后的输出。"""
    truncation: TruncationResult
    """截断详情。"""
    full_output_path: str | None = None
    """完整输出临时文件路径。"""
    last_line_bytes: int = 0
    """当前未闭合行的字节数。"""


class ShellCaptureOptions(BaseModel):
    """捕获选项（对应 TS ``ShellCaptureOptions``）。

    继承 ``ShellExecOptions`` 除 ``on_stdout`` / ``on_stderr`` 以外的字段，
    追加 ``on_chunk`` / ``return_execution_errors``。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    timeout: int | None = None
    abort_signal: CancellationToken | None = None
    on_chunk: Callable[[str, Callable[[], ShellCaptureProgress]], None] | None = None
    """每次输出数据块时的回调（chunk, get_progress）。"""
    return_execution_errors: bool = False
    """为 True 时，将执行失败编码进 result.execution_error 而非返回 err。"""


class ShellCaptureResult(ShellCaptureProgress):
    """最终捕获结果（对应 TS ``ShellCaptureResult``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    exit_code: int | None = None
    cancelled: bool = False
    truncated: bool = False
    execution_error: ExecutionError | None = None


def _to_execution_error(error: object) -> ExecutionError:
    """把任意异常/错误规范化为 ``ExecutionError``。"""
    if isinstance(error, ExecutionError):
        return error
    cause = to_error(error)
    return ExecutionError("unknown", cause.args[0] if cause.args else str(cause), cause=cause)


def sanitize_binary_output(raw: str) -> str:
    """过滤二进制控制字符，只保留可打印字符 + ``\\t`` ``\\n`` ``\\r``。"""
    result: list[str] = []
    for char in raw:
        code = ord(char)
        if code in (0x09, 0x0A, 0x0D):  # tab, newline, carriage return
            result.append(char)
            continue
        if code <= 0x1F:
            continue
        if 0xFFF9 <= code <= 0xFFFB:
            continue
        result.append(char)
    return "".join(result)


def _trim_to_last_utf8_bytes(text: str, max_bytes: int) -> str:
    """截取字符串末尾不超过 ``max_bytes`` 字节，保证 UTF-8 字符边界对齐。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    start = len(encoded) - max_bytes
    # 跳过 continuation bytes（10xxxxxx）
    while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
        start += 1
    return encoded[start:].decode("utf-8")


async def execute_shell_with_capture(
    env: ExecutionEnv,
    command: str,
    options: ShellCaptureOptions | None = None,
) -> Result[ShellCaptureResult, ExecutionError]:
    """执行 shell 命令并捕获输出（对应 TS ``executeShellWithCapture``）。

    契约：
    - 取消时返回 ``cancelled=True``，不报错
    - ``return_execution_errors=True`` 时，执行失败编码进 ``execution_error``
    - 输出超过 ``DEFAULT_MAX_LINES`` 行或 ``DEFAULT_MAX_BYTES`` 字节时自动落盘
    """
    opts = options or ShellCaptureOptions()
    tail_output = ""
    max_output_bytes = DEFAULT_MAX_BYTES * 2

    total_bytes = 0
    completed_lines = 0
    has_open_line = False
    current_line_bytes = 0
    full_output_path: str | None = None
    full_output_requested = False
    accepting_output = True
    write_lock = asyncio.Lock()
    write_error: ExecutionError | None = None
    capture_error: ExecutionError | None = None

    # ── 内部辅助 ────────────────────────────────────────────────────

    def _create_progress() -> ShellCaptureProgress:
        tail_truncation = truncate_tail(tail_output)
        total_lines_count = completed_lines + (1 if has_open_line else 0)
        truncated = total_lines_count > DEFAULT_MAX_LINES or total_bytes > DEFAULT_MAX_BYTES
        truncation = TruncationResult(
            content=tail_truncation.content,
            truncated=truncated,
            truncated_by=(
                tail_truncation.truncated_by
                if truncated and tail_truncation.truncated_by
                else (
                    "bytes"
                    if truncated and total_bytes > DEFAULT_MAX_BYTES
                    else ("lines" if truncated else None)
                )
            ),
            total_lines=total_lines_count,
            total_bytes=total_bytes,
            output_lines=tail_truncation.output_lines,
            output_bytes=tail_truncation.output_bytes,
            last_line_partial=tail_truncation.last_line_partial,
            first_line_exceeds_limit=tail_truncation.first_line_exceeds_limit,
            max_lines=tail_truncation.max_lines,
            max_bytes=tail_truncation.max_bytes,
        )
        return ShellCaptureProgress(
            output=truncation.content if truncated else tail_output,
            truncation=truncation,
            full_output_path=full_output_path,
            last_line_bytes=current_line_bytes,
        )

    async def _ensure_full_output_file(initial_content: str) -> None:
        nonlocal full_output_requested, full_output_path, write_error
        if full_output_requested or capture_error is not None:
            return
        full_output_requested = True
        async with write_lock:
            if write_error is not None:
                return
            temp_result = await env.create_temp_file({"prefix": "bash-", "suffix": ".log"})
            if not temp_result.is_ok():
                write_error = _to_execution_error(temp_result.error)
                return
            full_output_path = temp_result.value
            append_result = await env.append_file(full_output_path, initial_content)
            if not append_result.is_ok():
                write_error = _to_execution_error(append_result.error)

    async def _append_full_output(text: str) -> None:
        nonlocal write_error
        if not full_output_requested or capture_error is not None:
            return
        if full_output_path is None:
            write_error = ExecutionError("unknown", "Full output path was not created")
            return
        async with write_lock:
            if write_error is not None:
                return
            append_result = await env.append_file(full_output_path, text)
            if not append_result.is_ok():
                write_error = _to_execution_error(append_result.error)

    # ── on_chunk 回调 ───────────────────────────────────────────────

    def _on_chunk(chunk: str) -> None:
        nonlocal tail_output, total_bytes, completed_lines, has_open_line, current_line_bytes
        nonlocal capture_error
        if not accepting_output:
            return
        try:
            text = sanitize_binary_output(chunk).replace("\r", "")
            text_bytes = len(text.encode("utf-8"))
            total_bytes += text_bytes
            newline_count = text.count("\n")
            completed_lines += newline_count
            last_newline = text.rfind("\n")
            if last_newline >= 0:
                trailing_text = text[last_newline + 1 :]
                current_line_bytes = len(trailing_text.encode("utf-8"))
                has_open_line = len(trailing_text) > 0
            elif len(text) > 0:
                current_line_bytes += text_bytes
                has_open_line = True

            tail_output += text
            total_lines_count = completed_lines + (1 if has_open_line else 0)
            if (total_bytes > DEFAULT_MAX_BYTES or total_lines_count > DEFAULT_MAX_LINES) and not full_output_requested:
                asyncio.ensure_future(_ensure_full_output_file(tail_output))
            elif full_output_requested:
                asyncio.ensure_future(_append_full_output(text))

            tail_output = _trim_to_last_utf8_bytes(tail_output, max_output_bytes)
            if opts.on_chunk is not None:
                opts.on_chunk(text, _create_progress)
        except Exception as exc:
            capture_error = _to_execution_error(exc)

    # ── 主流程 ──────────────────────────────────────────────────────

    try:
        exec_opts = ShellExecOptions(
            cwd=opts.cwd,
            env=opts.env,
            inherit_env=opts.inherit_env,
            timeout=opts.timeout,
            abort_signal=opts.abort_signal,
            on_stdout=_on_chunk,
            on_stderr=_on_chunk,
        )
        result = await env.exec(command, exec_opts)
        accepting_output = False

        progress = _create_progress()
        if progress.truncation.truncated and not full_output_requested:
            await _ensure_full_output_file(tail_output)

        if write_error is not None:
            return err(write_error)
        if capture_error is not None:
            return err(capture_error)

        progress = _create_progress()

        if not result.is_ok():
            if result.error.code == "aborted" or (
                opts.abort_signal is not None and opts.abort_signal.aborted
            ):
                return ok(
                    ShellCaptureResult(
                        output=progress.output,
                        truncation=progress.truncation,
                        full_output_path=progress.full_output_path,
                        last_line_bytes=progress.last_line_bytes,
                        exit_code=None,
                        cancelled=True,
                        truncated=progress.truncation.truncated,
                    )
                )
            if opts.return_execution_errors:
                return ok(
                    ShellCaptureResult(
                        output=progress.output,
                        truncation=progress.truncation,
                        full_output_path=progress.full_output_path,
                        last_line_bytes=progress.last_line_bytes,
                        exit_code=None,
                        cancelled=False,
                        truncated=progress.truncation.truncated,
                        execution_error=result.error,
                    )
                )
            return err(result.error)

        cancelled = opts.abort_signal is not None and opts.abort_signal.aborted
        return ok(
            ShellCaptureResult(
                output=progress.output,
                truncation=progress.truncation,
                full_output_path=progress.full_output_path,
                last_line_bytes=progress.last_line_bytes,
                exit_code=None if cancelled else result.value.exit_code,
                cancelled=cancelled,
                truncated=progress.truncation.truncated,
            )
        )
    except Exception as exc:
        accepting_output = False
        return err(_to_execution_error(exc))