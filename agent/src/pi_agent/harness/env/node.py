"""Shell 执行 + NodeExecutionEnv。

组合 ``NodeFileSystem`` 实现完整的 ``ExecutionEnv`` 协议。
"""

from __future__ import annotations

import asyncio
import os
import signal
from collections.abc import Callable

from ...cancellation import CancellationToken
from ...result import Result, err, ok, to_error
from ..types import (
    ExecutionError,
    ShellExecOptions,
    ShellExecResult,
)
from .node_fs import NodeFileSystem, _resolve_path

# Node.js 的 setTimeout 最大值为 2^31 - 1 毫秒
MAX_TIMEOUT_MS = 2_147_483_647
MAX_TIMEOUT_SECONDS = MAX_TIMEOUT_MS // 1000
EXIT_STDIO_GRACE_SECONDS = 0.1


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _resolve_timeout_ms(timeout: int | None) -> Result[int | None, ExecutionError]:
    """校验并转换超时。"""
    if timeout is None:
        return ok(None)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        return err(
            ExecutionError(
                "timeout", "Invalid timeout: must be a finite number of seconds"
            )
        )
    timeout_ms = int(timeout * 1000)
    if timeout_ms > MAX_TIMEOUT_MS:
        return err(
            ExecutionError(
                "timeout",
                f"Invalid timeout: maximum is {MAX_TIMEOUT_SECONDS} seconds",
            )
        )
    return ok(timeout_ms)


async def _find_bash_on_path() -> str | None:
    """在 PATH 中查找 bash。"""
    try:
        proc = await asyncio.create_subprocess_exec(
            "which",
            "bash",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5.0)
        if proc.returncode == 0 and stdout:
            first = stdout.decode().strip().split("\n")[0]
            if first and os.path.isfile(first):
                return first
    except Exception:
        pass
    return None


async def _get_shell_config(
    custom_shell_path: str | None = None,
) -> Result[tuple[str, list[str], str | None], ExecutionError]:
    """获取 shell 配置。

    返回 ``(shell_path, args, command_transport)``。
    command_transport 为 "stdin" 表示通过 stdin 传命令，否则通过 argv。
    """
    if custom_shell_path:
        if os.path.isfile(custom_shell_path):
            return ok((custom_shell_path, ["-c"], None))
        return err(
            ExecutionError(
                "shell_unavailable",
                f"Custom shell path not found: {custom_shell_path}",
            )
        )

    # macOS/Linux: 优先使用 /bin/bash
    if os.path.isfile("/bin/bash"):
        return ok(("/bin/bash", ["-c"], None))

    bash_on_path = await _find_bash_on_path()
    if bash_on_path:
        return ok((bash_on_path, ["-c"], None))

    # 回退到 sh
    return ok(("sh", ["-c"], None))


def _get_shell_env(
    base_env: dict[str, str] | None = None,
    extra_env: dict[str, str] | None = None,
    inherit_env: bool = True,
) -> dict[str, str] | None:
    """构建 shell 环境变量。"""
    if not inherit_env:
        return {**(extra_env or {})}
    result = dict(os.environ)
    if base_env:
        result.update(base_env)
    if extra_env:
        result.update(extra_env)
    return result


def _kill_process_tree(pid: int) -> None:
    """杀掉进程树（macOS 实现）。"""
    try:
        os.killpg(pid, signal.SIGKILL)
    except (ProcessLookupError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass


# ---------------------------------------------------------------------------
# NodeExecutionEnv
# ---------------------------------------------------------------------------


class NodeExecutionEnv(NodeFileSystem):
    """执行环境。

    组合 ``NodeFileSystem`` 并实现 ``Shell`` 协议，提供完整的
    ``ExecutionEnv`` 实现。
    """

    def __init__(
        self,
        cwd: str,
        shell_path: str | None = None,
        shell_env: dict[str, str] | None = None,
    ) -> None:
        super().__init__(cwd)
        self._shell_path = shell_path
        self._shell_env = shell_env
        self._active_child_pids: set[int] = set()

    # ---- Shell 执行 ------------------------------------------------------

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]:
        """执行 shell 命令。"""
        opts = options or ShellExecOptions()

        # 检查 abort
        if opts.abort_signal is not None and opts.abort_signal.aborted:
            return err(ExecutionError("aborted", "aborted"))

        # 校验超时
        timeout_result = _resolve_timeout_ms(opts.timeout)
        if timeout_result.is_err():
            return err(timeout_result.error)
        timeout_ms = timeout_result.value

        # 解析 cwd
        cwd = _resolve_path(self.cwd, opts.cwd) if opts.cwd else self.cwd

        # 获取 shell 配置
        shell_result = await _get_shell_config(self._shell_path)
        if shell_result.is_err():
            return err(shell_result.error)
        shell_path, shell_args, _command_transport = shell_result.value

        # 校验 cwd 存在
        if not os.path.isdir(cwd):
            return err(
                ExecutionError(
                    "spawn_error",
                    f"Working directory does not exist: {cwd}\nCannot execute bash commands.",
                )
            )

        # 构建环境变量
        env = _get_shell_env(self._shell_env, opts.env, opts.inherit_env)

        return await self._exec_impl(
            command=command,
            shell_path=shell_path,
            shell_args=shell_args,
            cwd=cwd,
            env=env,
            timeout_ms=timeout_ms,
            on_stdout=opts.on_stdout,
            on_stderr=opts.on_stderr,
            abort_signal=opts.abort_signal,
        )

    async def _exec_impl(
        self,
        command: str,
        shell_path: str,
        shell_args: list[str],
        cwd: str,
        env: dict[str, str] | None,
        timeout_ms: int | None,
        on_stdout: Callable[[str], None] | None,
        on_stderr: Callable[[str], None] | None,
        abort_signal: CancellationToken | None,
    ) -> Result[ShellExecResult, ExecutionError]:
        """exec 核心实现。"""
        callback_error: ExecutionError | None = None
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                shell_path,
                *shell_args,
                command,
                cwd=cwd,
                env=env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except Exception as exc:
            cause = to_error(exc)
            return err(ExecutionError("spawn_error", str(cause), cause=cause))

        if proc.pid is not None:
            self._active_child_pids.add(proc.pid)

        async def _read_stream(
            stream: asyncio.StreamReader | None,
            callback: Callable[[str], None] | None,
            buf_list: list[str],
        ) -> None:
            if stream is None:
                return
            try:
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    text = line.decode("utf-8", errors="replace")
                    buf_list.append(text)
                    if callback:
                        try:
                            callback(text)
                        except Exception as exc:
                            nonlocal callback_error
                            callback_error = ExecutionError(
                                "callback_error", str(exc), cause=to_error(exc)
                            )
            except Exception:
                pass

        # 启动读取任务
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        read_tasks = [
            _read_stream(proc.stdout, on_stdout, stdout_lines),
            _read_stream(proc.stderr, on_stderr, stderr_lines),
        ]

        # 等待进程完成或超时
        try:
            proc_wait = asyncio.ensure_future(proc.wait())
            if timeout_ms is not None:
                timeout_sec = timeout_ms / 1000.0
                await asyncio.wait_for(proc_wait, timeout=timeout_sec)
            else:
                await proc_wait
        except asyncio.TimeoutError:
            timed_out = True
            if proc.pid is not None:
                _kill_process_tree(proc.pid)
        except asyncio.CancelledError:
            if proc.pid is not None:
                _kill_process_tree(proc.pid)
            raise

        # 检查 abort
        if abort_signal is not None and abort_signal.aborted:
            if proc.pid is not None:
                _kill_process_tree(proc.pid)
            # 等待读取任务完成
            await asyncio.gather(*read_tasks, return_exceptions=True)
            if proc.pid is not None:
                self._active_child_pids.discard(proc.pid)
            return err(ExecutionError("aborted", "aborted"))

        # 等待 stdio 缓冲 flush
        await asyncio.sleep(EXIT_STDIO_GRACE_SECONDS)
        await asyncio.gather(*read_tasks, return_exceptions=True)

        # 清理
        if proc.pid is not None:
            self._active_child_pids.discard(proc.pid)

        # 回调错误优先
        if callback_error is not None:
            return err(callback_error)

        if timed_out:
            return err(ExecutionError("timeout", f"timeout:{timeout_ms}ms"))

        stdout = "".join(stdout_lines)
        stderr = "".join(stderr_lines)

        return ok(
            ShellExecResult(
                stdout=stdout,
                stderr=stderr,
                exit_code=proc.returncode or 0,
            )
        )

    # ---- 清理 ------------------------------------------------------------

    async def cleanup(self) -> None:
        """清理所有活跃子进程。"""
        for pid in self._active_child_pids:
            _kill_process_tree(pid)
        self._active_child_pids.clear()
        await super().cleanup()
