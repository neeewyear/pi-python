"""文件系统操作实现（对应 ``harness/env/nodejs.ts`` 的 FileSystem 部分）。

使用 ``aiofiles`` 实现异步文件 IO，所有方法返回 ``Result`` 包装。
"""

from __future__ import annotations

import errno
import os
import shutil
import stat
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import aiofiles
import aiofiles.os

from ...cancellation import CancellationToken
from ...result import Result, err, ok, to_error
from ..types import (
    FileError,
    FileInfo,
    FileKind,
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _resolve_path(cwd: str, path: str) -> str:
    """路径解析（对应 TS ``resolvePath``）。

    - ``~`` / ``~/...`` → 展开为用户目录
    - ``file://`` URL → 提取路径
    - 相对路径 → 基于 cwd 解析
    """
    normalized = path

    # 展开 ~
    if normalized == "~":
        normalized = str(Path.home())
    elif normalized.startswith("~/"):
        normalized = str(Path.home() / normalized[2:])

    # 处理 file:// URL
    if normalized.startswith("file://"):
        try:
            parsed = urlparse(normalized)
            normalized = parsed.path
        except Exception:
            pass

    # 绝对路径直接返回，否则基于 cwd 拼接
    if os.path.isabs(normalized):
        return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(cwd, normalized))


def _to_file_error(error: object, path: str | None = None) -> FileError:
    """将 Python 异常映射为 ``FileError``（对应 TS ``toFileError``）。"""
    if isinstance(error, FileError):
        return error

    cause = to_error(error)

    if isinstance(error, OSError):
        errno_code = getattr(error, "errno", None)
        if errno_code == errno.ENOENT:
            return FileError("not_found", str(error), path, cause)
        if errno_code in (errno.EACCES, errno.EPERM):
            return FileError("permission_denied", str(error), path, cause)
        if errno_code == errno.ENOTDIR:
            return FileError("not_directory", str(error), path, cause)
        if errno_code == errno.EISDIR:
            return FileError("is_directory", str(error), path, cause)
        if errno_code == errno.EINVAL:
            return FileError("invalid", str(error), path, cause)

    return FileError("unknown", str(cause), path, cause)


def _abort_result(
    token: CancellationToken | None, path: str | None = None
) -> Result[object, FileError] | None:
    """检查令牌是否已取消（对应 TS ``abortResult``）。"""
    if token is not None and token.aborted:
        return err(FileError("aborted", "aborted", path))
    return None


def _file_info_from_stats(path: str, st: os.stat_result) -> Result[FileInfo, FileError]:
    """从 ``os.stat_result`` 构造 ``FileInfo``（对应 TS ``fileInfoFromStats``）。"""
    # 判断文件类型
    if stat.S_ISREG(st.st_mode):
        kind: FileKind = "file"
    elif stat.S_ISDIR(st.st_mode):
        kind = "directory"
    elif stat.S_ISLNK(st.st_mode):
        kind = "symlink"
    else:
        return err(FileError("invalid", "Unsupported file type", path))

    return ok(
        FileInfo(
            name=os.path.basename(path),
            path=path,
            kind=kind,
            size=st.st_size,
            mtime_ms=int(st.st_mtime * 1000),
        )
    )


# ---------------------------------------------------------------------------
# NodeFileSystem
# ---------------------------------------------------------------------------


class NodeFileSystem:
    """文件系统实现（对应 TS ``NodeExecutionEnv`` 的 FileSystem 部分）。

    实现 ``FileSystem`` 协议，所有方法返回 ``Result`` 包装。
    """

    def __init__(self, cwd: str) -> None:
        self.cwd = cwd

    # ---- 路径操作 --------------------------------------------------------

    async def absolute_path(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]:
        return ok(_resolve_path(self.cwd, path))

    async def join_path(
        self, parts: list[str], abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]:
        return ok(os.path.join(*parts))

    # ---- 读取 ------------------------------------------------------------

    async def read_text_file(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted  # type: ignore[return-value]
        try:
            async with aiofiles.open(resolved, encoding="utf-8") as f:
                return ok(await f.read())
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def read_text_lines(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[list[str], FileError]:
        resolved = _resolve_path(self.cwd, path)
        max_lines = options.get("maxLines") if options else None
        abort_signal = options.get("abortSignal") if options else None

        if max_lines is not None and isinstance(max_lines, int) and max_lines <= 0:
            return ok([])

        try:
            async with aiofiles.open(resolved, encoding="utf-8") as f:
                lines: list[str] = []
                async for line in f:
                    if abort_signal is not None and isinstance(
                        abort_signal, CancellationToken
                    ):
                        aborted = _abort_result(abort_signal, resolved)
                        if aborted is not None:
                            return aborted  # type: ignore[return-value]
                    lines.append(line.rstrip("\n").rstrip("\r"))
                    if (
                        max_lines is not None
                        and isinstance(max_lines, int)
                        and len(lines) >= max_lines
                    ):
                        break
                return ok(lines)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def read_binary_file(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[bytes, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted  # type: ignore[return-value]
        try:
            async with aiofiles.open(resolved, mode="rb") as f:
                return ok(await f.read())
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    # ---- 写入 ------------------------------------------------------------

    async def write_file(
        self,
        path: str,
        content: str | bytes,
        abort_signal: CancellationToken | None = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted  # type: ignore[return-value]
        try:
            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)
            mode = "wb" if isinstance(content, bytes) else "w"
            encoding = None if isinstance(content, bytes) else "utf-8"
            async with aiofiles.open(resolved, mode=mode, encoding=encoding) as f:  # type: ignore[call-overload]
                await f.write(content)
            return ok(None)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def append_file(
        self,
        path: str,
        content: str | bytes,
        abort_signal: CancellationToken | None = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            parent = os.path.dirname(resolved)
            if parent:
                os.makedirs(parent, exist_ok=True)
            mode = "ab" if isinstance(content, bytes) else "a"
            encoding = None if isinstance(content, bytes) else "utf-8"
            async with aiofiles.open(resolved, mode=mode, encoding=encoding) as f:  # type: ignore[call-overload]
                await f.write(content)
            return ok(None)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    # ---- 元数据 ----------------------------------------------------------

    async def file_info(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[FileInfo, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            st = os.lstat(resolved)
            return _file_info_from_stats(resolved, st)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def list_dir(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[list[FileInfo], FileError]:
        resolved = _resolve_path(self.cwd, path)
        aborted = _abort_result(abort_signal, resolved)
        if aborted is not None:
            return aborted  # type: ignore[return-value]
        try:
            infos: list[FileInfo] = []
            with os.scandir(resolved) as it:
                for entry in it:
                    loop_abort = _abort_result(abort_signal, resolved)
                    if loop_abort is not None:
                        return loop_abort  # type: ignore[return-value]
                    try:
                        st = entry.stat(follow_symlinks=False)
                        info_result = _file_info_from_stats(entry.path, st)
                        if info_result.is_ok():
                            infos.append(info_result.value)
                    except Exception as exc:
                        return err(_to_file_error(exc, entry.path))
            return ok(infos)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def canonical_path(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]:
        resolved = _resolve_path(self.cwd, path)
        try:
            return ok(os.path.realpath(resolved))
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def exists(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[bool, FileError]:
        result = await self.file_info(path)
        if result.is_ok():
            return ok(True)
        if result.error.code == "not_found":
            return ok(False)
        return err(result.error)

    # ---- 目录 / 文件操作 -------------------------------------------------

    async def create_dir(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        recursive = options.get("recursive", True) if options else True
        try:
            os.makedirs(resolved, exist_ok=bool(recursive))
            return ok(None)
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def remove(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[None, FileError]:
        resolved = _resolve_path(self.cwd, path)
        recursive = options.get("recursive", False) if options else False
        force = options.get("force", False) if options else False
        try:
            if os.path.isdir(resolved) and not os.path.islink(resolved):
                if recursive:
                    shutil.rmtree(resolved)
                else:
                    os.rmdir(resolved)
            else:
                os.unlink(resolved)
            return ok(None)
        except FileNotFoundError:
            if force:
                return ok(None)
            return err(
                _to_file_error(FileNotFoundError(f"No such file: {resolved}"), resolved)
            )
        except Exception as exc:
            return err(_to_file_error(exc, resolved))

    async def create_temp_dir(
        self, prefix: str | None = None, abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]:
        try:
            pfx = prefix if prefix else "tmp-"
            result = tempfile.mkdtemp(prefix=pfx)
            return ok(result)
        except Exception as exc:
            return err(_to_file_error(exc))

    async def create_temp_file(
        self, options: dict[str, object] | None = None
    ) -> Result[str, FileError]:
        prefix = str(options.get("prefix", "")) if options else ""
        suffix = str(options.get("suffix", "")) if options else ""
        try:
            fd, path = tempfile.mkstemp(prefix=prefix, suffix=suffix)
            os.close(fd)
            return ok(path)
        except Exception as exc:
            return err(_to_file_error(exc))

    # ---- 清理 ------------------------------------------------------------

    async def cleanup(self) -> None:
        """清理（文件系统无需额外清理）。"""
        return
