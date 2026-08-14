"""harness 基础设施抽象。

包含 ``Result`` 工具函数、``Skill`` / ``PromptTemplate``、文件与执行环境
抽象（``FileSystem`` / ``Shell`` / ``ExecutionEnv``）以及各域错误类型。
"""

from __future__ import annotations

from typing import Awaitable, Callable, ClassVar, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict

from ..cancellation import CancellationToken
from ..result import AgentError, Result, err, get_or_throw, get_or_undefined, ok, to_error
from ..types import AgentToolResult, AgentToolUpdateCallback, SimpleStreamOptions, Tool, Usage

__all__ = [
    "AgentHarnessResources",
    "AgentHarnessStreamOptions",
    "AgentHarnessStreamOptionsPatch",
    "AgentHarnessTool",
    "AgentHarnessToolContextSource",
    "BranchSummaryError",
    "BranchSummaryErrorCode",
    "CompactionError",
    "CompactionErrorCode",
    "ExecutionEnv",
    "ExecutionError",
    "ExecutionErrorCode",
    "FileError",
    "FileErrorCode",
    "FileInfo",
    "FileKind",
    "FileSystem",
    "PromptTemplate",
    "Shell",
    "ShellExecOptions",
    "Skill",
    "err",
    "get_or_throw",
    "get_or_undefined",
    "ok",
    "to_error",
]

# ---------------------------------------------------------------------------
# 文件与执行错误
# ---------------------------------------------------------------------------

FileErrorCode: TypeAlias = Literal[
    "aborted",
    "not_found",
    "permission_denied",
    "not_directory",
    "is_directory",
    "invalid",
    "not_supported",
    "unknown",
]
"""后端无关的文件错误码。"""


class FileError(AgentError):
    """文件系统操作错误。"""

    code: ClassVar[FileErrorCode] = "unknown"

    def __init__(self, code: FileErrorCode, message: str, path: str | None = None, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
        object.__setattr__(self, "code", code)
        self.path = path


ExecutionErrorCode: TypeAlias = Literal[
    "aborted",
    "timeout",
    "shell_unavailable",
    "spawn_error",
    "callback_error",
    "unknown",
]
"""执行错误码。"""


class ExecutionError(AgentError):
    """shell 执行错误。"""

    code: ClassVar[ExecutionErrorCode] = "unknown"

    def __init__(self, code: ExecutionErrorCode, message: str, *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
        object.__setattr__(self, "code", code)


CompactionErrorCode: TypeAlias = Literal["aborted", "summarization_failed"]


class CompactionError(AgentError):
    """压缩错误。"""

    code: ClassVar[CompactionErrorCode] = "summarization_failed"

    def __init__(self, code: str | None = None, message: str = "", *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
        if code is not None:
            object.__setattr__(self, "code", code)


BranchSummaryErrorCode: TypeAlias = Literal["aborted", "summarization_failed"]


class BranchSummaryError(AgentError):
    """分支摘要错误。"""

    code: ClassVar[BranchSummaryErrorCode] = "summarization_failed"

    def __init__(self, code: str | None = None, message: str = "", *, cause: BaseException | None = None) -> None:
        super().__init__(message, cause=cause)
        if code is not None:
            object.__setattr__(self, "code", code)


# ---------------------------------------------------------------------------
# 文件信息
# ---------------------------------------------------------------------------

FileKind: TypeAlias = Literal["file", "directory", "symlink"]
"""文件系统对象类型。符号链接默认不自动跟随。"""


class FileInfo(BaseModel):
    """一个文件系统对象的元数据。"""

    name: str
    path: str
    kind: FileKind
    size: int
    mtime_ms: int


# ---------------------------------------------------------------------------
# 资源（Skill / PromptTemplate）
# ---------------------------------------------------------------------------

class Skill(BaseModel):
    """从 ``SKILL.md`` 加载或由应用提供的技能。

    ``name`` / ``description`` / ``file_path`` 会以 XML 块形式插入系统提示词
    （见 ``system_prompt.py``）。
    """

    name: str
    description: str
    content: str
    file_path: str
    disable_model_invocation: bool = False


class PromptTemplate(BaseModel):
    """可被显式调用格式化的提示词模板。"""

    name: str
    description: str | None = None
    content: str


class AgentHarnessResources(BaseModel):
    """显式调用方法与系统提示词回调可用的资源。"""

    prompt_templates: list[PromptTemplate] | None = None
    skills: list[Skill] | None = None


# ---------------------------------------------------------------------------
# stream options
# ---------------------------------------------------------------------------

class AgentHarnessStreamOptions(BaseModel):
    """harness 持有并在每回合快照的 provider 请求选项。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: object | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, object] | None = None
    cache_retention: str | None = None


class AgentHarnessStreamOptionsPatch(BaseModel):
    """provider 钩子返回的逐字段 patch。

    ``headers`` / ``metadata`` 中值为 ``None`` 表示删除该键；显式传空 dict 表示清空。
    """

    transport: object | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str | None] | None = None
    metadata: dict[str, object | None] | None = None
    cache_retention: str | None = None


# ---------------------------------------------------------------------------
# AgentHarnessTool
# ---------------------------------------------------------------------------

class AgentHarnessTool(Tool):
    """由 ``AgentHarness`` 执行、带应用定义上下文的工具。

    ``execute`` 额外接收当前回合快照解析出的 ``context``。
    """

    label: str
    execute: Callable[
        [str, dict[str, object], CancellationToken | None, AgentToolUpdateCallback | None, object],
        Awaitable[AgentToolResult],
    ]


AgentHarnessToolContextSource: TypeAlias = object | Callable[[], object | Awaitable[object]]
"""静态工具上下文或零参提供器（每个回合快照解析一次）。"""


# ---------------------------------------------------------------------------
# FileSystem / Shell / ExecutionEnv
# ---------------------------------------------------------------------------

class FileSystem(Protocol):
    """harness 使用的文件系统能力。

    约定：**方法永不抛异常**，所有失败（含意外后端错误）编码进返回的
    ``Result``。
    """

    cwd: str

    async def absolute_path(self, path: str, abort_signal: CancellationToken | None = None) -> Result[str, FileError]: ...
    async def join_path(self, parts: list[str], abort_signal: CancellationToken | None = None) -> Result[str, FileError]: ...
    async def read_text_file(self, path: str, abort_signal: CancellationToken | None = None) -> Result[str, FileError]: ...
    async def read_text_lines(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[list[str], FileError]: ...
    async def read_binary_file(
        self, path: str, abort_signal: CancellationToken | None = None
    ) -> Result[bytes, FileError]: ...
    async def write_file(
        self, path: str, content: str | bytes, abort_signal: CancellationToken | None = None
    ) -> Result[None, FileError]: ...
    async def append_file(
        self, path: str, content: str | bytes, abort_signal: CancellationToken | None = None
    ) -> Result[None, FileError]: ...
    async def file_info(self, path: str, abort_signal: CancellationToken | None = None) -> Result[FileInfo, FileError]: ...
    async def list_dir(self, path: str, abort_signal: CancellationToken | None = None) -> Result[list[FileInfo], FileError]: ...
    async def canonical_path(self, path: str, abort_signal: CancellationToken | None = None) -> Result[str, FileError]: ...
    async def exists(self, path: str, abort_signal: CancellationToken | None = None) -> Result[bool, FileError]: ...
    async def create_dir(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[None, FileError]: ...
    async def remove(
        self,
        path: str,
        options: dict[str, object] | None = None,
    ) -> Result[None, FileError]: ...
    async def create_temp_dir(
        self, prefix: str | None = None, abort_signal: CancellationToken | None = None
    ) -> Result[str, FileError]: ...
    async def create_temp_file(
        self, options: dict[str, object] | None = None
    ) -> Result[str, FileError]: ...
    async def cleanup(self) -> None: ...


class ShellExecOptions(BaseModel):
    """``Shell.exec`` 选项。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    cwd: str | None = None
    env: dict[str, str] | None = None
    inherit_env: bool = True
    timeout: int | None = None
    abort_signal: CancellationToken | None = None
    on_stdout: Callable[[str], None] | None = None
    on_stderr: Callable[[str], None] | None = None


class ShellExecResult(BaseModel):
    """``Shell.exec`` 成功值。"""

    stdout: str
    stderr: str
    exit_code: int


class Shell(Protocol):
    """shell 执行能力。"""

    async def exec(
        self, command: str, options: ShellExecOptions | None = None
    ) -> Result[ShellExecResult, ExecutionError]: ...
    async def cleanup(self) -> None: ...


class ExecutionEnv(FileSystem, Shell, Protocol):
    """文件系统 + 进程执行环境。"""
