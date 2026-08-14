"""Write 工具。

提供 ``createWriteTool`` 工厂函数，用于创建/覆盖文件。
"""

from __future__ import annotations

from pydantic import BaseModel

from ...cancellation import CancellationToken
from ...result import get_or_throw
from ...types import AgentToolResult, AgentToolUpdateCallback, TextContent
from ..types import AgentHarnessTool
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class WriteToolInput(BaseModel):
    """Write 工具输入参数。"""

    path: str
    content: str


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_write_tool() -> AgentHarnessTool:
    """创建 Write 工具。"""

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
        context: object,
    ) -> AgentToolResult:
        path = str(params["path"])
        content = str(params["content"])

        if not isinstance(context, ExecutionToolContext):
            raise TypeError("context must be ExecutionToolContext")
        env = context.env

        absolute_path = await resolve_tool_path(env, path, signal)

        async def _write() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            get_or_throw(await env.write_file(absolute_path, content, signal))
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully wrote {len(content)} bytes to {path}",
                    )
                ],
                details=None,
            )

        return await with_file_mutation_queue(env, absolute_path, _write)

    return AgentHarnessTool(
        name="write",
        label="write",
        description=(
            "Write content to a file. Creates the file if it doesn't exist, "
            "overwrites if it does. Automatically creates parent directories."
        ),
        parameters={"type": "object", "properties": {}},
        execute=_execute,
    )