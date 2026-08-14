"""Edit 工具。

提供 ``createEditTool`` 工厂函数，支持精确文本替换 + 模糊匹配 +
Unified Patch 生成 + 文件变更队列串行化。
"""

from __future__ import annotations

from pydantic import BaseModel

from ...cancellation import CancellationToken
from ...types import AgentToolResult, AgentToolUpdateCallback, TextContent
from ..types import AgentHarnessTool, FileError
from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_tool_path
from .tool_context import ExecutionToolContext

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class ReplaceEdit(BaseModel):
    """一次替换编辑。"""

    old_text: str
    new_text: str


class EditToolInput(BaseModel):
    """Edit 工具输入参数。"""

    path: str
    edits: list[ReplaceEdit]


class EditToolDetails(BaseModel):
    """Edit 工具输出详情。"""

    diff: str
    patch: str
    first_changed_line: int | None = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _prepare_edit_arguments(input_data: dict[str, object]) -> dict[str, object]:
    """预处理编辑参数。

    处理 legacy 格式（oldText/newText）和 JSON 字符串 edits。
    """
    args = dict(input_data)

    edits_raw = args.get("edits")
    if isinstance(edits_raw, str):
        import json

        try:
            parsed = json.loads(edits_raw)
            if isinstance(parsed, list):
                args["edits"] = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    old_text = args.get("oldText")
    new_text = args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        edits_raw2 = args.get("edits")
        edits: list[dict[str, object]] = (
            list(edits_raw2) if isinstance(edits_raw2, list) else []
        )
        edits.append({"oldText": old_text, "newText": new_text})
        args.pop("oldText", None)
        args.pop("newText", None)
        args["edits"] = edits

    return args


def _validate_edit_input(input_data: dict[str, object]) -> tuple[str, list[Edit]]:
    """校验并提取编辑参数。"""
    edits_raw = input_data.get("edits")
    if not isinstance(edits_raw, list) or len(edits_raw) == 0:
        raise ValueError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )
    edits = [
        Edit(
            old_text=str(e.get("oldText", e.get("old_text", ""))),
            new_text=str(e.get("newText", e.get("new_text", ""))),
        )
        for e in edits_raw
        if isinstance(e, dict)
    ]
    return str(input_data["path"]), edits


def _edit_access_error(path: str, error: FileError) -> RuntimeError:
    return RuntimeError(
        f"Could not edit file: {path}. Error code: {error.code}.",
    )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_edit_tool() -> AgentHarnessTool:
    """创建 Edit 工具。"""

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
        context: object,
    ) -> AgentToolResult:
        args = _prepare_edit_arguments(params)
        path, edits = _validate_edit_input(args)

        if not isinstance(context, ExecutionToolContext):
            raise TypeError("context must be ExecutionToolContext")
        env = context.env

        absolute_path = await resolve_tool_path(env, path, signal)

        async def _edit() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            info = await env.file_info(absolute_path, signal)
            if not info.is_ok():
                raise _edit_access_error(path, info.error)
            if info.value.kind not in ("file", "symlink"):
                raise RuntimeError(f"Could not edit file: {path}. Path is not a file.")

            read_result = await env.read_text_file(absolute_path, signal)
            if not read_result.is_ok():
                raise _edit_access_error(path, read_result.error)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            stripped = strip_bom(read_result.value)
            original_ending = detect_line_ending(stripped["text"])
            normalized_content = normalize_to_lf(stripped["text"])
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            final_content = stripped["bom"] + restore_line_endings(
                applied.new_content, original_ending
            )
            write_result = await env.write_file(absolute_path, final_content, signal)
            if not write_result.is_ok():
                raise _edit_access_error(path, write_result.error)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            diff_result = generate_diff_string(
                applied.base_content, applied.new_content
            )
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully replaced {len(edits)} block(s) in {path}.",
                    )
                ],
                details=EditToolDetails(
                    diff=str(diff_result["diff"]),
                    patch=generate_unified_patch(
                        path, applied.base_content, applied.new_content
                    ),
                    first_changed_line=(
                        int(diff_result["firstChangedLine"])  # type: ignore[call-overload]
                        if diff_result.get("firstChangedLine") is not None
                        else None
                    ),
                ),
            )

        return await with_file_mutation_queue(env, absolute_path, _edit)

    return AgentHarnessTool(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every "
            "edits[].oldText must match a unique, non-overlapping region of "
            "the original file. If two changes affect the same block or "
            "nearby lines, merge them into one edit instead of emitting "
            "overlapping edits."
        ),
        parameters={"type": "object", "properties": {}},
        prepare_arguments=_prepare_edit_arguments,
        execute=_execute,
    )
