"""Read 工具。

提供 ``createReadTool`` 工厂函数，支持文本文件读取（offset/limit + 截断）
与图片文件读取（MIME 检测 + Base64 编码）。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from ...cancellation import CancellationToken
from ...result import get_or_throw
from ...types import AgentToolResult, AgentToolUpdateCallback, ImageContent, TextContent
from ..types import AgentHarnessTool
from ..utils.truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationResult,
    format_size,
    truncate_head,
)
from .image import detect_supported_image_mime_type, encode_base64
from .path_utils import resolve_read_tool_path
from .tool_context import ExecutionToolContext

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class ReadToolInput(BaseModel):
    """Read 工具输入参数。"""

    path: str
    offset: int | None = None
    limit: int | None = None


class ReadToolDetails(BaseModel):
    """Read 工具输出详情。"""

    truncation: TruncationResult | None = None


ReadImageProcessorResult = dict[str, object]
"""图片处理器结果。"""

ReadImageProcessor = Callable[
    [bytes, str, dict[str, object]],
    Awaitable[ReadImageProcessorResult],
]
"""图片处理器。"""


class ReadToolOptions(BaseModel):
    """Read 工具选项。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    auto_resize_images: bool = True
    image_processor: ReadImageProcessor | None = None


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_read_tool(
    options: ReadToolOptions | None = None,
) -> AgentHarnessTool:
    """创建 Read 工具。"""
    opts = options or ReadToolOptions()

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
        context: object,
    ) -> AgentToolResult:
        path = str(params["path"])
        offset_val = params.get("offset")
        offset: int | None = None
        if offset_val is not None:
            if isinstance(offset_val, (int, float)) or isinstance(offset_val, str):
                offset = int(offset_val)
        limit_val = params.get("limit")
        limit: int | None = None
        if limit_val is not None:
            if isinstance(limit_val, (int, float)) or isinstance(limit_val, str):
                limit = int(limit_val)

        if not isinstance(context, ExecutionToolContext):
            raise TypeError("context must be ExecutionToolContext")
        env = context.env

        absolute_path = await resolve_read_tool_path(env, path, signal)
        bytes_result = await env.read_binary_file(absolute_path, signal)
        file_bytes = get_or_throw(bytes_result)
        mime_type = detect_supported_image_mime_type(file_bytes)

        # ── 图片处理 ────────────────────────────────────────────────
        if mime_type is not None:
            if opts.image_processor is not None:
                processed = await opts.image_processor(
                    file_bytes,
                    mime_type,
                    {"auto_resize_images": opts.auto_resize_images},
                )
                ok_flag = processed.get("ok", False)
                if not ok_flag:
                    return AgentToolResult(
                        content=[
                            TextContent(
                                type="text",
                                text=(
                                    f"Read image file [{mime_type}]\n"
                                    f"{processed.get('message', '')}"
                                ),
                            )
                        ],
                        details=None,
                    )
                hints = processed.get("hints", [])
                hints_text = (
                    "\n" + "\n".join(hints) if isinstance(hints, list) and hints else ""
                )
                proc_mime = str(processed.get("mime_type", mime_type))
                return AgentToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=f"Read image file [{proc_mime}]{hints_text}",
                        ),
                        ImageContent(
                            type="image",
                            data=str(processed["data"]),
                            mime_type=proc_mime,
                        ),
                    ],
                    details=None,
                )
            if mime_type == "image/bmp":
                return AgentToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=(
                                "Read image file [image/bmp]\n"
                                "[Image omitted: configure an imageProcessor "
                                "to convert BMP images.]"
                            ),
                        )
                    ],
                    details=None,
                )
            return AgentToolResult(
                content=[
                    TextContent(type="text", text=f"Read image file [{mime_type}]"),
                    ImageContent(
                        type="image",
                        data=encode_base64(file_bytes),
                        mime_type=mime_type,
                    ),
                ],
                details=None,
            )

        # ── 文本处理 ────────────────────────────────────────────────
        text_content = file_bytes.decode("utf-8")
        all_lines = text_content.split("\n")
        total_file_lines = len(all_lines)
        start_line = max(0, offset - 1) if offset is not None else 0
        start_line_display = start_line + 1

        if start_line >= len(all_lines):
            raise RuntimeError(
                f"Offset {offset} is beyond end of file ({all_lines} lines total)"
            )

        if limit is not None:
            end_line = min(start_line + limit, len(all_lines))
            selected_content = "\n".join(all_lines[start_line:end_line])
            user_limited_lines = end_line - start_line
        else:
            selected_content = "\n".join(all_lines[start_line:])
            user_limited_lines = None

        truncation = truncate_head(selected_content)
        details: ReadToolDetails | None = None

        if truncation.first_line_exceeds_limit:
            first_line_size = format_size(len(all_lines[start_line].encode("utf-8")))
            output_text = (
                f"[Line {start_line_display} is {first_line_size}, "
                f"exceeds {format_size(DEFAULT_MAX_BYTES)} limit. "
                f"Use bash: sed -n '{start_line_display}p' {path} | "
                f"head -c {DEFAULT_MAX_BYTES}]"
            )
            details = ReadToolDetails(truncation=truncation)
        elif truncation.truncated:
            end_line_display = start_line_display + truncation.output_lines - 1
            next_offset = end_line_display + 1
            output_text = truncation.content
            if truncation.truncated_by == "lines":
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines}. Use offset={next_offset} to continue.]"
                )
            else:
                output_text += (
                    f"\n\n[Showing lines {start_line_display}-{end_line_display} "
                    f"of {total_file_lines} ({format_size(DEFAULT_MAX_BYTES)} limit). "
                    f"Use offset={next_offset} to continue.]"
                )
            details = ReadToolDetails(truncation=truncation)
        elif user_limited_lines is not None and start_line + user_limited_lines < len(
            all_lines
        ):
            remaining = len(all_lines) - (start_line + user_limited_lines)
            next_offset = start_line + user_limited_lines + 1
            output_text = (
                f"{truncation.content}\n\n"
                f"[{remaining} more lines in file. Use offset={next_offset} to continue.]"
            )
        else:
            output_text = truncation.content

        return AgentToolResult(
            content=[TextContent(type="text", text=output_text)],
            details=details,
        )

    return AgentHarnessTool(
        name="read",
        label="read",
        description=(
            f"Read the contents of a file. Supports text files and images "
            f"(jpg, png, gif, webp, bmp). Images are sent as attachments. "
            f"For text files, output is truncated to {DEFAULT_MAX_LINES} lines "
            f"or {DEFAULT_MAX_BYTES // 1024}KB (whichever is hit first). "
            f"Use offset/limit for large files."
        ),
        parameters={"type": "object", "properties": {}},
        execute=_execute,
    )
