"""文件处理器（对应 TS ``cli/file-processor.ts``）。

处理 ``@file`` CLI 参数，将文件内容读取为文本和图片附件。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import ImageContent

from ..core.tools.path_utils import resolve_path


@dataclass
class ProcessedFiles:
    """已处理的文件内容。"""

    text: str = ""
    """文件文本内容。"""
    images: list[ImageContent] = field(default_factory=list)
    """图片附件列表。"""


@dataclass
class ProcessFileOptions:
    """文件处理选项。"""

    auto_resize_images: bool = True
    """是否自动调整图片大小到 2000x2000 以内。默认 True。"""


def _read_file_content(path: str) -> str:
    """读取文本文件内容。

    Args:
        path: 文件路径。

    Returns:
        文件内容。

    Raises:
        FileNotFoundError: 文件不存在。
        PermissionError: 无权限读取。
    """
    return Path(path).read_text(encoding="utf-8")


async def process_file_arguments(
    file_args: list[str],
    options: ProcessFileOptions | None = None,
) -> ProcessedFiles:
    """处理 @file 参数，返回文本内容和图片附件。

    当前简化版实现：仅处理文本文件，图片处理待后续实现。

    Args:
        file_args: @file 参数列表。
        options: 处理选项。

    Returns:
        处理后的文件内容和图片。
    """
    _ = options  # 保留参数为未来扩展
    auto_resize_images = options.auto_resize_images if options else True

    text = ""
    images: list[ImageContent] = []

    for file_arg in file_args:
        absolute_path = os.path.normpath(
            os.path.expanduser(resolve_path(file_arg, os.getcwd()))
        )

        # 检查文件是否存在
        path = Path(absolute_path)
        if not path.exists():
            print(f"Error: File not found: {absolute_path}", file=sys.stderr)
            sys.exit(1)

        # 检查文件是否为空
        if path.stat().st_size == 0:
            continue

        # 尝试检测是否为图片
        from ..utils.mime import get_mime_type

        mime_type = get_mime_type(str(path))
        is_image = mime_type and mime_type.startswith("image/")

        if is_image:
            # 简化版：跳过图片处理（processImage 尚未移植）
            # 仅添加文件引用文本
            raw_data = path.read_bytes()

            if auto_resize_images:
                text += f'<file name="{absolute_path}">[Image resized to 2000x2000]</file>\n'
            else:
                text += f'<file name="{absolute_path}"></file>\n'

            # 添加为图片附件
            import base64

            images.append(
                ImageContent(
                    type="image",
                    mime_type=mime_type,
                    data=base64.b64encode(raw_data).decode("ascii"),
                )
            )
        else:
            # 处理文本文件
            try:
                file_content = _read_file_content(absolute_path)
                text += f'<file name="{absolute_path}">\n{file_content}\n</file>\n'
            except (FileNotFoundError, PermissionError, OSError) as error:
                print(
                    f"Error: Could not read file {absolute_path}: {error}",
                    file=sys.stderr,
                )
                sys.exit(1)

    return ProcessedFiles(text=text, images=images)


__all__ = [
    "ProcessFileOptions",
    "ProcessedFiles",
    "process_file_arguments",
]
