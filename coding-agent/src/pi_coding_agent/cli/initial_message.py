"""初始消息构建（对应 TS ``cli/initial-message.ts``）。

从 stdin 内容、@file 文本和 CLI 消息构建初始 prompt。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pi_ai.types import ImageContent

    from .args import Args


class InitialMessageInput:
    """构建初始消息的输入。"""

    def __init__(
        self,
        *,
        parsed: Args,
        file_text: str | None = None,
        file_images: list[ImageContent] | None = None,
        stdin_content: str | None = None,
    ) -> None:
        self.parsed = parsed
        self.file_text = file_text
        self.file_images = file_images
        self.stdin_content = stdin_content


class InitialMessageResult:
    """构建初始消息的结果。"""

    def __init__(
        self,
        *,
        initial_message: str | None = None,
        initial_images: list[ImageContent] | None = None,
    ) -> None:
        self.initial_message = initial_message
        self.initial_images = initial_images


def build_initial_message(input_: InitialMessageInput) -> InitialMessageResult:
    """将 stdin 内容、@file 文本和第一条 CLI 消息合并为单个初始 prompt。

    用于非交互式模式。

    Args:
        input_: 构建初始消息的输入。

    Returns:
        包含初始消息和图片的结果。
    """
    parts: list[str] = []
    if input_.stdin_content is not None:
        parts.append(input_.stdin_content)
    if input_.file_text:
        parts.append(input_.file_text)

    if input_.parsed.messages:
        parts.append(input_.parsed.messages[0])
        input_.parsed.messages.pop(0)

    return InitialMessageResult(
        initial_message="".join(parts) if parts else None,
        initial_images=input_.file_images if input_.file_images else None,
    )


__all__ = [
    "InitialMessageInput",
    "InitialMessageResult",
    "build_initial_message",
]