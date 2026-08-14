"""消息格式转换"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

from ..types import (
    AssistantMessage,
    ImageContent,
    Message,
    Model,
    TextContent,
    ToolCallContent,
    ToolResultMessage,
    UserMessage,
)

NON_VISION_USER_IMAGE_PLACEHOLDER = "(image omitted: model does not support images)"
NON_VISION_TOOL_IMAGE_PLACEHOLDER = (
    "(tool image omitted: model does not support images)"
)


def _replace_images_with_placeholder(
    content: list[TextContent | ImageContent],
    placeholder: str,
) -> list[TextContent]:
    """将图片替换为占位文本。"""
    result: list[TextContent] = []
    previous_was_placeholder = False
    for block in content:
        if block.type == "image":
            if not previous_was_placeholder:
                result.append(TextContent(text=placeholder))
            previous_was_placeholder = True
            continue
        result.append(block)
        previous_was_placeholder = block.text == placeholder
    return result


def _downgrade_unsupported_images(
    messages: list[Message],
    model: Model,
) -> list[Message]:
    """降级不支持的图片消息。"""
    supports_image = "image" in getattr(model, "input_types", []) or "image" in getattr(
        model, "input", []
    )
    if supports_image:
        return messages

    result: list[Message] = []
    for msg in messages:
        if msg.role == "user":
            user_msg = msg
            if isinstance(user_msg.content, list) and any(
                getattr(c, "type", None) == "image" for c in user_msg.content
            ):
                text_content = _replace_images_with_placeholder(
                    [c for c in user_msg.content if c.type in ("text", "image")],
                    NON_VISION_USER_IMAGE_PLACEHOLDER,
                )
                result.append(user_msg.model_copy(update={"content": text_content}))
            else:
                result.append(msg)
        elif msg.role == "toolResult":
            tool_result = msg
            if isinstance(tool_result.content, list) and any(
                getattr(c, "type", None) == "image" for c in tool_result.content
            ):
                text_content = _replace_images_with_placeholder(
                    [c for c in tool_result.content if c.type in ("text", "image")],
                    NON_VISION_TOOL_IMAGE_PLACEHOLDER,
                )
                result.append(tool_result.model_copy(update={"content": text_content}))
            else:
                result.append(msg)
        else:
            result.append(msg)
    return result


def transform_messages(
    messages: list[Message],
    model: Model,
    normalize_tool_call_id: Callable[[str, Model, AssistantMessage], str] | None = None,
) -> list[Message]:
    """转换消息以适配不同模型格式。

    处理：图片降级、思考块转换、tool call ID 归一化、孤儿 tool call 修复。
    """
    # 构建 tool call ID 映射
    tool_call_id_map: dict[str, str] = {}

    # 规范化空内容
    normalized_messages: list[Message] = []
    for msg in messages:
        if (
            isinstance(msg, AssistantMessage)
            and not msg.content
            or isinstance(msg, ToolResultMessage)
            and not msg.content
            or isinstance(msg, UserMessage)
            and not msg.content
        ):
            normalized_messages.append(msg.model_copy(update={"content": []}))
        else:
            normalized_messages.append(msg)

    image_aware_messages = _downgrade_unsupported_images(normalized_messages, model)

    # 第一遍：转换消息
    transformed: list[Message] = []
    for msg in image_aware_messages:
        if isinstance(msg, UserMessage):
            transformed.append(msg)
        elif isinstance(msg, ToolResultMessage):
            normalized_id = tool_call_id_map.get(msg.tool_call_id)
            if normalized_id and normalized_id != msg.tool_call_id:
                transformed.append(
                    msg.model_copy(update={"tool_call_id": normalized_id})
                )
            else:
                transformed.append(msg)
        elif isinstance(msg, AssistantMessage):
            is_same_model = (
                msg.provider == model.provider
                and msg.api == model.api
                and msg.model == model.model_id
            )

            # 转换内容块（使用 Any 避免 ContentBlock 类型限制）
            content_blocks = cast("list[Any]", msg.content)
            transformed_content: list[Any] = []
            for block in content_blocks:
                block_type = getattr(block, "type", None)
                if block_type == "thinking":
                    # 红acted 思考块仅对同一模型有效
                    if getattr(block, "redacted", False):
                        if is_same_model:
                            transformed_content.append(block)
                        continue
                    # 同一模型：保留有签名的思考块
                    if is_same_model and getattr(block, "signature", None):
                        transformed_content.append(block)
                        continue
                    # 跳过空思考块
                    if (
                        not getattr(block, "text", "")
                        or getattr(block, "text", "").strip() == ""
                    ):
                        continue
                    if is_same_model:
                        transformed_content.append(block)
                    else:
                        # 跨模型时转为文本
                        transformed_content.append(
                            TextContent(text=getattr(block, "text", ""))
                        )
                elif block_type == "text":
                    if is_same_model:
                        transformed_content.append(block)
                    else:
                        transformed_content.append(TextContent(text=block.text))
                elif block_type == "toolCall":
                    tool_call = cast(ToolCallContent, block)
                    normalized_tool_call = tool_call

                    if not is_same_model and getattr(
                        tool_call, "thought_signature", None
                    ):
                        normalized_tool_call = tool_call.model_copy(
                            update={"thought_signature": None}
                        )

                    if not is_same_model and normalize_tool_call_id:
                        normalized_id = normalize_tool_call_id(
                            tool_call.tool_call_id, model, msg
                        )
                        if normalized_id != tool_call.tool_call_id:
                            tool_call_id_map[tool_call.tool_call_id] = normalized_id
                            normalized_tool_call = normalized_tool_call.model_copy(
                                update={"tool_call_id": normalized_id}
                            )

                    transformed_content.append(normalized_tool_call)
                else:
                    transformed_content.append(block)

            transformed.append(msg.model_copy(update={"content": transformed_content}))
        else:
            transformed.append(msg)

    # 第二遍：插入合成工具结果
    result: list[Message] = []
    pending_tool_calls: list[ToolCallContent] = []
    existing_tool_result_ids: set[str] = set()

    def insert_synthetic_tool_results() -> None:
        nonlocal pending_tool_calls, existing_tool_result_ids
        if pending_tool_calls:
            import time

            for tc in pending_tool_calls:
                if tc.tool_call_id not in existing_tool_result_ids:
                    result.append(
                        ToolResultMessage(
                            tool_call_id=tc.tool_call_id,
                            tool_name=tc.name,
                            content=[TextContent(text="No result provided")],
                            is_error=True,
                            timestamp=int(time.time() * 1000),
                        )
                    )
            pending_tool_calls = []
            existing_tool_result_ids = set()

    for msg in transformed:
        if isinstance(msg, AssistantMessage):
            insert_synthetic_tool_results()

            if msg.stop_reason in ("error", "aborted"):
                continue

            content_blocks = cast("list[Any]", msg.content)
            tool_calls = [
                b for b in content_blocks if getattr(b, "type", None) == "toolCall"
            ]
            if tool_calls:
                pending_tool_calls = cast("list[ToolCallContent]", tool_calls)
                existing_tool_result_ids = set()

            result.append(msg)
        elif isinstance(msg, ToolResultMessage):
            existing_tool_result_ids.add(msg.tool_call_id)
            result.append(msg)
        elif isinstance(msg, UserMessage):
            insert_synthetic_tool_results()
            result.append(msg)
        else:
            result.append(msg)

    insert_synthetic_tool_results()
    return result
