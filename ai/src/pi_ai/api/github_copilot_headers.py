"""GitHub Copilot 请求头"""

from __future__ import annotations

from typing import cast

from ..types import Message


def infer_copilot_initiator(messages: list[Message]) -> str:
    """推断 Copilot 发起者。"""
    if not messages:
        return "user"
    last = messages[-1]
    return "agent" if last.role != "user" else "user"


def has_copilot_vision_input(messages: list[Message]) -> bool:
    """检查是否有 Copilot 视觉输入。"""
    for msg in messages:
        if msg.role == "user" and hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if getattr(c, "type", None) == "image":
                        return True
        if msg.role == "toolResult" and hasattr(msg, "content"):
            content = msg.content
            if isinstance(content, list):
                for c in content:
                    if getattr(c, "type", None) == "image":
                        return True
    return False


def build_copilot_dynamic_headers(params: dict[str, object]) -> dict[str, str]:
    """构建 Copilot 动态请求头。"""
    messages = params.get("messages", [])
    messages_list = messages if isinstance(messages, list) else []
    headers: dict[str, str] = {
        "X-Initiator": infer_copilot_initiator(cast("list[Message]", messages_list)),
        "Openai-Intent": "conversation-edits",
    }
    has_images = params.get("has_images", False)
    if isinstance(has_images, bool) and has_images:
        headers["Copilot-Vision-Request"] = "true"
    return headers
