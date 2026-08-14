"""Google Generative AI 和 Google Vertex provider 共享工具。""" 

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from ..types import (
    Context,
    StopReason,
)
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import resolve_json_schema_strict_sampling
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

GoogleApiType = Literal["google-generative-ai", "google-vertex"]

GoogleThinkingLevel = Literal[
    "THINKING_LEVEL_UNSPECIFIED", "MINIMAL", "LOW", "MEDIUM", "HIGH"
]
"""思考级别，对应 Google 的 ThinkingLevel 枚举值。"""


# ---------------------------------------------------------------------------
# 思考部分判断
# ---------------------------------------------------------------------------


def is_thinking_part(part: Any) -> bool:
    """判断流式 Gemini Part 是否应被视为"思考"内容。

    Protocol note (Gemini / Vertex AI thought signatures):
    - ``thought: true`` 是思考内容的明确标记（思考摘要）。
    - ``thoughtSignature`` 是模型内部思考过程的加密表示，用于在多轮交互中保留推理上下文。
    - ``thoughtSignature`` 可以出现在任何 part 类型上（text、functionCall 等）- 这并不表示该 part 本身是思考内容。
    """
    return getattr(part, "thought", False) is True


def retain_thought_signature(existing: str | None, incoming: str | None) -> str | None:
    """在流式传输过程中保留思考签名。

    某些后端仅发送第一个 delta 的 ``thoughtSignature``；后续的 delta 可能省略它。
    此辅助函数保留当前块中最后一个非空签名。

    注意：这不会跨不同响应部分合并或移动签名。它仅防止签名在同一流式块内被 None 覆盖。
    """
    if isinstance(incoming, str) and len(incoming) > 0:
        return incoming
    return existing


# 思考签名必须是 base64（Google API 的 TYPE_BYTES）。
_BASE64_SIGNATURE_PATTERN = re.compile(r"^[A-Za-z0-9+/]+={0,2}$")


def _is_valid_thought_signature(signature: str | None) -> bool:
    if not signature:
        return False
    if len(signature) % 4 != 0:
        return False
    return bool(_BASE64_SIGNATURE_PATTERN.match(signature))


def _resolve_thought_signature(
    is_same_provider_and_model: bool, signature: str | None
) -> str | None:
    """仅保留来自同一 provider/model 且有效的 base64 签名。"""
    if is_same_provider_and_model and _is_valid_thought_signature(signature):
        return signature
    return None


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _get_gemini_major_version(model_id: str) -> int | None:
    """获取 Gemini 主要版本号。"""
    match = re.match(r"^gemini(?:-live)?-(\d+)", model_id.lower())
    if not match:
        return None
    return int(match.group(1))


def requires_tool_call_id(model_id: str) -> bool:
    """检查模型是否需要显式的 tool call ID。"""
    gemini_major_version = _get_gemini_major_version(model_id)
    return (
        model_id.startswith("claude-")
        or model_id.startswith("gpt-oss-")
        or (gemini_major_version is not None and gemini_major_version >= 3)
    )


def _supports_multimodal_function_response(model_id: str) -> bool:
    """检查模型是否支持多模态函数响应。"""
    gemini_major_version = _get_gemini_major_version(model_id)
    if gemini_major_version is not None:
        return gemini_major_version >= 3
    return True


# ---------------------------------------------------------------------------
# 消息转换
# ---------------------------------------------------------------------------


def convert_messages(model: Any, context: Context) -> list[dict[str, Any]]:
    """转换内部消息为 Google Gemini Content[] 格式。"""
    contents: list[dict[str, Any]] = []
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "")

    def _normalize_tool_call_id(id: str, _model: Any = None, _msg: Any = None) -> str:
        if not requires_tool_call_id(model_id):
            return id
        return re.sub(r"[^a-zA-Z0-9_-]", "_", id)[:64]

    transformed_messages = transform_messages(
        context.messages, model, _normalize_tool_call_id
    )

    for msg in transformed_messages:
        if msg.role == "user":
            if isinstance(msg.content, str):
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": sanitize_surrogates(msg.content)}],
                    }
                )
            else:
                parts: list[dict[str, Any]] = []
                for item in msg.content:
                    if item.type == "text":
                        parts.append({"text": sanitize_surrogates(item.text)})
                    else:
                        parts.append(
                            {
                                "inlineData": {
                                    "mimeType": item.mime_type,
                                    "data": item.data,
                                }
                            }
                        )
                if not parts:
                    continue
                contents.append({"role": "user", "parts": parts})
        elif msg.role == "assistant":
            assistant_parts: list[dict[str, Any]] = []
            is_same_provider_and_model = (
                getattr(msg, "provider", None) == getattr(model, "provider", "")
                and getattr(msg, "model", None) == model_id
            )

            for block in msg.content:
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model,
                        getattr(block, "text_signature", None),
                    )
                    # 跳过空文本块 — 除非它们携带 thought signature
                    if (
                        not getattr(block, "text", "")
                        or getattr(block, "text", "").strip() == ""
                    ) and not thought_signature:
                        continue
                    part: dict[str, Any] = {
                        "text": sanitize_surrogates(getattr(block, "text", "") or "")
                    }
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    assistant_parts.append(part)
                elif block_type == "thinking":
                    if is_same_provider_and_model:
                        thought_signature = _resolve_thought_signature(
                            is_same_provider_and_model,
                            getattr(block, "signature", None),
                        )
                        if (
                            not getattr(block, "text", "")
                            or getattr(block, "text", "").strip() == ""
                        ) and not thought_signature:
                            continue
                        part = {
                            "thought": True,
                            "text": sanitize_surrogates(
                                getattr(block, "text", "") or ""
                            ),
                        }
                        if thought_signature:
                            part["thoughtSignature"] = thought_signature
                        assistant_parts.append(part)
                    else:
                        # 跨 provider/model：签名不可用，空块丢弃
                        if (
                            not getattr(block, "text", "")
                            or getattr(block, "text", "").strip() == ""
                        ):
                            continue
                        assistant_parts.append(
                            {
                                "text": sanitize_surrogates(
                                    getattr(block, "text", "") or ""
                                )
                            }
                        )
                elif block_type == "toolCall":
                    thought_signature = _resolve_thought_signature(
                        is_same_provider_and_model,
                        getattr(block, "thought_signature", None),
                    )
                    fc: dict[str, Any] = {
                        "name": getattr(block, "name", ""),
                        "args": (
                            getattr(block, "args", {}) if hasattr(block, "args") else {}
                        ),
                    }
                    if requires_tool_call_id(model_id):
                        fc["id"] = getattr(block, "tool_call_id", "")
                    part = {"functionCall": fc}
                    if thought_signature:
                        part["thoughtSignature"] = thought_signature
                    assistant_parts.append(part)

            if not assistant_parts:
                continue
            contents.append({"role": "model", "parts": assistant_parts})
        elif msg.role == "toolResult":
            # 提取文本和图片内容
            text_content = [c for c in msg.content if c.type == "text"]
            text_result = "\n".join(c.text for c in text_content)
            supports_image = "image" in (
                getattr(model, "input", None)
                or getattr(model, "input_types", None)
                or []
            )
            image_content = (
                [c for c in msg.content if c.type == "image"] if supports_image else []
            )

            has_text = len(text_result) > 0
            has_images = len(image_content) > 0

            model_supports_multimodal = _supports_multimodal_function_response(model_id)

            response_value = (
                sanitize_surrogates(text_result)
                if has_text
                else "(see attached image)"
                if has_images
                else ""
            )

            image_parts: list[dict[str, Any]] = [
                {
                    "inlineData": {
                        "mimeType": img.mime_type,
                        "data": img.data,
                    }
                }
                for img in image_content
            ]

            include_id = requires_tool_call_id(model_id)

            function_response_part: dict[str, Any] = {
                "functionResponse": {
                    "name": getattr(msg, "tool_name", ""),
                    "response": (
                        {"error": response_value}
                        if getattr(msg, "is_error", False)
                        else {"output": response_value}
                    ),
                }
            }
            if has_images and model_supports_multimodal:
                function_response_part["functionResponse"]["parts"] = image_parts
            if include_id:
                function_response_part["functionResponse"]["id"] = getattr(
                    msg, "tool_call_id", ""
                )

            # 合并到上一个 user 消息（如果已有 functionResponse）
            last_content = contents[-1] if contents else None
            if (
                last_content
                and last_content.get("role") == "user"
                and any(
                    p.get("functionResponse") for p in (last_content.get("parts") or [])
                )
            ):
                last_content.setdefault("parts", []).append(function_response_part)
            else:
                contents.append({"role": "user", "parts": [function_response_part]})

            # Gemini < 3：图片需要单独的 user 消息
            if has_images and not model_supports_multimodal:
                contents.append(
                    {
                        "role": "user",
                        "parts": [{"text": "Tool result image:"}, *image_parts],
                    }
                )

    return contents


# ---------------------------------------------------------------------------
# JSON Schema 元声明
# ---------------------------------------------------------------------------

_JSON_SCHEMA_META_DECLARATIONS = frozenset(
    {
        "$schema",
        "$id",
        "$anchor",
        "$dynamicAnchor",
        "$vocabulary",
        "$comment",
        "$defs",
        "definitions",  # pre-draft-2019-09 版本的 $defs
    }
)


def _sanitize_for_open_api(schema: Any) -> Any:
    """从 schema 对象中移除 meta 声明。"""
    if not isinstance(schema, dict):
        return schema
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in _JSON_SCHEMA_META_DECLARATIONS:
            continue
        result[key] = _sanitize_for_open_api(value)
    return result


# ---------------------------------------------------------------------------
# 工具转换
# ---------------------------------------------------------------------------


def convert_tools(
    tools: list[Any],
    use_parameters: bool = False,
) -> list[dict[str, Any]] | None:
    """转换工具为 Google function declarations 格式。

    默认使用 ``parametersJsonSchema``（支持完整 JSON Schema，包括 anyOf、oneOf、const 等）。
    设置 ``use_parameters=True`` 可使用遗留的 ``parameters`` 字段（OpenAPI 3.03 Schema）。
    """
    if not tools:
        return None
    return [
        {
            "functionDeclarations": [
                {
                    "name": tool.name,
                    "description": tool.description,
                    **(
                        {"parameters": _sanitize_for_open_api(tool.parameters)}
                        if use_parameters
                        else {"parametersJsonSchema": tool.parameters}
                    ),
                }
                for tool in tools
            ]
        }
    ]


# ---------------------------------------------------------------------------
# 严格工具采样
# ---------------------------------------------------------------------------


def supports_google_strict_tool_sampling(model_id: str) -> bool:
    """Gemini 3+ 在已验证的工具调用模式下强制执行必需的函数参数。"""
    major_version = _get_gemini_major_version(model_id)
    return major_version is not None and major_version >= 3


# ---------------------------------------------------------------------------
# 工具选择映射
# ---------------------------------------------------------------------------


def map_tool_choice(choice: str) -> str:
    """映射工具选择字符串到 Google FunctionCallingConfigMode。"""
    mapping: dict[str, str] = {
        "auto": "AUTO",
        "none": "NONE",
        "any": "ANY",
    }
    return mapping.get(choice, "AUTO")


def resolve_google_function_calling_mode(
    tools: list[Any],
    tool_choice: str | None,
    supports_strict_mode: bool,
) -> str | None:
    """解析 Google 函数调用模式。"""
    use_strict_mode = any(
        resolve_json_schema_strict_sampling(tool, supports_strict_mode) is True
        for tool in tools
    )
    if tool_choice in ("none", "any"):
        return map_tool_choice(tool_choice)
    if use_strict_mode:
        return "VALIDATED"
    return map_tool_choice(tool_choice) if tool_choice else None


# ---------------------------------------------------------------------------
# 停止原因映射
# ---------------------------------------------------------------------------


FINISH_REASON_MAP: dict[str, StopReason] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
}

FINISH_REASON_ERROR_SET: frozenset[str] = frozenset(
    {
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "SPII",
        "SAFETY",
        "IMAGE_SAFETY",
        "IMAGE_PROHIBITED_CONTENT",
        "IMAGE_RECITATION",
        "IMAGE_OTHER",
        "RECITATION",
        "FINISH_REASON_UNSPECIFIED",
        "OTHER",
        "LANGUAGE",
        "MALFORMED_FUNCTION_CALL",
        "UNEXPECTED_TOOL_CALL",
        "NO_IMAGE",
    }
)


def map_stop_reason(reason: Any) -> StopReason:
    """映射 Google FinishReason 到 StopReason。"""
    reason_str: str = (
        reason if isinstance(reason, str) else getattr(reason, "value", str(reason))
    )
    mapped = FINISH_REASON_MAP.get(reason_str)
    if mapped is not None:
        return mapped
    if reason_str in FINISH_REASON_ERROR_SET:
        return "error"
    raise ValueError(f"Unhandled stop reason: {reason_str}")


def map_stop_reason_string(reason: str) -> StopReason:
    """映射字符串形式的 finish reason 到 StopReason。"""
    if reason == "STOP":
        return "stop"
    if reason == "MAX_TOKENS":
        return "length"
    return "error"


# ---------------------------------------------------------------------------
# 重试
# ---------------------------------------------------------------------------


async def retry_google_request(
    request: Callable[[], Awaitable[Any]],
    options: Any | None = None,
) -> Any:
    """使用共享的 provider 重试策略运行 Google GenAI SDK 请求。

    SDK 的 ApiError 有 ``status`` 属性但没有 ``headers`` 属性，而
    retry_provider_request 只对同时携带两者的错误进行重试。
    因此将错误标准化为 ProviderError 以支持重试。
    """
    return await retry_provider_request(
        lambda: _wrapped_request(request),
        ProviderRetryOptions(
            max_retries=getattr(options, "max_retries", 0) if options else 0,
            max_retry_delay_ms=(
                getattr(options, "max_retry_delay_ms", None) if options else None
            ),
            signal=getattr(options, "signal", None) if options else None,
        ),
    )


async def _wrapped_request(
    request: Callable[[], Awaitable[Any]],
) -> Any:
    """包装请求以标准化错误。"""
    from ..utils.provider_retry import ProviderError

    try:
        return await request()
    except Exception as error:
        status = getattr(error, "status", None)
        if status is not None and isinstance(status, int):
            raise ProviderError(
                message=str(error),
                status=status,
            ) from error
        raise
