"""Provider HTTP 错误标准化（对应 ``utils/error-body.ts``）。

提供 ``normalize_provider_error``、``format_provider_error``、
``truncate_error_text``、``safe_json_stringify``。
"""

from __future__ import annotations

import json

MAX_PROVIDER_ERROR_BODY_CHARS = 4000


class NormalizedProviderError:
    """标准化后的 provider 错误（对应 TS ``NormalizedProviderError``）。"""

    __slots__ = ("status", "body", "message", "message_carries_body")

    def __init__(
        self,
        message: str,
        status: int | None = None,
        body: str | None = None,
        message_carries_body: bool = False,
    ) -> None:
        self.status = status
        self.body = body
        self.message = message
        self.message_carries_body = message_carries_body


def normalize_provider_error(error: object) -> NormalizedProviderError:
    """标准化 provider 错误（对应 TS ``normalizeProviderError``）。

    探测 SDK 错误对象的字段形状（statusCode、status、body、error、$metadata 等）。
    """
    if not isinstance(error, BaseException):
        return NormalizedProviderError(message=safe_json_stringify(error))

    sdk_error = error
    status = _extract_status(sdk_error)
    body = _extract_body(sdk_error)
    message_carries_body = body is None or str(error).find(body) != -1

    return NormalizedProviderError(
        status=status,
        body=body,
        message=str(error),
        message_carries_body=message_carries_body,
    )


def _extract_status(error: BaseException) -> int | None:
    """从 SDK 错误对象中提取 HTTP 状态码。"""
    for attr in ("statusCode", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val

    metadata = getattr(error, "$metadata", None)
    if metadata is not None:
        code = getattr(metadata, "httpStatusCode", None)
        if isinstance(code, int):
            return code

    response = getattr(error, "$response", None)
    if response is not None:
        code = getattr(response, "statusCode", None)
        if isinstance(code, int):
            return code

    return None


def _extract_body(error: BaseException) -> str | None:
    """从 SDK 错误对象中提取响应体。"""
    body_text = _pick_body_text(error)
    if body_text is None:
        return None
    trimmed = body_text.strip()
    if not trimmed:
        return None
    return truncate_error_text(trimmed, MAX_PROVIDER_ERROR_BODY_CHARS)


def _pick_body_text(error: BaseException) -> str | None:
    """选择最佳 body 文本来源。"""
    body = getattr(error, "body", None)
    if isinstance(body, str):
        return body

    err_obj = getattr(error, "error", None)
    if _is_plain_non_empty_object(err_obj):
        return safe_json_stringify(err_obj)

    response = getattr(error, "$response", None)
    if response is not None:
        response_body = getattr(response, "body", None)
        if isinstance(response_body, str):
            return response_body
        if _is_plain_non_empty_object(response_body):
            return safe_json_stringify(response_body)

    return None


def _is_plain_non_empty_object(value: object) -> bool:
    """检查是否为纯非空对象（对应 TS ``isPlainNonEmptyObject``）。"""
    if not isinstance(value, dict):
        return False
    return len(value) > 0


def format_provider_error(
    norm: NormalizedProviderError, prefix: str | None = None
) -> str:
    """格式化 provider 错误显示字符串（对应 TS ``formatProviderError``）。"""
    if norm.message_carries_body or norm.status is None or norm.body is None:
        if prefix is not None and norm.status is not None:
            return f"{prefix} ({norm.status}): {norm.message}"
        return norm.message
    if prefix is not None:
        return f"{prefix} ({norm.status}): {norm.body}"
    return f"{norm.status}: {norm.body}"


def truncate_error_text(text: str, max_chars: int) -> str:
    """截断错误文本（对应 TS ``truncateErrorText``）。"""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def safe_json_stringify(value: object) -> str:
    """安全地 JSON 序列化（对应 TS ``safeJsonStringify``）。"""
    try:
        serialized = json.dumps(value)
        return serialized if serialized != "undefined" else str(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)