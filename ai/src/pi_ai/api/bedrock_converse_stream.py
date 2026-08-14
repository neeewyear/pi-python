"""Bedrock Converse API 流式传输。

使用 ``aioboto3`` 实现异步 Bedrock Converse Stream 调用。
"""

from __future__ import annotations

import base64
import re
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast
from urllib.parse import urlparse

from pydantic import BaseModel

from ..models import calculate_cost
from ..types import (
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    AssistantTextDelta,
    AssistantThinkingDelta,
    AssistantToolCallEnd,
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    CacheRetention,
    Context,
    Cost,
    ImageContent,
    StreamOptions,
    TextContent,
    ThinkingBlock,
    ThinkingLevel,
    ToolCallContent,
    ToolResultMessage,
    Usage,
    UserMessage,
)
from ..utils.diagnostics import append_assistant_message_diagnostic
from ..utils.error_body import normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import provider_headers_to_record
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import resolve_json_schema_strict_sampling
from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
    clamp_reasoning,
)
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

EMPTY_TEXT_PLACEHOLDER = "<empty>"

# 人类可读的 Bedrock SDK 异常名称前缀
BEDROCK_ERROR_PREFIXES: dict[str, str] = {
    "InternalServerException": "Internal server error",
    "ModelStreamErrorException": "Model stream error",
    "ValidationException": "Validation error",
    "ThrottlingException": "Throttling error",
    "ServiceUnavailableException": "Service unavailable",
}

# 数据保留模式文档链接
BEDROCK_DATA_RETENTION_DOCS_URL = (
    "https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html"
)

# 诊断值最大字符数
MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS = 200

# 保留的请求头（大小写不敏感比较）
RESERVED_HEADER_EXACT = frozenset({"authorization", "host"})

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

BedrockThinkingDisplay = Literal["summarized", "omitted"]


class BedrockOptions(StreamOptions):
    """Bedrock Converse API 流式选项。"""

    tool_choice: Literal["auto", "none", "any"] | None = None
    reasoning: ThinkingLevel | None = None
    thinking_budgets: Any | None = None
    interleaved_thinking: bool | None = None
    thinking_display: BedrockThinkingDisplay | None = None
    signal: Any | None = None
    on_payload: (
        Callable[[dict[str, Any], Any], Awaitable[dict[str, Any] | None]] | None
    ) = None
    on_response: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None
    fetch: Any = None
    env: Any = None
    cache_retention: Any = None
    session_id: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    sampling_params: dict[str, Any] | None = None
    headers: dict[str, str | None] | None = None
    api_key: str | None = None
    # Bedrock 特有字段
    region: str | None = None
    profile: str | None = None
    access_key_id: str | None = None
    secret_access_key: str | None = None
    session_token: str | None = None
    role_arn: str | None = None
    external_id: str | None = None
    request_metadata: dict[str, str] | None = None
    bearer_token: str | None = None


# 内部块类型（含流式暂存字段）
class _Block(BaseModel):
    """内部内容块，用于跟踪流式处理中的 index 和 partial_json。"""

    type: str = "text"
    index: int | None = None
    partial_json: str | None = None
    tool_call_id: str = ""
    name: str = ""
    args: dict[str, object] = {}


# ---------------------------------------------------------------------------
# 辅助函数 —— 错误处理
# ---------------------------------------------------------------------------


def _format_bedrock_error(error: object) -> str:
    """格式化 Bedrock 错误。"""
    norm = normalize_provider_error(error)
    # 当 SDK 未将 body 并入消息时，优先使用原始 HTTP body
    core: str
    if (
        not norm.message_carries_body
        and norm.status is not None
        and norm.body is not None
    ):
        core = f"{norm.status}: {norm.body}"
    else:
        core = norm.message

    data_retention_hint = (
        f" See {BEDROCK_DATA_RETENTION_DOCS_URL} for supported data retention modes."
        if re.search(r"data retention mode", core, re.IGNORECASE)
        else ""
    )

    # 检查是否为 Bedrock 服务异常
    error_name = getattr(error, "__class__", None)
    error_cls_name = error_name.__name__ if error_name else ""
    if not error_cls_name:
        error_cls_name = getattr(error, "name", "") or type(error).__name__
    prefix = BEDROCK_ERROR_PREFIXES.get(error_cls_name, error_cls_name)
    if (
        isinstance(error, BaseException)
        and error_cls_name
        and error_cls_name in BEDROCK_ERROR_PREFIXES
    ):
        return f"{prefix}: {core}{data_retention_hint}"
    return f"{core}{data_retention_hint}"


def _normalize_diagnostic_value(value: object) -> str | None:
    """归一化诊断值。"""
    if not isinstance(value, str):
        return None
    trimmed = value.strip()
    if not trimmed or len(trimmed) > MAX_BEDROCK_DIAGNOSTIC_VALUE_CHARS:
        return None
    return trimmed


def _extract_bedrock_error_code(error: object) -> str | None:
    """提取 Bedrock 错误码。"""
    if not isinstance(error, BaseException):
        return None
    error_name = type(error).__name__
    if not error_name.endswith("Exception"):
        return None
    return _normalize_diagnostic_value(error_name)


def _append_bedrock_failure_diagnostic(
    output: AssistantMessage,
    error: object,
    fallback_request_id: str | None,
) -> None:
    """追加 Bedrock 失败诊断。"""
    metadata = getattr(error, "$metadata", None) or getattr(
        error, "response_metadata", None
    )
    details: dict[str, object] = {}

    if metadata is not None:
        http_status = getattr(metadata, "http_status_code", None) or getattr(
            metadata, "HTTPStatusCode", None
        )
        if isinstance(http_status, int):
            details["status"] = http_status
        request_id = getattr(metadata, "request_id", None) or getattr(
            metadata, "RequestId", None
        )
        req_id = _normalize_diagnostic_value(request_id) or fallback_request_id
        if req_id is not None:
            details["request_id"] = req_id

    error_code = _extract_bedrock_error_code(error)
    if error_code is not None:
        details["error_code"] = error_code

    if not details:
        return

    append_assistant_message_diagnostic(
        cast("dict[str, object]", output),
        type(
            "AssistantMessageDiagnostic",
            (),
            {
                "type": "bedrock_response_failure",
                "timestamp": time.time(),
                "details": details,
                "error": None,
            },
        )(),
    )


# ---------------------------------------------------------------------------
# 辅助函数 —— 请求头
# ---------------------------------------------------------------------------


def _is_reserved_header(key: str) -> bool:
    """检查是否为保留请求头。"""
    lower = key.lower()
    if lower.startswith("x-amz-"):
        return True
    return lower in RESERVED_HEADER_EXACT


def _add_custom_headers(client: Any, headers: dict[str, str]) -> None:
    """通过 boto3 事件系统添加自定义请求头。"""

    def _before_send(request: Any, **kwargs: Any) -> None:
        for key, value in headers.items():
            if not _is_reserved_header(key):
                request.headers[key] = value

    client.meta.events.register(
        "before-send.bedrock-runtime.ConverseStream",
        _before_send,
    )


# ---------------------------------------------------------------------------
# 辅助函数 —— 流式处理
# ---------------------------------------------------------------------------


def _handle_content_block_start(
    event: dict[str, Any],
    blocks: list[Any],
    output: AssistantMessage,
    event_stream: AssistantMessageEventStream,
) -> None:
    """处理 content_block_start 事件。"""
    index = event.get("contentBlockIndex", 0)
    start = event.get("start", {})

    if "toolUse" in start:
        tool_use = start["toolUse"]
        block = _Block(
            type="toolCall",
            tool_call_id=tool_use.get("toolUseId", ""),
            name=tool_use.get("name", ""),
            args={},
            partial_json="",
            index=index,
        )
        output.content.append(
            ToolCallContent(
                tool_call_id=block.tool_call_id,
                name=block.name,
                args={},
            )
        )
        content_index = len(output.content) - 1
        event_stream.push(
            AssistantToolCallStart(
                tool_call_id=block.tool_call_id,
                name=block.name,
            )
        )
        # Store block tracking info
        _set_block_tracking(output, content_index, index, block)


def _handle_content_block_delta(
    event: dict[str, Any],
    blocks: list[Any],
    output: AssistantMessage,
    event_stream: AssistantMessageEventStream,
) -> None:
    """处理 content_block_delta 事件。"""
    content_block_index = event.get("contentBlockIndex", 0)
    delta = event.get("delta", {})

    # Find existing block by index
    tracking = _get_block_tracking(output, content_block_index)

    if "text" in delta:
        text_delta = delta["text"]
        if tracking is None:
            # No text block exists yet, create one
            text_block = TextContent(text="")
            output.content.append(text_block)
            content_index = len(output.content) - 1
            _set_block_tracking(output, content_index, content_block_index, None)
            tracking = _get_block_tracking(output, content_block_index)
            event_stream.push(AssistantMessageSnapshot(message=output))

        if tracking is not None:
            text_block = cast(TextContent, tracking["block"])
            text_block.text += text_delta
            event_stream.push(AssistantTextDelta(delta=text_delta))

    elif "toolUse" in delta and tracking is not None:
        tool_use_delta = delta["toolUse"]
        input_delta = tool_use_delta.get("input", "")
        block = tracking["block"]
        if hasattr(block, "partial_json"):
            block.partial_json = (block.partial_json or "") + input_delta
            block.args = parse_streaming_json(block.partial_json)
            # Update the ToolCallContent in output
            tc = cast(ToolCallContent, output.content[tracking["content_index"]])
            object.__setattr__(tc, "args", block.args)
            event_stream.push(
                AssistantToolCallUpdate(
                    tool_call_id=tc.tool_call_id,
                    args=tc.args,
                )
            )

    elif "reasoningContent" in delta:
        reasoning = delta["reasoningContent"]
        if tracking is None:
            # No thinking block exists yet, create one
            think_block = ThinkingBlock(text="", signature="")
            output.thinking = (output.thinking or []) + [think_block]
            content_index = len(output.thinking) - 1
            _set_thinking_tracking(
                output, content_index, content_block_index, think_block
            )
            tracking = _get_thinking_tracking(output, content_block_index)
            event_stream.push(AssistantMessageSnapshot(message=output))

        if tracking is not None:
            think_block = cast(ThinkingBlock, tracking["block"])
            if "text" in reasoning:
                think_block.text += reasoning["text"]
                event_stream.push(AssistantThinkingDelta(delta=reasoning["text"]))
            if "signature" in reasoning:
                think_block.signature = (think_block.signature or "") + reasoning[
                    "signature"
                ]


def _handle_content_block_stop(
    event: dict[str, Any],
    blocks: list[Any],
    output: AssistantMessage,
    event_stream: AssistantMessageEventStream,
) -> None:
    """处理 content_block_stop 事件。"""
    content_block_index = event.get("contentBlockIndex", 0)

    # Check text/toolCall blocks
    tracking = _pop_block_tracking(output, content_block_index)
    if tracking is not None:
        block = tracking["block"]
        content_index = tracking["content_index"]
        if block.type == "text":
            event_stream.push(AssistantMessageSnapshot(message=output))
        elif block.type == "toolCall":
            if hasattr(block, "partial_json") and block.partial_json:
                block.args = parse_streaming_json(block.partial_json)
                # Update the ToolCallContent in output
                tc = cast(ToolCallContent, output.content[content_index])
                object.__setattr__(tc, "args", block.args)
            event_stream.push(
                AssistantToolCallEnd(
                    tool_call_id=block.tool_call_id,
                    content=[
                        ToolCallContent(
                            type="toolCall",
                            tool_call_id=block.tool_call_id,
                            name=block.name,
                            args=block.args,
                        )
                    ],
                )
            )

    # Check thinking blocks
    thinking_tracking = _pop_thinking_tracking(output, content_block_index)
    if thinking_tracking is not None:
        event_stream.push(AssistantMessageSnapshot(message=output))


def _handle_metadata(
    event: dict[str, Any],
    model: Any,
    output: AssistantMessage,
) -> None:
    """处理 metadata 事件。"""
    usage = event.get("usage", {})
    if usage:
        output_usage = output.usage
        assert output_usage is not None
        output_usage.input = usage.get("inputTokens", 0) or 0
        output_usage.output = usage.get("outputTokens", 0) or 0
        output_usage.cache_read = usage.get("cacheReadInputTokens", 0) or 0
        output_usage.cache_write = usage.get("cacheWriteInputTokens", 0) or 0
        output_usage.total_tokens = (
            usage.get("totalTokens", 0) or output_usage.input + output_usage.output
        )
        calculate_cost(model, output_usage)


# Block tracking helpers (using object.__setattr__ on AssistantMessage to store private state)
_BLOCK_TRACKING_ATTR = "_bedrock_block_tracking"
_THINKING_TRACKING_ATTR = "_bedrock_thinking_tracking"


def _get_tracking_dict(
    output: AssistantMessage, attr: str
) -> dict[int, dict[str, Any]] | None:
    """获取跟踪字典。"""
    val = getattr(output, attr, None)
    if val is None:
        return None
    return cast("dict[int, dict[str, Any]]", val)


def _get_block_tracking(output: AssistantMessage, index: int) -> dict[str, Any] | None:
    """获取内容块跟踪信息。"""
    tracking = _get_tracking_dict(output, _BLOCK_TRACKING_ATTR)
    if tracking is None:
        return None
    return tracking.get(index)


def _set_block_tracking(
    output: AssistantMessage, content_index: int, block_index: int, block: Any
) -> None:
    """设置内容块跟踪信息。"""
    tracking = _get_tracking_dict(output, _BLOCK_TRACKING_ATTR)
    if tracking is None:
        tracking = {}
        object.__setattr__(output, _BLOCK_TRACKING_ATTR, tracking)
    tracking[block_index] = {
        "content_index": content_index,
        "block": block,
        "type": block.type if block is not None else "text",
    }


def _pop_block_tracking(output: AssistantMessage, index: int) -> dict[str, Any] | None:
    """弹出并返回内容块跟踪信息。"""
    tracking = _get_tracking_dict(output, _BLOCK_TRACKING_ATTR)
    if tracking is None:
        return None
    return tracking.pop(index, None)


def _get_thinking_tracking(
    output: AssistantMessage, index: int
) -> dict[str, Any] | None:
    """获取思考块跟踪信息。"""
    tracking = _get_tracking_dict(output, _THINKING_TRACKING_ATTR)
    if tracking is None:
        return None
    return tracking.get(index)


def _set_thinking_tracking(
    output: AssistantMessage, content_index: int, block_index: int, block: Any
) -> None:
    """设置思考块跟踪信息。"""
    tracking = _get_tracking_dict(output, _THINKING_TRACKING_ATTR)
    if tracking is None:
        tracking = {}
        object.__setattr__(output, _THINKING_TRACKING_ATTR, tracking)
    tracking[block_index] = {
        "content_index": content_index,
        "block": block,
        "type": "thinking",
    }


def _pop_thinking_tracking(
    output: AssistantMessage, index: int
) -> dict[str, Any] | None:
    """弹出并返回思考块跟踪信息。"""
    tracking = _get_tracking_dict(output, _THINKING_TRACKING_ATTR)
    if tracking is None:
        return None
    return tracking.pop(index, None)


# ---------------------------------------------------------------------------
# 辅助函数 —— 模型检测
# ---------------------------------------------------------------------------


def _get_model_match_candidates(
    model_id: str, model_name: str | None = None
) -> list[str]:
    """获取模型匹配候选列表。"""
    values = [model_id]
    if model_name:
        values.append(model_name)
    result: list[str] = []
    for value in values:
        lower = value.lower()
        result.append(lower)
        result.append(re.sub(r"[\s_.:]+", "-", lower))
    return result


def _supports_adaptive_thinking(model_id: str, model_name: str | None = None) -> bool:
    """检查是否支持自适应思考。"""  
    candidates = _get_model_match_candidates(model_id, model_name)
    return any(
        "opus-4-6" in c
        or "opus-4-7" in c
        or "opus-4-8" in c
        or "opus-5" in c
        or "sonnet-4-6" in c
        or "sonnet-5" in c
        or "fable-5" in c
        for c in candidates
    )


def _supports_native_xhigh_effort(model: Any) -> bool:
    """检查是否支持原生 xhigh effort。"""
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
    model_name = getattr(model, "name", None)
    candidates = _get_model_match_candidates(model_id, model_name)
    return any(
        "opus-4-7" in c
        or "opus-4-8" in c
        or "opus-5" in c
        or "sonnet-5" in c
        or "fable-5" in c
        for c in candidates
    )


def _map_thinking_level_to_effort(
    model: Any,
    level: str | None,
) -> str:
    """映射思考级别到 effort。"""
    if level == "xhigh" and _supports_native_xhigh_effort(model):
        return "xhigh"

    thinking_level_map = getattr(model, "thinking_level_map", None) or {}
    mapped = thinking_level_map.get(level)
    if isinstance(mapped, str) and mapped in ("low", "medium", "high", "xhigh", "max"):
        return mapped

    level_map: dict[str, str] = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    return level_map.get(level, "high")  # type: ignore[arg-type]


def _is_anthropic_claude_model(model: Any) -> bool:
    """检查是否为 Anthropic Claude 模型。"""
    model_id = (getattr(model, "model_id", "") or getattr(model, "id", "")).lower()
    model_name = (getattr(model, "name", "") or "").lower()
    return (
        "anthropic.claude" in model_id
        or "anthropic/claude" in model_id
        or "anthropic.claude" in model_name
        or "anthropic/claude" in model_name
        or "claude" in model_name
    )


def _supports_prompt_caching(model: Any, env: Any = None) -> bool:
    """检查是否支持提示缓存。"""
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
    model_name = getattr(model, "name", None)
    candidates = _get_model_match_candidates(model_id, model_name)

    has_claude_ref = any("claude" in c for c in candidates)
    if not has_claude_ref:
        if get_provider_env_value("AWS_BEDROCK_FORCE_CACHE", env) == "1":
            return True
        return False

    # Claude 5 models
    if any("fable-5" in c or "opus-5" in c or "sonnet-5" in c for c in candidates):
        return True
    # Claude 4.x models
    if any("-4-" in c for c in candidates):
        return True
    # Claude 3.7 Sonnet
    if any("claude-3-7-sonnet" in c for c in candidates):
        return True
    # Claude 3.5 Haiku
    if any("claude-3-5-haiku" in c for c in candidates):
        return True
    return False


def _supports_thinking_signature(model: Any) -> bool:
    """检查是否支持思考签名。"""
    return _is_anthropic_claude_model(model)


# ---------------------------------------------------------------------------
# 辅助函数 —— 缓存
# ---------------------------------------------------------------------------


def _resolve_cache_retention(
    cache_retention: CacheRetention | None = None,
    env: Any = None,
) -> CacheRetention:
    """解析缓存保留策略。"""
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


# ---------------------------------------------------------------------------
# 辅助函数 —— 系统提示词
# ---------------------------------------------------------------------------


def _build_system_prompt(
    system_prompt: str | None,
    model: Any,
    cache_retention: CacheRetention,
    env: Any = None,
) -> list[dict[str, Any]] | None:
    """构建系统提示词。"""
    if not system_prompt:
        return None

    blocks: list[dict[str, Any]] = [{"text": sanitize_surrogates(system_prompt)}]

    # 为支持的 Claude 模型添加缓存点
    if cache_retention != "none" and _supports_prompt_caching(model, env):
        cache_point: dict[str, Any] = {"type": "default"}
        if cache_retention == "long":
            cache_point["ttl"] = 3600  # ONE_HOUR in seconds
        blocks.append({"cachePoint": cache_point})

    return blocks


# ---------------------------------------------------------------------------
# 辅助函数 —— 消息转换
# ---------------------------------------------------------------------------


def _normalize_tool_call_id(id: str, _model: Any = None, _msg: Any = None) -> str:
    """归一化 tool call ID。"""
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", id)
    return sanitized[:64]


def _create_non_blank_text_block(text: str) -> dict[str, Any] | None:
    """创建非空文本块。"""
    sanitized = sanitize_surrogates(text)
    if not sanitized.strip():
        return None
    return {"text": sanitized}


def _create_required_text_block(text: str) -> dict[str, Any]:
    """创建必要文本块。"""
    block = _create_non_blank_text_block(text)
    if block is not None:
        return block
    return {"text": EMPTY_TEXT_PLACEHOLDER}


def _create_image_block(mime_type: str, data: str) -> dict[str, Any]:
    """创建图片块。"""
    format_map: dict[str, str] = {
        "image/jpeg": "jpeg",
        "image/jpg": "jpeg",
        "image/png": "png",
        "image/gif": "gif",
        "image/webp": "webp",
    }
    fmt = format_map.get(mime_type)
    if fmt is None:
        raise ValueError(f"Unknown image type: {mime_type}")

    bytes_data = base64.b64decode(data)
    return {
        "image": {
            "format": fmt,
            "source": {"bytes": bytes_data},
        }
    }


def _convert_tool_result_content(
    content: list[TextContent | ImageContent],
) -> list[dict[str, Any]]:
    """转换工具结果内容。"""
    result: list[dict[str, Any]] = []
    for block in content:
        if block.type == "image":
            image_block = _create_image_block(block.mime_type, block.data)
            result.append(image_block)
        else:
            text_block = _create_non_blank_text_block(block.text)
            if text_block is not None:
                result.append(text_block)
    if not result:
        result.append({"text": EMPTY_TEXT_PLACEHOLDER})
    return result


def _convert_messages(
    context: Context,
    model: Any,
    cache_retention: CacheRetention,
    env: Any = None,
) -> list[dict[str, Any]]:
    """转换消息为 Bedrock 格式。"""
    result: list[dict[str, Any]] = []
    transformed_messages = transform_messages(
        context.messages, model, _normalize_tool_call_id
    )

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        if isinstance(msg, UserMessage):
            content: list[dict[str, Any]] = []
            msg_content = msg.content
            if isinstance(msg_content, str):
                content.append(_create_required_text_block(msg_content))
            else:
                for block in msg_content:
                    if block.type == "text":
                        text_block = _create_non_blank_text_block(block.text)
                        if text_block is not None:
                            content.append(text_block)
                    elif block.type == "image":
                        image_block = _create_image_block(block.mime_type, block.data)
                        content.append(image_block)
                if not content:
                    content.append({"text": EMPTY_TEXT_PLACEHOLDER})
            result.append({"role": "user", "content": content})

        elif isinstance(msg, AssistantMessage):
            # 跳过空内容（如中止请求）
            if not msg.content:
                i += 1
                continue

            content_blocks: list[dict[str, Any]] = []
            for cb in msg.content:
                block_type = getattr(cb, "type", None)
                if block_type == "text":
                    text_block = _create_non_blank_text_block(
                        cast(TextContent, cb).text
                    )
                    if text_block is not None:
                        content_blocks.append(text_block)
                elif block_type == "toolCall":
                    tc = cast(ToolCallContent, cb)
                    content_blocks.append(
                        {
                            "toolUse": {
                                "toolUseId": tc.tool_call_id,
                                "name": tc.name,
                                "input": tc.args,
                            },
                        }
                    )
                elif block_type == "thinking":
                    think_block = cast(ThinkingBlock, cb)
                    thinking = sanitize_surrogates(think_block.text)
                    if not thinking.strip():
                        continue
                    if _supports_thinking_signature(model):
                        if (
                            not think_block.signature
                            or not think_block.signature.strip()
                        ):
                            # 缺少签名时回退到纯文本
                            content_blocks.append({"text": thinking})
                        else:
                            content_blocks.append(
                                {
                                    "reasoningContent": {
                                        "reasoningText": {
                                            "text": thinking,
                                            "signature": think_block.signature,
                                        },
                                    },
                                }
                            )
                    else:
                        content_blocks.append(
                            {
                                "reasoningContent": {
                                    "reasoningText": {"text": thinking},
                                },
                            }
                        )

            if not content_blocks:
                i += 1
                continue
            result.append({"role": "assistant", "content": content_blocks})

        elif isinstance(msg, ToolResultMessage):
            # 收集所有连续的 toolResult 消息
            tool_results: list[dict[str, Any]] = []

            # 添加当前工具结果
            tool_results.append(
                {
                    "toolResult": {
                        "toolUseId": msg.tool_call_id,
                        "content": _convert_tool_result_content(msg.content),
                        "status": "error" if msg.is_error else "success",
                    },
                }
            )

            # 向前查找连续的工具结果消息
            j = i + 1
            while j < len(transformed_messages) and isinstance(
                transformed_messages[j], ToolResultMessage
            ):
                next_msg = cast(ToolResultMessage, transformed_messages[j])
                tool_results.append(
                    {
                        "toolResult": {
                            "toolUseId": next_msg.tool_call_id,
                            "content": _convert_tool_result_content(next_msg.content),
                            "status": "error" if next_msg.is_error else "success",
                        },
                    }
                )
                j += 1

            # 跳过已处理的消息
            i = j - 1

            result.append({"role": "user", "content": tool_results})

        i += 1

    # 为支持的 Claude 模型添加缓存点
    if cache_retention != "none" and _supports_prompt_caching(model, env) and result:
        last_message = result[-1]
        if last_message.get("role") == "user" and last_message.get("content"):
            cache_point: dict[str, Any] = {"type": "default"}
            if cache_retention == "long":
                cache_point["ttl"] = 3600
            last_message["content"].append({"cachePoint": cache_point})

    return result


# ---------------------------------------------------------------------------
# 辅助函数 —— 工具配置
# ---------------------------------------------------------------------------


def _convert_tool_config(
    tools: list[Any] | None,
    tool_choice: str | None,
    supports_strict_mode: bool,
) -> dict[str, Any] | None:
    """转换工具配置。"""    
    if not tools:
        return None
    if tool_choice == "none":
        return None

    bedrock_tools: list[dict[str, Any]] = []
    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        params = tool.parameters if isinstance(tool.parameters, dict) else {}
        tool_spec: dict[str, Any] = {
            "name": tool.name,
            "description": tool.description,
            "inputSchema": {"json": params},
        }
        if strict is True:
            tool_spec["strict"] = True
        bedrock_tools.append({"toolSpec": tool_spec})

    bedrock_tool_choice: dict[str, Any] | None = None
    if tool_choice == "auto":
        bedrock_tool_choice = {"auto": {}}
    elif tool_choice == "any":
        bedrock_tool_choice = {"any": {}}

    if bedrock_tool_choice is None:
        return None

    return {"tools": bedrock_tools, "toolChoice": bedrock_tool_choice}


# ---------------------------------------------------------------------------
# 辅助函数 —— Stop Reason 映射
# ---------------------------------------------------------------------------


def _map_stop_reason(reason: str | None) -> dict[str, Any]:
    """映射 stop reason。"""
    mapping: dict[str, dict[str, Any]] = {
        "end_turn": {"stop_reason": "stop"},
        "stop_sequence": {"stop_reason": "stop"},
        "max_tokens": {"stop_reason": "length"},
        "model_context_window_exceeded": {"stop_reason": "length"},
        "tool_use": {"stop_reason": "tool_use"},
    }
    if reason is not None and reason in mapping:
        return mapping[reason]
    if reason:
        return {
            "stop_reason": "error",
            "error_message": f"Provider stopped with: {reason}",
        }
    return {"stop_reason": "error"}


# ---------------------------------------------------------------------------
# 辅助函数 —— 区域/认证
# ---------------------------------------------------------------------------


def _get_configured_bedrock_region(options: BedrockOptions) -> str | None:
    """获取配置的 Bedrock 区域。"""
    return (
        options.region
        or get_provider_env_value("AWS_REGION", options.env)
        or get_provider_env_value("AWS_DEFAULT_REGION", options.env)
        or None
    )


def _get_configured_bedrock_credentials(env: Any = None) -> dict[str, str] | None:
    """获取配置的 Bedrock 凭证。"""
    access_key_id = get_provider_env_value("AWS_ACCESS_KEY_ID", env)
    secret_access_key = get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
    if not access_key_id or not secret_access_key:
        return None
    result: dict[str, str] = {
        "aws_access_key_id": access_key_id,
        "aws_secret_access_key": secret_access_key,
    }
    session_token = get_provider_env_value("AWS_SESSION_TOKEN", env)
    if session_token:
        result["aws_session_token"] = session_token
    return result


def _get_standard_bedrock_endpoint_region(base_url: str | None) -> str | None:
    """从标准 Bedrock 端点 URL 提取区域。"""
    if not base_url:
        return None
    try:
        parsed = urlparse(base_url)
        hostname = parsed.hostname or ""
        match = re.match(
            r"^bedrock-runtime(?:-fips)?\.([a-z0-9-]+)\.amazonaws\.com(?:\.cn)?$",
            hostname.lower(),
        )
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def _should_use_explicit_bedrock_endpoint(
    base_url: str,
    configured_region: str | None,
    has_ambient_configured_profile: bool,
) -> bool:
    """判断是否应使用显式端点。"""
    endpoint_region = _get_standard_bedrock_endpoint_region(base_url)
    if not endpoint_region:
        return True
    return not configured_region and not has_ambient_configured_profile


def _is_gov_cloud_bedrock_target(model: Any, options: BedrockOptions) -> bool:
    """检查是否为 GovCloud 目标。"""
    region = _get_configured_bedrock_region(options)
    if region and region.lower().startswith("us-gov-"):
        return True
    model_id = (getattr(model, "model_id", "") or getattr(model, "id", "")).lower()
    return model_id.startswith("us-gov.") or model_id.startswith("arn:aws-us-gov:")


# ---------------------------------------------------------------------------
# 辅助函数 —— 额外模型请求字段
# ---------------------------------------------------------------------------


def _build_additional_model_request_fields(
    model: Any,
    options: BedrockOptions,
) -> dict[str, Any] | None:
    """构建额外模型请求字段。"""
    model_reasoning = getattr(model, "reasoning", None)
    if not options.reasoning or not model_reasoning:
        return None

    if _is_anthropic_claude_model(model):
        # GovCloud 暂不支持 thinking.display 字段
        display: str | None = (
            None
            if _is_gov_cloud_bedrock_target(model, options)
            else (options.thinking_display or "summarized")
        )

        model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
        model_name = getattr(model, "name", None)

        if _supports_adaptive_thinking(model_id, model_name):
            result: dict[str, Any] = {
                "thinking": {
                    "type": "adaptive",
                    **({} if display is None else {"display": display}),
                },
                "output_config": {
                    "effort": _map_thinking_level_to_effort(model, options.reasoning),
                },
            }
        else:
            default_budgets: dict[str, int] = {
                "minimal": 1024,
                "low": 2048,
                "medium": 8192,
                "high": 16384,
                "xhigh": 16384,
                "max": 16384,
            }

            level = (
                options.reasoning
                if options.reasoning not in ("xhigh", "max")
                else "high"
            )
            thinking_budgets = getattr(options, "thinking_budgets", None) or {}
            budget = thinking_budgets.get(
                level, default_budgets.get(options.reasoning, 8192)
            )

            result = {
                "thinking": {
                    "type": "enabled",
                    "budget_tokens": budget,
                    **({} if display is None else {"display": display}),
                },
            }

        if not _supports_adaptive_thinking(model_id, model_name):
            interleaved = (
                options.interleaved_thinking
                if options.interleaved_thinking is not None
                else True
            )
            if interleaved:
                result["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]

        return result

    return None


# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------


async def _create_client(
    session: Any,
    model: Any,
    options: BedrockOptions,
) -> Any:
    """创建 Bedrock Runtime 客户端。"""
    from botocore.config import (  # type: ignore[import-not-found]
        Config as BotoCoreConfig,
    )

    config_kwargs: dict[str, Any] = {}

    # 区域解析
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
    options_profile = options.profile or (
        getattr(options.env, "AWS_PROFILE", None)
        if options.env and isinstance(options.env, dict)
        else None
    )
    has_ambient_profile = bool(get_provider_env_value("AWS_PROFILE"))
    configured_region = _get_configured_bedrock_region(options)
    endpoint_region = _get_standard_bedrock_endpoint_region(
        getattr(model, "base_url", None) or getattr(model, "baseUrl", None)
    )
    use_explicit_endpoint = _should_use_explicit_bedrock_endpoint(
        getattr(model, "base_url", None) or getattr(model, "baseUrl", None) or "",
        configured_region,
        has_ambient_profile,
    )

    # ARN 区域提取
    arn_match = re.match(r"^arn:aws(?:-[a-z0-9-]+)?:bedrock:([a-z0-9-]+):", model_id)
    if arn_match:
        config_kwargs["region_name"] = arn_match.group(1)
    elif configured_region:
        config_kwargs["region_name"] = configured_region
    elif endpoint_region and use_explicit_endpoint:
        config_kwargs["region_name"] = endpoint_region
    elif not has_ambient_profile:
        config_kwargs["region_name"] = "us-east-1"

    # 端点
    base_url = getattr(model, "base_url", None) or getattr(model, "baseUrl", None)
    if use_explicit_endpoint and base_url:
        config_kwargs["endpoint_url"] = base_url

    # 凭证
    skip_auth = get_provider_env_value("AWS_BEDROCK_SKIP_AUTH", options.env) == "1"
    bearer_token = (
        options.bearer_token
        or options.api_key
        or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", options.env)
        or None
    )
    use_bearer_token = bearer_token is not None and not skip_auth

    if skip_auth:
        config_kwargs["aws_access_key_id"] = "dummy-access-key"
        config_kwargs["aws_secret_access_key"] = "dummy-secret-key"

    credentials = _get_configured_bedrock_credentials(options.env)
    if not skip_auth and credentials and not options_profile:
        config_kwargs.update(credentials)

    # 代理
    target_url = (
        getattr(model, "base_url", None) or getattr(model, "baseUrl", None) or ""
    )
    proxy_url = None
    if target_url:
        from ..utils.node_http_proxy import resolve_http_proxy_url_for_target

        proxy_url = resolve_http_proxy_url_for_target(target_url, options.env)

    boto_config = BotoCoreConfig()
    if proxy_url:
        boto_config = BotoCoreConfig(proxies={"https": proxy_url})
    elif get_provider_env_value("AWS_BEDROCK_FORCE_HTTP1", options.env) == "1":
        # 强制 HTTP/1.1
        boto_config = BotoCoreConfig(
            max_pool_connections=10,
        )

    if boto_config is not None:
        config_kwargs["config"] = boto_config

    # 创建 session 和 client
    client = await session.client("bedrock-runtime", **config_kwargs)

    # 自定义请求头
    custom_headers = provider_headers_to_record(options.headers)
    if custom_headers:
        _add_custom_headers(client, custom_headers)

    # Bearer token 认证
    if use_bearer_token and bearer_token is not None:
        _add_bearer_token_auth(client, bearer_token)

    return client


def _add_bearer_token_auth(client: Any, token: str) -> None:
    """通过事件系统添加 Bearer token 认证。"""

    def _add_auth_header(request: Any, **kwargs: Any) -> None:
        request.headers["Authorization"] = f"Bearer {token}"

    client.meta.events.register(
        "before-send.bedrock-runtime.ConverseStream", _add_auth_header
    )


# ---------------------------------------------------------------------------
# build_params
# ---------------------------------------------------------------------------


def _build_params(
    model: Any,
    context: Context,
    options: BedrockOptions,
    cache_retention: CacheRetention,
) -> dict[str, Any]:
    """构建请求参数。"""
    # 消息转换
    messages = _convert_messages(context, model, cache_retention, options.env)

    # 系统提示词
    system = _build_system_prompt(
        context.system_prompt, model, cache_retention, options.env
    )

    # 推理配置
    inference_max_tokens = options.max_tokens or (
        (getattr(model, "max_tokens", None) or getattr(model, "maxTokens", None))
        if _is_anthropic_claude_model(model)
        else None
    )
    inference_config: dict[str, Any] = {}
    if inference_max_tokens is not None:
        inference_config["maxTokens"] = inference_max_tokens
    if options.temperature is not None:
        inference_config["temperature"] = options.temperature

    # 工具配置
    supports_strict = getattr(
        getattr(model, "compat", None) or {}, "supportsStrictMode", False
    )
    tool_config = _convert_tool_config(
        context.tools,
        options.tool_choice,
        supports_strict,
    )

    # 额外模型请求字段（thinking）
    additional_fields = _build_additional_model_request_fields(model, options)

    # 构建参数
    params: dict[str, Any] = {
        "modelId": getattr(model, "model_id", "") or getattr(model, "id", ""),
        "messages": messages,
    }
    if system is not None:
        params["system"] = system
    if inference_config:
        params["inferenceConfig"] = inference_config
    if tool_config is not None:
        params["toolConfig"] = tool_config
    if additional_fields is not None:
        params["additionalModelRequestFields"] = additional_fields
    if options.request_metadata is not None:
        params["requestMetadata"] = options.request_metadata

    return params


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: BedrockOptions | None = None,
) -> AssistantMessageEventStream:
    """Bedrock Converse API 流式生成函数。"""
    event_stream = AssistantMessageEventStream()
    opts = options or BedrockOptions()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=getattr(model, "api", ""),
            provider=getattr(model, "provider", ""),
            model=getattr(model, "model_id", "") or getattr(model, "id", ""),
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
                cost=Cost(
                    input=0.0, output=0.0, cache_read=0.0, cache_write=0.0, total=0.0
                ),
            ),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )

        response_request_id: str | None = None

        try:
            # 创建 aioboto3 session
            import aioboto3  # type: ignore[import-not-found]

            session = aioboto3.Session()

            client = await _create_client(session, model, opts)

            # 缓存保留策略
            cache_retention = _resolve_cache_retention(opts.cache_retention, opts.env)

            # 构建请求参数
            params = _build_params(model, context, opts, cache_retention)

            # onPayload 回调
            if opts.on_payload:
                next_params = await opts.on_payload(dict(params), model)
                if next_params is not None:
                    params = next_params

            # 发起流式请求
            response = await client.converse_stream(**params)

            # 获取响应元数据
            response_metadata = response.get("ResponseMetadata", {})
            req_id = response_metadata.get("RequestId")
            if isinstance(req_id, str):
                response_request_id = _normalize_diagnostic_value(req_id)

            # onResponse 回调
            if opts.on_response:
                http_status = response_metadata.get("HTTPStatusCode", 0)
                response_headers: dict[str, str] = {}
                if req_id:
                    response_headers["x-amzn-requestid"] = str(req_id)
                await opts.on_response(
                    {"status": http_status, "headers": response_headers},
                    model,
                )

            # 流式处理
            stream_data = response.get("stream")
            if stream_data is None:
                raise RuntimeError("Bedrock response missing stream")

            # 推送开始事件
            event_stream.push(AssistantMessageSnapshot(message=output))

            async for event in stream_data:
                if "messageStart" in event:
                    msg_start = event["messageStart"]
                    if msg_start.get("role") != "assistant":
                        raise RuntimeError(
                            "Unexpected assistant message start but got user message start instead"
                        )
                    # 已通过 AssistantMessageSnapshot 推送开始事件

                elif "contentBlockStart" in event:
                    _handle_content_block_start(
                        event["contentBlockStart"],
                        output.content,
                        output,
                        event_stream,
                    )

                elif "contentBlockDelta" in event:
                    _handle_content_block_delta(
                        event["contentBlockDelta"],
                        output.content,
                        output,
                        event_stream,
                    )

                elif "contentBlockStop" in event:
                    _handle_content_block_stop(
                        event["contentBlockStop"],
                        output.content,
                        output,
                        event_stream,
                    )

                elif "messageStop" in event:
                    msg_stop = event["messageStop"]
                    object.__setattr__(
                        output, "raw_stop_reason", msg_stop.get("stopReason")
                    )
                    mapped = _map_stop_reason(msg_stop.get("stopReason"))
                    output.stop_reason = mapped["stop_reason"]
                    if "error_message" in mapped:
                        output.error_message = mapped["error_message"]

                elif "metadata" in event:
                    _handle_metadata(event["metadata"], model, output)

                elif "internalServerException" in event:
                    raise event["internalServerException"]
                elif "modelStreamErrorException" in event:
                    raise event["modelStreamErrorException"]
                elif "validationException" in event:
                    raise event["validationException"]
                elif "throttlingException" in event:
                    raise event["throttlingException"]
                elif "serviceUnavailableException" in event:
                    raise event["serviceUnavailableException"]

            # 检查中止
            if opts.signal and getattr(opts.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Bedrock stream ended without a stop reason")
            if output.stop_reason in ("error", "aborted"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # 清理临时字段
            for block in output.content:
                if hasattr(block, "index"):
                    try:
                        delattr(block, "index")
                    except (AttributeError, TypeError):
                        pass
                if hasattr(block, "partial_json"):
                    try:
                        delattr(block, "partial_json")
                    except (AttributeError, TypeError):
                        pass

            output.stop_reason = (
                "aborted"
                if (opts.signal and getattr(opts.signal, "aborted", False))
                else "error"
            )
            output.error_message = _format_bedrock_error(error)
            if output.stop_reason == "error":
                _append_bedrock_failure_diagnostic(output, error, response_request_id)
            event_stream.push(
                AssistantErrorEvent(reason=output.stop_reason, error=output)
            )
            event_stream.end()

    import asyncio

    asyncio.ensure_future(_run())
    return event_stream


# ---------------------------------------------------------------------------
# stream_simple() - 简化接口
# ---------------------------------------------------------------------------


def stream_simple(
    model: Any,
    context: Context,
    options: Any | None = None,
) -> AssistantMessageEventStream:
    """简化的流式接口。"""
    base = build_base_options(model, context, options, None)

    if not options or not getattr(options, "reasoning", None):
        return stream(
            model,
            context,
            BedrockOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                reasoning=None,
            ),
        )

    if _is_anthropic_claude_model(model):
        model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
        model_name = getattr(model, "name", None)

        if _supports_adaptive_thinking(model_id, model_name):
            return stream(
                model,
                context,
                BedrockOptions(
                    **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                    reasoning=options.reasoning,
                    thinking_budgets=getattr(options, "thinking_budgets", None),
                ),
            )

        # Undefined means the caller did not request an output cap
        model_max_tokens = getattr(model, "max_tokens", 0) or getattr(
            model, "maxTokens", 0
        )
        adjusted = adjust_max_tokens_for_thinking(
            base.max_tokens,
            model_max_tokens,
            options.reasoning,
            getattr(options, "thinking_budgets", None),
        )

        max_tokens = clamp_max_tokens_to_context(model, context, adjusted[0])

        return stream(
            model,
            context,
            BedrockOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                max_tokens=max_tokens,
                reasoning=options.reasoning,
                thinking_budgets={
                    **(getattr(options, "thinking_budgets", None) or {}),
                    clamp_reasoning(options.reasoning): min(
                        adjusted[1], max(0, max_tokens - 1024)
                    ),
                },
            ),
        )

    return stream(
        model,
        context,
        BedrockOptions(
            **base.model_dump() if hasattr(base, "model_dump") else dict(base),
            reasoning=options.reasoning,
            thinking_budgets=getattr(options, "thinking_budgets", None),
        ),
    )
