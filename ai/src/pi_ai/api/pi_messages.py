"""pi-messages API implementation.

Streams pi's own message protocol directly to a backend: the request is a
single POST of ``{ model, context, options }`` to ``<baseUrl>/messages``, the
response is an SSE stream of serialized assistant-message events plus a
terminal ``done``/``error`` event. This is the wire protocol spoken by the
Radius gateway, but any backend implementing it can be used, e.g. via a
models.json custom provider with ``"api": "pi-messages"``.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias

import httpx
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
    AssistantToolCallUpdate,
    Context,
    Cost,
    StreamOptions,
    ToolCallContent,
    Usage,
)
from ..utils.diagnostics import (
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.headers import provider_headers_to_record
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value

# ---------------------------------------------------------------------------
# PiMessagesOptions
# ---------------------------------------------------------------------------


class PiMessagesOptions(StreamOptions):
    """pi-messages 流式选项。"""

    reasoning: Any | None = None  # ThinkingLevel
    tool_choice: Literal["auto", "none", "required"] | dict[str, Any] | None = None
    debug: bool | None = None
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


# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

PiMessagesUsage: TypeAlias = dict[str, Any]
"""pi-messages 用量类型。"""

PiMessagesStopReason: TypeAlias = str
"""pi-messages stop reason 类型。"""


# ---------------------------------------------------------------------------
# PiMessagesRewriteImpact
# ---------------------------------------------------------------------------


class PiMessagesRewriteImpact(BaseModel):
    """服务端消息重写影响摘要。"""

    policy_id: str
    policy_version: int
    changed: bool
    token_count_change: int
    message_count_change: int
    system_prompt_changed: bool


# ---------------------------------------------------------------------------
# PiMessagesEvent 类型
# ---------------------------------------------------------------------------


class PiMessagesEventStart(BaseModel):
    """开始事件。"""

    type: Literal["start"] = "start"


class PiMessagesEventTextStart(BaseModel):
    """文本开始事件。"""

    type: Literal["text_start"] = "text_start"
    content_index: int


class PiMessagesEventTextDelta(BaseModel):
    """文本增量事件。"""

    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str


class PiMessagesEventTextEnd(BaseModel):
    """文本结束事件。"""

    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str
    content_signature: str | None = None


class PiMessagesEventThinkingStart(BaseModel):
    """思考开始事件。"""

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int


class PiMessagesEventThinkingDelta(BaseModel):
    """思考增量事件。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str


class PiMessagesEventThinkingEnd(BaseModel):
    """思考结束事件。"""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str
    content_signature: str | None = None


class PiMessagesEventToolCallStart(BaseModel):
    """工具调用开始事件。"""

    type: Literal["tool_call_start"] = "tool_call_start"
    content_index: int


class PiMessagesEventToolCallDelta(BaseModel):
    """工具调用增量事件。"""

    type: Literal["tool_call_delta"] = "tool_call_delta"
    content_index: int
    delta: str


class PiMessagesEventToolCallEnd(BaseModel):
    """工具调用结束事件。"""

    type: Literal["tool_call_end"] = "tool_call_end"
    content_index: int
    tool_call_id: str
    name: str
    args: str


class PiMessagesEventRewrite(BaseModel):
    """重写事件。"""

    type: Literal["rewrite"] = "rewrite"
    impact: PiMessagesRewriteImpact


class PiMessagesEventDone(BaseModel):
    """完成事件。"""

    type: Literal["done"] = "done"
    usage: dict[str, Any]
    stop_reason: str


class PiMessagesEventError(BaseModel):
    """错误事件。"""

    type: Literal["error"] = "error"
    error: str
    status_code: int | None = None


PiMessagesEvent = (
    PiMessagesEventStart
    | PiMessagesEventTextStart
    | PiMessagesEventTextDelta
    | PiMessagesEventTextEnd
    | PiMessagesEventThinkingStart
    | PiMessagesEventThinkingDelta
    | PiMessagesEventThinkingEnd
    | PiMessagesEventToolCallStart
    | PiMessagesEventToolCallDelta
    | PiMessagesEventToolCallEnd
    | PiMessagesEventRewrite
    | PiMessagesEventDone
    | PiMessagesEventError
)
"""pi-messages 事件联合类型。"""


# ---------------------------------------------------------------------------
# PiMessagesResponseError
# ---------------------------------------------------------------------------


class PiMessagesResponseError(Exception):
    """pi-messages 响应错误。"""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        self.status_code = status_code
        super().__init__(message)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _camel_to_snake(name: str) -> str:
    """将 camelCase 转换为 snake_case。"""
    result: list[str] = []
    for i, char in enumerate(name):
        if char.isupper():
            if i > 0 and name[i - 1].islower():
                result.append("_")
            result.append(char.lower())
        else:
            result.append(char)
    return "".join(result)


def _convert_event_dict(data: dict[str, Any]) -> dict[str, Any]:
    """将 camelCase 事件字典转换为 snake_case。"""
    return {_camel_to_snake(k): v for k, v in data.items()}


def get_client_api_key(
    provider: str,
    api_key: str | None,
    headers: dict[str, str | None] | None,
) -> str:
    """获取客户端 API key。"""
    if api_key:
        return api_key
    if headers and has_header(headers, "authorization"):
        return "unused"
    if headers and has_header(headers, "x-api-key"):
        return "unused"
    raise ValueError(f"No API key provided for provider: {provider}")


def has_header(
    headers: dict[str, str | None] | None,
    name: str,
) -> bool:
    """检查请求头是否存在且非空。"""
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def format_pi_messages_error(error: object) -> str:
    """格式化 pi-messages 错误。"""
    if isinstance(error, PiMessagesResponseError):
        return str(error)
    return f"pi-messages API error: {error}"


def build_url(model: Any) -> str:
    """构建请求 URL。"""
    base_url = getattr(model, "base_url", "") or ""
    base_url = base_url.rstrip("/")
    return f"{base_url}/messages"


def build_params(
    model: Any,
    context: Context,
    options: PiMessagesOptions | None = None,
) -> dict[str, Any]:
    """构建请求参数。"""
    payload: dict[str, Any] = {
        "model": getattr(model, "model_id", "") or getattr(model, "id", ""),
        "context": context.model_dump() if hasattr(context, "model_dump") else {},
        "options": {
            "temperature": options.temperature if options else None,
            "maxTokens": options.max_tokens if options else None,
            "reasoning": options.reasoning if options else None,
            "cacheRetention": _resolve_cache_retention(
                options.cache_retention if options else None,
                options.env if options else None,
            ),
            "sessionId": options.session_id if options else None,
            "toolChoice": options.tool_choice if options else None,
        },
    }
    return payload


def _resolve_cache_retention(
    cache_retention: str | None,
    env: dict[str, str] | None = None,
) -> str | None:
    """解析缓存保留策略。"""
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return None


def deserialize_pi_message_event(line: str) -> PiMessagesEvent | None:
    """反序列化 SSE 事件行。

    Args:
        line: 以 ``data: `` 开头的 SSE 事件行。

    Returns:
        解析后的事件对象，无效行返回 ``None``。
    """
    if not line.startswith("data:"):
        return None
    data = line[5:].strip()
    if not data or data == "[DONE]":
        return None

    try:
        raw = json.loads(data)
    except json.JSONDecodeError:
        return None

    if not isinstance(raw, dict):
        return None

    event_type = raw.get("type")
    if not isinstance(event_type, str):
        return None

    converted = _convert_event_dict(raw)

    # 根据 type 选择正确的模型
    if event_type == "start":
        return PiMessagesEventStart(**converted)
    elif event_type == "text_start":
        return PiMessagesEventTextStart(**converted)
    elif event_type == "text_delta":
        return PiMessagesEventTextDelta(**converted)
    elif event_type == "text_end":
        return PiMessagesEventTextEnd(**converted)
    elif event_type == "thinking_start":
        return PiMessagesEventThinkingStart(**converted)
    elif event_type == "thinking_delta":
        return PiMessagesEventThinkingDelta(**converted)
    elif event_type == "thinking_end":
        return PiMessagesEventThinkingEnd(**converted)
    elif event_type == "tool_call_start" or event_type == "toolcall_start":
        return PiMessagesEventToolCallStart(**converted)
    elif event_type == "tool_call_delta" or event_type == "toolcall_delta":
        return PiMessagesEventToolCallDelta(**converted)
    elif event_type == "tool_call_end" or event_type == "toolcall_end":
        return PiMessagesEventToolCallEnd(**converted)
    elif event_type == "rewrite":
        return PiMessagesEventRewrite(**converted)
    elif event_type == "done":
        return PiMessagesEventDone(**converted)
    elif event_type == "error":
        return PiMessagesEventError(**converted)
    else:
        return None


def _create_empty_usage() -> dict[str, Any]:
    """创建空用量。"""
    return {
        "input": 0,
        "output": 0,
        "cacheRead": 0,
        "cacheWrite": 0,
        "totalTokens": 0,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": 0,
        },
    }


def _create_pi_usage(usage_data: dict[str, Any]) -> dict[str, Any]:
    """创建 pi-messages 用量。"""
    input_tokens = usage_data.get("input", 0) or usage_data.get("inputTokens", 0)
    output_tokens = usage_data.get("output", 0) or usage_data.get("outputTokens", 0)
    cache_read = usage_data.get("cacheRead", 0) or usage_data.get(
        "cacheReadInputTokens", 0
    )
    cache_write = usage_data.get("cacheWrite", 0) or usage_data.get(
        "cacheWriteInputTokens", 0
    )
    total = input_tokens + output_tokens + cache_read + cache_write

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cacheRead": cache_read,
        "cacheWrite": cache_write,
        "totalTokens": total,
        "cost": {
            "input": 0,
            "output": 0,
            "cacheRead": 0,
            "cacheWrite": 0,
            "total": 0,
        },
    }


def _append_rewrite_diagnostic(
    message: AssistantMessage,
    rewrite: PiMessagesRewriteImpact | None,
) -> None:
    """追加重写诊断信息。"""
    if not rewrite:
        return
    append_assistant_message_diagnostic(
        message.model_dump(),
        create_assistant_message_diagnostic(
            "pi_messages_rewrite",
            None,
            details=rewrite.model_dump(),
        ),
    )


def _create_error_event(
    model: Any,
    error: object,
    aborted: bool,
) -> AssistantErrorEvent:
    """创建错误事件。"""
    reason = "aborted" if aborted else "error"
    assistant_message = AssistantMessage(
        role="assistant",
        content=[],
        api=getattr(model, "api", ""),
        provider=getattr(model, "provider", ""),
        model=getattr(model, "model_id", ""),
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
        stop_reason=reason,
        error_message=str(error) if not isinstance(error, str) else error,
        timestamp=int(time.time() * 1000),
    )

    if not aborted and isinstance(error, PiMessagesResponseError):
        append_assistant_message_diagnostic(
            assistant_message.model_dump(),
            create_assistant_message_diagnostic(
                "pi_messages_response_failure",
                error,
                details={"status_code": error.status_code}
                if error.status_code
                else None,
            ),
        )

    return AssistantErrorEvent(reason=reason, error=assistant_message)


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: PiMessagesOptions | None = None,
) -> AssistantMessageEventStream:
    """pi-messages API 流式生成函数。"""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=getattr(model, "api", ""),
            provider=getattr(model, "provider", ""),
            model=getattr(model, "model_id", ""),
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
        usage = output.usage
        assert usage is not None

        try:
            provider = getattr(model, "provider", "")
            api_key = options.api_key if options else None
            api_key = get_client_api_key(
                provider, api_key, options.headers if options else None
            )

            url = build_url(model)
            if options and options.debug:
                url += "?debug=1"

            payload = build_params(model, context, options)

            # onPayload 回调允许修改参数
            if options and options.on_payload:
                next_payload = await options.on_payload(dict(payload), model)
                if next_payload is not None:
                    payload = next_payload

            # 构建请求头
            request_headers: dict[str, str] = {
                "authorization": f"Bearer {api_key}",
                "accept": "text/event-stream",
                "content-type": "application/json",
            }
            provider_headers = provider_headers_to_record(
                options.headers if options else None
            )
            if provider_headers:
                request_headers.update(provider_headers)

            # 发起 HTTP 请求
            request_timeout = None
            if options and options.timeout_ms is not None:
                request_timeout = options.timeout_ms / 1000.0

            async with httpx.AsyncClient(timeout=request_timeout) as client:
                response = await client.post(
                    url,
                    headers=request_headers,
                    content=json.dumps(payload),
                )

                # onResponse 回调
                if options and options.on_response:
                    await options.on_response(
                        {
                            "status": response.status_code,
                            "headers": dict(response.headers),
                        },
                        model,
                    )

                if response.status_code != 200:
                    body = await response.aread()
                    body_text = body.decode("utf-8", errors="replace")
                    raise PiMessagesResponseError(
                        f"{response.status_code}: {body_text}",
                        status_code=response.status_code,
                    )

                # 初始化输出事件
                event_stream.push(AssistantMessageSnapshot(message=output))

                # 跟踪工具调用的 JSON 积累
                tool_json_map: dict[int, str] = {}

                # 读取 SSE 流
                buffer = ""
                async for chunk in response.aiter_bytes():
                    chunk_text = chunk.decode("utf-8", errors="replace")
                    buffer += chunk_text
                    buffer = buffer.replace("\r\n", "\n")

                    # 按 \n\n 分割事件
                    while "\n\n" in buffer:
                        event_block, buffer = buffer.split("\n\n", 1)
                        pi_event = deserialize_pi_message_event(event_block)
                        if pi_event is None:
                            continue

                        # 处理各类事件
                        if isinstance(pi_event, PiMessagesEventStart):
                            pass  # 初始化，无需操作

                        elif isinstance(pi_event, PiMessagesEventTextStart):
                            pass  # 文本块将在 delta 中创建

                        elif isinstance(pi_event, PiMessagesEventTextDelta):
                            ci = pi_event.content_index
                            if ci >= len(output.content):
                                # 创建新的文本内容块
                                from ..types import TextContent

                                output.content.append(TextContent(text=""))
                            block = output.content[ci]
                            if hasattr(block, "text"):
                                block.text += pi_event.delta
                            event_stream.push(AssistantTextDelta(delta=pi_event.delta))

                        elif isinstance(pi_event, PiMessagesEventTextEnd):
                            ci = pi_event.content_index
                            if ci < len(output.content):
                                block = output.content[ci]
                                if hasattr(block, "text"):
                                    block.text = pi_event.content

                        elif isinstance(pi_event, PiMessagesEventThinkingStart):
                            pass  # 思考块将在 delta 中创建

                        elif isinstance(pi_event, PiMessagesEventThinkingDelta):
                            ci = pi_event.content_index
                            if output.thinking is None:
                                output.thinking = []
                            if ci >= len(output.thinking):
                                from ..types import ThinkingBlock

                                output.thinking.append(
                                    ThinkingBlock(text="", signature=None)
                                )
                            think_block = output.thinking[ci]
                            think_block.text += pi_event.delta
                            event_stream.push(
                                AssistantThinkingDelta(delta=pi_event.delta)
                            )

                        elif isinstance(pi_event, PiMessagesEventThinkingEnd):
                            ci = pi_event.content_index
                            if output.thinking and ci < len(output.thinking):
                                output.thinking[ci].text = pi_event.content

                        elif isinstance(pi_event, PiMessagesEventToolCallStart):
                            output.content.append(
                                ToolCallContent(
                                    tool_call_id="",
                                    name="",
                                    args={},
                                )
                            )
                            tool_json_map[pi_event.content_index] = ""

                        elif isinstance(pi_event, PiMessagesEventToolCallDelta):
                            ci = pi_event.content_index
                            current_json = tool_json_map.get(ci, "")
                            current_json += pi_event.delta
                            tool_json_map[ci] = current_json
                            args = parse_streaming_json(current_json)
                            if ci < len(output.content):
                                block = output.content[ci]
                                if isinstance(block, ToolCallContent):
                                    block.args = args
                                    event_stream.push(
                                        AssistantToolCallUpdate(
                                            tool_call_id=block.tool_call_id,
                                            args=args,
                                        )
                                    )

                        elif isinstance(pi_event, PiMessagesEventToolCallEnd):
                            ci = pi_event.content_index
                            if ci < len(output.content):
                                block = output.content[ci]
                                if isinstance(block, ToolCallContent):
                                    block.tool_call_id = pi_event.tool_call_id
                                    block.name = pi_event.name
                                    block.args = parse_streaming_json(pi_event.args)
                                    tool_json_map.pop(ci, None)
                                    event_stream.push(
                                        AssistantToolCallEnd(
                                            tool_call_id=block.tool_call_id,
                                            content=[block],
                                        )
                                    )

                        elif isinstance(pi_event, PiMessagesEventRewrite):
                            # 记录重写影响
                            append_assistant_message_diagnostic(
                                output.model_dump(),
                                create_assistant_message_diagnostic(
                                    "pi_messages_rewrite",
                                    None,
                                    details=pi_event.impact.model_dump(),
                                ),
                            )

                        elif isinstance(pi_event, PiMessagesEventDone):
                            # 完成事件
                            stop_reason = pi_event.stop_reason
                            # 映射到标准 stop reason
                            stop_reason_map: dict[str, str] = {
                                "stop": "stop",
                                "length": "length",
                                "tool_use": "tool_use",
                                "toolUse": "tool_use",
                                "max_tokens": "length",
                                "error": "error",
                                "aborted": "aborted",
                            }
                            mapped_reason = stop_reason_map.get(stop_reason, "stop")
                            object.__setattr__(output, "stop_reason", mapped_reason)
                            pi_usage = _create_pi_usage(pi_event.usage)
                            usage.input = pi_usage.get("input", 0)
                            usage.output = pi_usage.get("output", 0)
                            usage.cache_read = pi_usage.get("cacheRead", 0)
                            usage.cache_write = pi_usage.get("cacheWrite", 0)
                            usage.total_tokens = (
                                usage.input
                                + usage.output
                                + usage.cache_read
                                + usage.cache_write
                            )
                            calculate_cost(model, usage)

                            event_stream.push(
                                AssistantStreamEnd(
                                    reason=mapped_reason,
                                    message=output,
                                )
                            )
                            event_stream.end()
                            return

                        elif isinstance(pi_event, PiMessagesEventError):
                            # 错误事件
                            output.stop_reason = "error"
                            output.error_message = pi_event.error
                            event_stream.push(
                                AssistantErrorEvent(
                                    reason="error",
                                    error=output,
                                )
                            )
                            event_stream.end()
                            return

                # 流结束，未收到终端事件
                if output.stop_reason == "pending":
                    raise RuntimeError(
                        f"{provider} stream ended without a terminal event"
                    )

        except Exception as error:
            aborted = bool(
                options and options.signal and getattr(options.signal, "aborted", False)
            )
            event_stream.push(
                _create_error_event(
                    model,
                    error,
                    aborted,
                )
            )
            event_stream.end()

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
    """简化的 pi-messages 流式接口。""" 
    extra = options if options else None
    return stream(
        model,
        context,
        PiMessagesOptions(
            api_key=getattr(options, "api_key", None) if options else None,
            headers=getattr(options, "headers", None) if options else None,
            reasoning=getattr(options, "reasoning", None) if options else None,
            tool_choice=getattr(extra, "tool_choice", None) if extra else None,
            debug=getattr(extra, "debug", None) if extra else None,
        ),
    )
