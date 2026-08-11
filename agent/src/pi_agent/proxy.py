"""SSE 流式代理（对应 ``proxy.ts``）。

``streamProxy`` 将 LLM 请求通过 HTTP SSE 代理到 ``/api/stream`` 端点，
服务器管理认证并转发请求到 LLM 提供商。

客户端重建 partial assistant 消息，服务器剥离 ``partial`` 字段以减少带宽。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, Literal, TypeAlias

import orjson
from pydantic import BaseModel, ConfigDict, Field

from .cancellation import CancellationToken
from .types import (
    AssistantAbortedEvent,
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantStreamEnd,
    AssistantTextDelta,
    AssistantThinkingDelta,
    AssistantToolCallEnd,
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    ContentBlock,
    Context,
    Model,
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
)

# ---------------------------------------------------------------------------
# Proxy 事件（12 种，discriminated union）
# ---------------------------------------------------------------------------


class ProxyStartEvent(BaseModel):
    """流开始。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["start"] = "start"


class ProxyTextStartEvent(BaseModel):
    """文本块开始。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text_start"] = "text_start"
    content_index: int = Field(alias="contentIndex")


class ProxyTextDeltaEvent(BaseModel):
    """文本增量。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text_delta"] = "text_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str


class ProxyTextEndEvent(BaseModel):
    """文本块结束。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["text_end"] = "text_end"
    content_index: int = Field(alias="contentIndex")
    content_signature: str | None = Field(default=None, alias="contentSignature")


class ProxyThinkingStartEvent(BaseModel):
    """思考块开始。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int = Field(alias="contentIndex")


class ProxyThinkingDeltaEvent(BaseModel):
    """思考增量。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str


class ProxyThinkingEndEvent(BaseModel):
    """思考块结束。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int = Field(alias="contentIndex")
    content_signature: str | None = Field(default=None, alias="contentSignature")


class ProxyToolcallStartEvent(BaseModel):
    """工具调用开始。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int = Field(alias="contentIndex")
    id: str
    tool_name: str = Field(alias="toolName")


class ProxyToolcallDeltaEvent(BaseModel):
    """工具调用参数增量。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int = Field(alias="contentIndex")
    delta: str


class ProxyToolcallEndEvent(BaseModel):
    """工具调用结束。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int = Field(alias="contentIndex")


class ProxyDoneEvent(BaseModel):
    """流完成。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["done"] = "done"
    reason: StopReason
    usage: Usage | None = None


class ProxyErrorEvent(BaseModel):
    """流错误。"""

    model_config = ConfigDict(populate_by_name=True)

    type: Literal["error"] = "error"
    reason: StopReason
    error_message: str | None = Field(default=None, alias="errorMessage")
    usage: Usage | None = None


ProxyAssistantMessageEvent: TypeAlias = Annotated[
    ProxyStartEvent
    | ProxyTextStartEvent
    | ProxyTextDeltaEvent
    | ProxyTextEndEvent
    | ProxyThinkingStartEvent
    | ProxyThinkingDeltaEvent
    | ProxyThinkingEndEvent
    | ProxyToolcallStartEvent
    | ProxyToolcallDeltaEvent
    | ProxyToolcallEndEvent
    | ProxyDoneEvent
    | ProxyErrorEvent,
    Field(discriminator="type"),
]
"""代理事件判别联合（对应 TS ``ProxyAssistantMessageEvent``）。"""


# ---------------------------------------------------------------------------
# ProxyStreamOptions
# ---------------------------------------------------------------------------


class ProxyStreamOptions(BaseModel):
    """代理流式选项（对应 TS ``ProxyStreamOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    proxy_url: str = Field(alias="proxyUrl")
    """代理服务器 URL（如 ``https://genai.example.com``）。"""
    auth_token: str = Field(alias="authToken")
    """代理服务器认证 token。"""
    signal: CancellationToken | None = None
    """本地取消信号。"""

    # 可序列化的流式参数（对应 TS ``ProxySerializableStreamOptions``）
    temperature: float | None = None
    sampling_params: dict[str, object] | None = Field(
        default=None, alias="samplingParams"
    )
    max_tokens: int | None = Field(default=None, alias="maxTokens")
    reasoning: str | None = None
    cache_retention: str | None = Field(default=None, alias="cacheRetention")
    session_id: str | None = Field(default=None, alias="sessionId")
    headers: dict[str, str] | None = None
    metadata: dict[str, object] | None = None
    transport: object | None = None
    thinking_budgets: dict[str, object] | None = Field(
        default=None, alias="thinkingBudgets"
    )
    max_retry_delay_ms: int | None = Field(default=None, alias="maxRetryDelayMs")


# ---------------------------------------------------------------------------
# ProxyEventStream
# ---------------------------------------------------------------------------


class ProxyEventStream:
    """代理事件流：支持 push 生产和 async for 消费。

    对应 TS 的 ``ProxyMessageEventStream``（extends ``EventStream``）。
    调用方通过 ``async for`` 消费事件，完成后通过 ``final_message`` 获取最终消息。
    """

    def __init__(self) -> None:
        self._queue: asyncio.Queue[AssistantMessageEvent | None] = asyncio.Queue()
        self._final_message: AssistantMessage | None = None

    def push(self, event: AssistantMessageEvent) -> None:
        """向流中推送一个事件。"""
        self._queue.put_nowait(event)

    def end(self, final_message: AssistantMessage | None = None) -> None:
        """结束流，可选附带最终消息。"""
        self._final_message = final_message
        self._queue.put_nowait(None)  # sentinel

    @property
    def final_message(self) -> AssistantMessage | None:
        """流结束后的最终 assistant 消息。"""
        return self._final_message

    async def __aiter__(self) -> AsyncIterator[AssistantMessageEvent]:
        while True:
            event = await self._queue.get()
            if event is None:
                break
            yield event


# ---------------------------------------------------------------------------
# 增量 JSON 解析
# ---------------------------------------------------------------------------


def _parse_streaming_json(partial: str) -> dict[str, object]:
    """尝试将部分 JSON 字符串解析为 dict，失败返回空 dict。

    对应 TS 的 ``parseStreamingJson``（来自 pi-ai）。
    """
    try:
        result = orjson.loads(partial)
        if isinstance(result, dict):
            return result
        return {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def _build_proxy_request_options(options: ProxyStreamOptions) -> dict[str, object]:
    """构建发往代理服务器的请求选项（对应 TS ``buildProxyRequestOptions``）。"""
    payload: dict[str, object] = {}
    if options.temperature is not None:
        payload["temperature"] = options.temperature
    if options.sampling_params is not None:
        payload["samplingParams"] = options.sampling_params
    if options.max_tokens is not None:
        payload["maxTokens"] = options.max_tokens
    if options.reasoning is not None:
        payload["reasoning"] = options.reasoning
    if options.cache_retention is not None:
        payload["cacheRetention"] = options.cache_retention
    if options.session_id is not None:
        payload["sessionId"] = options.session_id
    if options.headers is not None:
        payload["headers"] = options.headers
    if options.metadata is not None:
        payload["metadata"] = options.metadata
    if options.transport is not None:
        payload["transport"] = options.transport
    if options.thinking_budgets is not None:
        payload["thinkingBudgets"] = options.thinking_budgets
    if options.max_retry_delay_ms is not None:
        payload["maxRetryDelayMs"] = options.max_retry_delay_ms
    return payload


def _process_proxy_event(
    proxy_event: ProxyAssistantMessageEvent,
    partial: dict[str, Any],
) -> AssistantMessageEvent | None:
    """处理代理事件，更新 partial 消息并返回 ``AssistantMessageEvent``。

    返回 ``None`` 表示该事件仅用于更新 partial 而不产生外部事件。
    """
    content_blocks: list[dict[str, Any]] = partial["content"]

    if isinstance(proxy_event, ProxyStartEvent):
        return None

    elif isinstance(proxy_event, ProxyTextStartEvent):
        content_blocks.append({"type": "text", "text": ""})
        return None

    elif isinstance(proxy_event, ProxyTextDeltaEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "text":
            raise ValueError("Received text_delta for non-text content")
        block["text"] += proxy_event.delta
        return AssistantTextDelta(delta=proxy_event.delta)

    elif isinstance(proxy_event, ProxyTextEndEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "text":
            raise ValueError("Received text_end for non-text content")
        block["content_signature"] = proxy_event.content_signature
        return None

    elif isinstance(proxy_event, ProxyThinkingStartEvent):
        content_blocks.append({"type": "thinking", "text": ""})
        return None

    elif isinstance(proxy_event, ProxyThinkingDeltaEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "thinking":
            raise ValueError("Received thinking_delta for non-thinking content")
        block["text"] += proxy_event.delta
        return AssistantThinkingDelta(delta=proxy_event.delta)

    elif isinstance(proxy_event, ProxyThinkingEndEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "thinking":
            raise ValueError("Received thinking_end for non-thinking content")
        block["thinking_signature"] = proxy_event.content_signature
        return None

    elif isinstance(proxy_event, ProxyToolcallStartEvent):
        content_blocks.append(
            {
                "type": "toolCall",
                "id": proxy_event.id,
                "name": proxy_event.tool_name,
                "arguments": {},
                "partial_json": "",
            }
        )
        return AssistantToolCallStart(
            tool_call_id=proxy_event.id,
            name=proxy_event.tool_name,
        )

    elif isinstance(proxy_event, ProxyToolcallDeltaEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "toolCall":
            raise ValueError("Received toolcall_delta for non-toolCall content")
        block["partial_json"] += proxy_event.delta
        block["arguments"] = _parse_streaming_json(block["partial_json"])
        content_blocks[proxy_event.content_index] = dict(block)
        return AssistantToolCallUpdate(
            tool_call_id=block["id"],
            args=block["arguments"],
        )

    elif isinstance(proxy_event, ProxyToolcallEndEvent):
        block = content_blocks[proxy_event.content_index]
        if block.get("type") != "toolCall":
            return None
        block.pop("partial_json", None)
        return AssistantToolCallEnd(
            tool_call_id=block["id"],
            content=[
                ToolCallContent(
                    tool_call_id=block["id"],
                    name=block["name"],
                    args=block.get("arguments", {}),
                )
            ],
        )

    elif isinstance(proxy_event, ProxyDoneEvent):
        partial["stop_reason"] = proxy_event.reason
        partial["usage"] = proxy_event.usage
        return AssistantStreamEnd()

    elif isinstance(proxy_event, ProxyErrorEvent):
        partial["stop_reason"] = proxy_event.reason
        partial["error_message"] = proxy_event.error_message
        partial["usage"] = proxy_event.usage
        if proxy_event.reason == "aborted":
            return AssistantAbortedEvent(error=proxy_event.error_message)
        return AssistantErrorEvent(
            error=proxy_event.error_message or "Unknown proxy error"
        )

    return None


def _build_assistant_message(partial: dict[str, Any], model: Model) -> AssistantMessage:
    """从 partial dict 构建最终的 ``AssistantMessage``。"""
    content: list[ContentBlock] = []
    for block in partial.get("content", []):
        block_type = block.get("type", "")
        if block_type == "text" or block_type == "thinking":
            content.append(TextContent(text=block.get("text", "")))
        elif block_type == "toolCall":
            content.append(
                ToolCallContent(
                    tool_call_id=block.get("id", ""),
                    name=block.get("name", ""),
                    args=block.get("arguments", {}),
                )
            )

    return AssistantMessage(
        role="assistant",
        content=content,
        api=model.api,
        provider=model.provider,
        model=model.model_id,
        usage=partial.get("usage"),
        stop_reason=partial.get("stop_reason", "pending"),
        error_message=partial.get("error_message"),
        timestamp=int(time.time() * 1000),
    )


async def stream_proxy(
    model: Model,
    context: Context,
    options: ProxyStreamOptions,
) -> ProxyEventStream:
    """通过代理服务器流式传输 LLM 请求（对应 TS ``streamProxy``）。

    用法：作为 ``streamFn`` 选项传给 Agent。

    Args:
        model: LLM 模型句柄。
        context: 发送给 LLM 的上下文。
        options: 代理配置（proxy_url、auth_token 等）。

    Returns:
        ``ProxyEventStream``：通过 ``async for`` 消费 ``AssistantMessageEvent``，
        完成后通过 ``stream.final_message`` 获取最终 ``AssistantMessage``。
    """
    stream = ProxyEventStream()

    # 初始化 partial 消息
    partial: dict[str, Any] = {
        "role": "assistant",
        "stop_reason": "pending",
        "content": [],
        "api": model.api,
        "provider": model.provider,
        "model": model.model_id,
        "usage": {
            "input": 0,
            "output": 0,
            "cache_read": 0,
            "cache_write": 0,
            "total_tokens": 0,
            "cost": {
                "input": 0.0,
                "output": 0.0,
                "cache_read": 0.0,
                "cache_write": 0.0,
                "total": 0.0,
            },
        },
        "timestamp": int(time.time() * 1000),
    }

    asyncio.ensure_future(_run_proxy_request(model, context, options, stream, partial))
    return stream


async def _run_proxy_request(
    model: Model,
    context: Context,
    options: ProxyStreamOptions,
    stream: ProxyEventStream,
    partial: dict[str, Any],
) -> None:
    """异步执行代理 HTTP 请求（内部实现）。"""
    try:
        import httpx
    except ImportError:
        stream.push(
            AssistantErrorEvent(
                error="httpx is required for proxy support. Install with: pip install httpx"
            )
        )
        stream.end()
        return

    async def _abort_handler() -> None:
        """取消信号处理：由 httpx 的 cancel 机制触发。"""

    if options.signal is not None:
        # 注册取消监听
        pass  # httpx 通过 cancel scope 处理取消

    try:
        request_body = {
            "model": {
                "api": model.api,
                "provider": model.provider,
                "id": model.model_id,
            },
            "context": {
                "messages": [msg.model_dump(mode="json") for msg in context.messages],
                "systemPrompt": context.system_prompt,
                "tools": [t.model_dump(mode="json") for t in (context.tools or [])],
                "sessionId": context.session_id,
                "thinkingLevel": context.thinking_level,
            },
            "options": _build_proxy_request_options(options),
        }

        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream(
                "POST",
                f"{options.proxy_url}/api/stream",
                headers={
                    "Authorization": f"Bearer {options.auth_token}",
                    "Content-Type": "application/json",
                },
                json=request_body,
            ) as response:
                if response.status_code != 200:
                    error_msg = (
                        f"Proxy error: {response.status_code} {response.reason_phrase}"
                    )
                    try:
                        error_data = response.json()
                        if isinstance(error_data, dict) and "error" in error_data:
                            error_msg = f"Proxy error: {error_data['error']}"
                    except Exception:
                        pass
                    partial["stop_reason"] = "error"
                    partial["error_message"] = error_msg
                    stream.push(AssistantErrorEvent(error=error_msg))
                    stream.end()
                    return

                buffer = ""
                async for chunk in response.aiter_bytes():
                    if options.signal is not None and options.signal.aborted:
                        raise asyncio.CancelledError("Request aborted by user")

                    buffer += chunk.decode("utf-8")
                    lines = buffer.split("\n")
                    buffer = lines.pop()

                    for line in lines:
                        if line.startswith("data: "):
                            data = line[6:].strip()
                            if not data:
                                continue
                            try:
                                proxy_raw = orjson.loads(data)
                                proxy_event = _parse_proxy_event(proxy_raw)
                                agent_event = _process_proxy_event(proxy_event, partial)
                                if agent_event is not None:
                                    stream.push(agent_event)
                            except Exception:
                                continue

                if options.signal is not None and options.signal.aborted:
                    raise asyncio.CancelledError("Request aborted by user")

                final_message = _build_assistant_message(partial, model)
                stream.end(final_message)

    except asyncio.CancelledError:
        reason: StopReason = "aborted"
        error_msg = "Request aborted by user"
        partial["stop_reason"] = reason
        partial["error_message"] = error_msg
        stream.push(AssistantAbortedEvent(error=error_msg))
        stream.end()
    except Exception as exc:
        error_reason: StopReason = "error"
        error_msg = str(exc)
        partial["stop_reason"] = error_reason
        partial["error_message"] = error_msg
        stream.push(AssistantErrorEvent(error=error_msg))
        stream.end()


def _parse_proxy_event(raw: dict[str, object]) -> ProxyAssistantMessageEvent:
    """将原始 JSON dict 解析为 ``ProxyAssistantMessageEvent`` 判别联合。

    根据 ``type`` 字段路由到对应的 Pydantic 模型。
    """
    event_type = raw.get("type")
    if not isinstance(event_type, str):
        raise ValueError(f"Missing or invalid 'type' in proxy event: {raw}")

    match event_type:
        case "start":
            return ProxyStartEvent.model_validate(raw)
        case "text_start":
            return ProxyTextStartEvent.model_validate(raw)
        case "text_delta":
            return ProxyTextDeltaEvent.model_validate(raw)
        case "text_end":
            return ProxyTextEndEvent.model_validate(raw)
        case "thinking_start":
            return ProxyThinkingStartEvent.model_validate(raw)
        case "thinking_delta":
            return ProxyThinkingDeltaEvent.model_validate(raw)
        case "thinking_end":
            return ProxyThinkingEndEvent.model_validate(raw)
        case "toolcall_start":
            return ProxyToolcallStartEvent.model_validate(raw)
        case "toolcall_delta":
            return ProxyToolcallDeltaEvent.model_validate(raw)
        case "toolcall_end":
            return ProxyToolcallEndEvent.model_validate(raw)
        case "done":
            return ProxyDoneEvent.model_validate(raw)
        case "error":
            return ProxyErrorEvent.model_validate(raw)
        case _:
            raise ValueError(f"Unknown proxy event type: {event_type}")
