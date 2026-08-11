"""Faux（假）Provider，用于测试（对应 ``faux.ts``）。

提供 ``faux_provider`` 工厂函数，创建可脚本化响应的测试 provider。
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from ..models import CreateProviderOptions, create_provider
from ..types import (
    AssistantAbortedEvent,
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    AssistantTextDelta,
    AssistantThinkingDelta,
    AssistantToolCallEnd,
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    ContentBlock,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    ImageContent,
    SimpleStreamOptions,
    StreamOptions,
    TextContent,
    ThinkingBlock,
    ToolCallContent,
    ToolResultMessage,
    Usage,
)
from ..utils.event_stream import AssistantMessageEventStream

# ---------------------------------------------------------------------------
# 默认值
# ---------------------------------------------------------------------------

DEFAULT_API = "faux"
DEFAULT_PROVIDER = "faux"
DEFAULT_MODEL_ID = "faux-1"
DEFAULT_MODEL_NAME = "Faux Model"
DEFAULT_BASE_URL = "http://localhost:0"
DEFAULT_MIN_TOKEN_SIZE = 3
DEFAULT_MAX_TOKEN_SIZE = 5

DEFAULT_USAGE: Usage = Usage(
    input=0, output=0, cache_read=0, cache_write=0, total_tokens=0
)

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


@dataclass
class FauxModelDefinition:
    """假模型定义。"""

    id: str
    name: str | None = None
    reasoning: bool = False
    input: tuple[str, ...] = ("text", "image")
    cost: dict[str, float] | None = None
    context_window: int = 128000
    max_tokens: int = 16384


@dataclass
class FauxProviderState:
    """假 provider 状态。"""

    call_count: int = 0
    deferred_fetch_count: int = 0
    cancelled_deferred: list[DeferredHandle] = field(default_factory=list)


FauxResponseStep = AssistantMessage | Callable[..., AssistantMessage | Any]


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def faux_text(text: str) -> TextContent:
    """创建假文本内容块。"""
    return TextContent(text=text)


def faux_thinking(text: str) -> ThinkingBlock:
    """创建假思考内容块。"""
    return ThinkingBlock(text=text)


def faux_tool_call(
    name: str,
    args: dict[str, object],
    *,
    tool_call_id: str | None = None,
) -> ToolCallContent:
    """创建假工具调用内容块。"""
    return ToolCallContent(
        tool_call_id=tool_call_id
        or f"tool:{int(time.time() * 1000)}:{hex(int(time.time()))[2:]}",
        name=name,
        args=args,
    )


def _normalize_faux_content(
    content: str | ContentBlock | list[ContentBlock],
) -> list[ContentBlock]:
    """规范化假内容到列表格式。"""
    if isinstance(content, str):
        return [faux_text(content)]
    if isinstance(content, list):
        return content
    return [content]


def faux_assistant_message(
    content: str | ContentBlock | list[ContentBlock],
    *,
    stop_reason: str = "stop",
    thinking: list[ThinkingBlock] | None = None,
    deferred: DeferredHandle | None = None,
    error_message: str | None = None,
    response_id: str | None = None,
    timestamp: int | None = None,
) -> AssistantMessage:
    """创建假助手消息。"""
    return AssistantMessage(
        content=_normalize_faux_content(content),
        api=DEFAULT_API,
        provider=DEFAULT_PROVIDER,
        model=DEFAULT_MODEL_ID,
        usage=DEFAULT_USAGE,
        stop_reason=cast(Any, stop_reason),
        thinking=thinking,
        deferred=deferred,
        error_message=error_message,
        response_id=response_id,
        timestamp=timestamp or int(time.time() * 1000),
    )


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------


def _estimate_tokens(text: str) -> int:
    """估算 token 数。"""
    return max(1, math.ceil(len(text) / 4))


def _random_id(prefix: str) -> str:
    """生成随机 ID。"""
    return f"{prefix}:{int(time.time() * 1000)}:{hex(int(time.time() * 16384))[2:]}"


def _content_to_text(content: str | list[TextContent | ImageContent]) -> str:
    """将内容块转换为文本。"""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ImageContent):
            parts.append(
                f"[image:{block.mime_type}:{len(block.data) if hasattr(block, 'data') else 0}]"
            )
    return "\n".join(parts)


def _assistant_content_to_text(content: list[ContentBlock]) -> str:
    """将助手内容块转换为文本。"""
    import json

    parts: list[str] = []
    for block in content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolCallContent):
            parts.append(f"{block.name}:{json.dumps(block.args)}")
    return "\n".join(parts)


def _tool_result_to_text(message: ToolResultMessage) -> str:
    """将工具结果消息转换为文本。"""
    tool_name = getattr(message, "tool_name", "")
    content_blocks = getattr(message, "content", [])
    text = (
        _content_to_text(content_blocks)
        if isinstance(content_blocks, list)
        else str(content_blocks)
    )
    return f"{tool_name}\n{text}"


def _message_to_text(message: Any) -> str:
    """将消息转换为文本。"""
    role = getattr(message, "role", "unknown")
    content = getattr(message, "content", "")
    if role == "user":
        return (
            _content_to_text(content)
            if isinstance(content, (str, list))
            else str(content)
        )
    if role == "assistant":
        return (
            _assistant_content_to_text(content)
            if isinstance(content, list)
            else str(content)
        )
    if role == "tool_result":
        return _tool_result_to_text(message)
    return str(content)


def _serialize_context(context: Context) -> str:
    """序列化上下文为文本。"""
    parts: list[str] = []
    if context.system_prompt:
        parts.append(f"system:{context.system_prompt}")
    for msg in context.messages:
        parts.append(f"{msg.role}:{_message_to_text(msg)}")
    if context.tools:
        import json

        parts.append(
            f"tools:{json.dumps([t.model_dump() if hasattr(t, 'model_dump') else dict(t) for t in context.tools])}"
        )
    return "\n\n".join(parts)


def _common_prefix_length(a: str, b: str) -> int:
    """计算公共前缀长度。"""
    length = min(len(a), len(b))
    idx = 0
    while idx < length and a[idx] == b[idx]:
        idx += 1
    return idx


def _with_usage_estimate(
    message: AssistantMessage,
    context: Context,
    options: SimpleStreamOptions | None,
    prompt_cache: dict[str, str],
) -> AssistantMessage:
    """估算并添加用量信息。"""
    prompt_text = _serialize_context(context)
    prompt_tokens = _estimate_tokens(prompt_text)
    output_tokens = _estimate_tokens(
        _assistant_content_to_text(
            getattr(message, "content", [])
            if isinstance(getattr(message, "content", None), list)
            else []
        )
    )
    input_tokens = prompt_tokens
    cache_read = 0
    cache_write = 0
    session_id = getattr(options, "session_id", None) if options else None

    if session_id:
        cache_retention = getattr(options, "cache_retention", None) if options else None
        if cache_retention != "none":
            previous = prompt_cache.get(session_id)
            if previous:
                cached = _common_prefix_length(previous, prompt_text)
                cache_read = _estimate_tokens(previous[:cached])
                cache_write = _estimate_tokens(prompt_text[cached:])
                input_tokens = max(0, prompt_tokens - cache_read)
            else:
                cache_write = prompt_tokens
            prompt_cache[session_id] = prompt_text

    message.usage = Usage(
        input=input_tokens,
        output=output_tokens,
        cache_read=cache_read,
        cache_write=cache_write,
        total_tokens=input_tokens + output_tokens + cache_read + cache_write,
    )
    return message


def _split_string_by_token_size(text: str, min_size: int, max_size: int) -> list[str]:
    """按 token 大小拆分字符串。"""
    chunks: list[str] = []
    idx = 0
    import random

    while idx < len(text):
        token_size = min_size + random.randint(0, max_size - min_size)
        char_size = max(1, token_size * 4)
        chunks.append(text[idx : idx + char_size])
        idx += char_size
    return chunks or [""]


def _clone_message(
    message: AssistantMessage,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    """克隆消息并更新元数据。"""
    import copy

    cloned = copy.deepcopy(message)
    cloned.api = api
    cloned.provider = provider
    cloned.model = model_id
    cloned.timestamp = getattr(cloned, "timestamp", None) or int(time.time() * 1000)
    cloned.usage = getattr(cloned, "usage", None) or DEFAULT_USAGE
    return cloned


def _create_deferred_message(model: Any, handle: DeferredHandle) -> AssistantMessage:
    """创建延迟响应消息。"""
    return AssistantMessage(
        content=[],
        api=getattr(model, "api", DEFAULT_API),
        provider=getattr(model, "provider", DEFAULT_PROVIDER),
        model=getattr(model, "model_id", getattr(model, "id", DEFAULT_MODEL_ID)),
        usage=DEFAULT_USAGE,
        stop_reason="deferred",
        deferred=handle,
        timestamp=int(time.time() * 1000),
    )


def _create_error_message(
    error: Exception,
    api: str,
    provider: str,
    model_id: str,
) -> AssistantMessage:
    """创建错误消息。"""
    return AssistantMessage(
        content=[],
        api=api,
        provider=provider,
        model=model_id,
        usage=DEFAULT_USAGE,
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


def _create_aborted_message(partial: AssistantMessage) -> AssistantMessage:
    """创建中止消息。"""
    partial.stop_reason = "aborted"
    partial.error_message = "Request was aborted"
    partial.timestamp = int(time.time() * 1000)
    return partial


async def _stream_with_deltas(
    stream: AssistantMessageEventStream,
    message: AssistantMessage,
    min_size: int,
    max_size: int,
    tokens_per_second: float | None,
    signal: Any | None,
) -> None:
    """带 delta 事件流式输出消息。"""
    partial = AssistantMessage(
        content=[],
        api=message.api,
        provider=message.provider,
        model=message.model,
        stop_reason="pending",
        timestamp=int(time.time() * 1000),
    )

    if signal and getattr(signal, "aborted", False):
        aborted = _create_aborted_message(partial)
        stream.push(AssistantErrorEvent(reason="aborted", error=aborted))
        stream.end(aborted)
        return

    stream.push(AssistantMessageSnapshot(message=partial))

    for idx, block in enumerate(getattr(message, "content", [])):
        if signal and getattr(signal, "aborted", False):
            aborted = _create_aborted_message(partial)
            stream.push(AssistantAbortedEvent(error="aborted"))
            stream.end(aborted)
            return

        if isinstance(block, TextContent):
            partial.content.append(TextContent(text=""))
            for chunk in _split_string_by_token_size(block.text, min_size, max_size):
                if tokens_per_second and tokens_per_second > 0:
                    delay = (_estimate_tokens(chunk) / tokens_per_second) * 1000
                    await asyncio.sleep(delay / 1000)
                if signal and getattr(signal, "aborted", False):
                    aborted = _create_aborted_message(partial)
                    stream.push(AssistantAbortedEvent(error="aborted"))
                    stream.end(aborted)
                    return
                partial.content[idx].text += chunk  # type: ignore[union-attr]
                stream.push(AssistantTextDelta(delta=chunk))
            continue

        if isinstance(block, ToolCallContent):
            import json

            partial.content.append(
                ToolCallContent(
                    tool_call_id=block.tool_call_id, name=block.name, args={}
                )
            )
            stream.push(
                AssistantToolCallStart(tool_call_id=block.tool_call_id, name=block.name)
            )
            args_str = json.dumps(block.args)
            for _chunk in _split_string_by_token_size(args_str, min_size, max_size):
                if tokens_per_second and tokens_per_second > 0:
                    delay = (_estimate_tokens(str(_chunk)) / tokens_per_second) * 1000
                    await asyncio.sleep(delay / 1000)
                if signal and getattr(signal, "aborted", False):
                    aborted = _create_aborted_message(partial)
                    stream.push(AssistantAbortedEvent(error="aborted"))
                    stream.end(aborted)
                    return
                stream.push(
                    AssistantToolCallUpdate(tool_call_id=block.tool_call_id, args={})
                )
            partial.content[idx].args = block.args  # type: ignore[union-attr]
            stream.push(
                AssistantToolCallEnd(tool_call_id=block.tool_call_id, content=[])
            )

    # 处理 thinking blocks（AssistantMessage 有独立的 thinking 字段）
    thinking = getattr(message, "thinking", None)
    if thinking:
        for think_block in thinking:
            if isinstance(think_block, ThinkingBlock):
                for chunk in _split_string_by_token_size(
                    think_block.text, min_size, max_size
                ):
                    if tokens_per_second and tokens_per_second > 0:
                        delay = (_estimate_tokens(chunk) / tokens_per_second) * 1000
                        await asyncio.sleep(delay / 1000)
                    if signal and getattr(signal, "aborted", False):
                        aborted = _create_aborted_message(partial)
                        stream.push(AssistantAbortedEvent(error="aborted"))
                        stream.end(aborted)
                        return
                    stream.push(AssistantThinkingDelta(delta=chunk))

    if message.stop_reason == "pending":
        raise RuntimeError("Faux response ended without a stop reason")
    if message.stop_reason in ("error", "aborted"):
        stream.push(AssistantErrorEvent(reason=message.stop_reason, error=message))
        stream.end(message)
        return

    stream.push(AssistantStreamEnd(reason=message.stop_reason, message=message))
    stream.end(message)


# ---------------------------------------------------------------------------
# Faux Provider 核心
# ---------------------------------------------------------------------------


def _create_faux_core(
    *,
    api: str | None = None,
    provider: str | None = None,
    models: list[FauxModelDefinition] | None = None,
    deferred_pending_fetches: int = 0,
    deferred_poll_after_ms: int | None = None,
    tokens_per_second: float | None = None,
    token_size_min: int = DEFAULT_MIN_TOKEN_SIZE,
    token_size_max: int = DEFAULT_MAX_TOKEN_SIZE,
) -> Any:
    """创建 Faux Provider 核心。"""
    resolved_api = api or _random_id(DEFAULT_API)
    resolved_provider = provider or DEFAULT_PROVIDER
    min_size = max(1, min(token_size_min, token_size_max))
    max_size = max(min_size, token_size_max)

    pending_responses: list[FauxResponseStep] = []
    state = FauxProviderState()
    prompt_cache: dict[str, str] = {}
    deferred_responses: dict[str, Any] = {}

    model_defs = models or [
        FauxModelDefinition(
            id=DEFAULT_MODEL_ID,
            name=DEFAULT_MODEL_NAME,
            reasoning=False,
            input=("text", "image"),
            cost={"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            context_window=128000,
            max_tokens=16384,
        )
    ]

    resolved_models = [
        {
            "id": m.id,
            "name": m.name or m.id,
            "api": resolved_api,
            "provider": resolved_provider,
            "base_url": DEFAULT_BASE_URL,
            "reasoning": m.reasoning,
            "input": list(m.input),
            "cost": m.cost
            or {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
            "context_window": m.context_window,
            "max_tokens": m.max_tokens,
        }
        for m in model_defs
    ]

    async def _resolve_response(
        step: FauxResponseStep,
        context: Context,
        stream_options: SimpleStreamOptions | None,
        request_model: Any,
    ) -> AssistantMessage:
        if callable(step):
            result = step(context, stream_options, state, request_model)
            if isinstance(result, AssistantMessage):
                resolved = result
            else:
                resolved = await result
        else:
            resolved = step
        return _with_usage_estimate(
            _clone_message(
                resolved,
                resolved_api,
                resolved_provider,
                getattr(request_model, "id", ""),
            ),
            context,
            stream_options,
            prompt_cache,
        )

    def _stream(
        request_model: Any,
        context: Context,
        stream_options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()
        step = pending_responses.pop(0) if pending_responses else None
        state.call_count += 1

        async def _run() -> None:
            try:
                if not step:
                    msg = _create_error_message(
                        RuntimeError("No more faux responses queued"),
                        resolved_api,
                        resolved_provider,
                        getattr(request_model, "id", ""),
                    )
                    msg = _with_usage_estimate(
                        msg,
                        context,
                        cast("SimpleStreamOptions | None", stream_options),
                        prompt_cache,
                    )
                    outer.push(AssistantErrorEvent(reason="error", error=msg))
                    outer.end(msg)
                    return

                if stream_options and getattr(stream_options, "deferred", None):
                    handle = DeferredHandle(
                        deferred_id=_random_id("deferred"),
                        polling_url=None,
                    )
                    deferred_responses[handle.deferred_id] = {
                        "handle": handle,
                        "step": step,
                        "context": context,
                        "options": stream_options,
                        "model": request_model,
                        "pending_fetches": max(0, deferred_pending_fetches),
                        "cancelled": False,
                    }
                    await _stream_with_deltas(
                        outer,
                        _create_deferred_message(request_model, handle),
                        min_size,
                        max_size,
                        tokens_per_second,
                        getattr(stream_options, "signal", None),
                    )
                    return

                message = await _resolve_response(
                    step,
                    context,
                    cast("SimpleStreamOptions | None", stream_options),
                    request_model,
                )
                await _stream_with_deltas(
                    outer,
                    message,
                    min_size,
                    max_size,
                    tokens_per_second,
                    getattr(stream_options, "signal", None) if stream_options else None,
                )
            except Exception as error:
                msg = _create_error_message(
                    error,
                    resolved_api,
                    resolved_provider,
                    getattr(request_model, "id", ""),
                )
                outer.push(AssistantErrorEvent(reason="error", error=msg))
                outer.end(msg)

        asyncio.ensure_future(_run())
        return outer

    def _stream_simple(
        request_model: Any,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return _stream(request_model, context, cast("StreamOptions | None", options))

    async def _fetch_deferred(
        request_model: Any,
        handle: DeferredHandle,
        fetch_options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        outer = AssistantMessageEventStream()
        state.deferred_fetch_count += 1

        async def _run() -> None:
            try:
                entry = deferred_responses.get(handle.deferred_id)
                if not entry:
                    msg = _create_error_message(
                        RuntimeError(
                            f"Unknown faux deferred response: {handle.deferred_id}"
                        ),
                        resolved_api,
                        resolved_provider,
                        getattr(request_model, "id", ""),
                    )
                    outer.push(AssistantErrorEvent(reason="error", error=msg))
                    outer.end(msg)
                    return
                if entry["cancelled"]:
                    msg = _create_error_message(
                        RuntimeError(
                            f"Faux deferred response was cancelled: {handle.deferred_id}"
                        ),
                        resolved_api,
                        resolved_provider,
                        getattr(request_model, "id", ""),
                    )
                    outer.push(AssistantErrorEvent(reason="error", error=msg))
                    outer.end(msg)
                    return

                if entry["pending_fetches"] > 0:
                    entry["pending_fetches"] -= 1
                    await _stream_with_deltas(
                        outer,
                        _create_deferred_message(request_model, entry["handle"]),
                        min_size,
                        max_size,
                        tokens_per_second,
                        getattr(fetch_options, "signal", None)
                        if fetch_options
                        else None,
                    )
                    return

                if "final" not in entry:
                    sub_options = dict(entry["options"] or {})
                    sub_options.pop("deferred", None)
                    sub_options.pop("signal", None)
                    sub_options.pop("on_response", None)
                    try:
                        entry["final"] = await _resolve_response(
                            entry["step"],
                            entry["context"],
                            type("SimpleStreamOptions", (), sub_options)()
                            if sub_options
                            else None,
                            entry["model"],
                        )
                    except Exception as error:
                        entry["final"] = _create_error_message(
                            error,
                            resolved_api,
                            resolved_provider,
                            getattr(entry["model"], "id", ""),
                        )
                await _stream_with_deltas(
                    outer,
                    entry["final"],
                    min_size,
                    max_size,
                    tokens_per_second,
                    getattr(fetch_options, "signal", None) if fetch_options else None,
                )
            except Exception as error:
                msg = _create_error_message(
                    error,
                    resolved_api,
                    resolved_provider,
                    getattr(request_model, "id", ""),
                )
                outer.push(AssistantErrorEvent(reason="error", error=msg))
                outer.end(msg)

        asyncio.ensure_future(_run())
        return outer

    async def _cancel_deferred(
        request_model: Any,
        handle: DeferredHandle,
        cancel_options: DeferredCancelOptions | None = None,
    ) -> None:
        state.cancelled_deferred.append(handle)
        entry = deferred_responses.get(handle.deferred_id)
        if entry:
            entry["cancelled"] = True

    def _get_model(model_id: str | None = None) -> Any:
        if not model_id:
            return resolved_models[0] if resolved_models else None
        for m in resolved_models:
            if m["id"] == model_id:
                return m
        return None

    def _set_responses(responses: list[FauxResponseStep]) -> None:
        pending_responses.clear()
        pending_responses.extend(responses)

    def _append_responses(responses: list[FauxResponseStep]) -> None:
        pending_responses.extend(responses)

    return {
        "api": resolved_api,
        "provider": resolved_provider,
        "models": resolved_models,
        "stream": _stream,
        "stream_simple": _stream_simple,
        "fetch_deferred": _fetch_deferred,
        "cancel_deferred": _cancel_deferred,
        "get_model": _get_model,
        "state": state,
        "set_responses": _set_responses,
        "append_responses": _append_responses,
        "get_pending_response_count": lambda: len(pending_responses),
    }


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


def faux_provider(
    *,
    api: str | None = None,
    provider: str | None = None,
    models: list[FauxModelDefinition] | None = None,
    deferred_pending_fetches: int = 0,
    deferred_poll_after_ms: int | None = None,
    tokens_per_second: float | None = None,
    token_size_min: int = DEFAULT_MIN_TOKEN_SIZE,
    token_size_max: int = DEFAULT_MAX_TOKEN_SIZE,
) -> Any:
    """创建 Faux Provider（测试用）。

    Args:
        api: API 标识（默认随机生成）。
        provider: Provider ID（默认随机生成）。
        models: 模型定义列表。
        deferred_pending_fetches: 延迟响应返回原始 handle 的 fetch 次数。
        deferred_poll_after_ms: 延迟响应轮询间隔（ms）。
        tokens_per_second: 模拟 token 输出速率。
        token_size_min: 模拟 token 最小字符数。
        token_size_max: 模拟 token 最大字符数。

    Returns:
        包含 ``provider``/``stream``/``stream_simple``/``fetch_deferred``/
        ``cancel_deferred``/``get_model``/``state``/``set_responses``/
        ``append_responses``/``get_pending_response_count`` 的对象。
    """
    core = _create_faux_core(
        api=api,
        provider=provider,
        models=models,
        deferred_pending_fetches=deferred_pending_fetches,
        deferred_poll_after_ms=deferred_poll_after_ms,
        tokens_per_second=tokens_per_second,
        token_size_min=token_size_min,
        token_size_max=token_size_max,
    )

    provider_obj = create_provider(
        CreateProviderOptions(
            id=core["provider"],
            models=core["models"],
            api={
                "stream": core["stream"],
                "stream_simple": core["stream_simple"],
                "fetch_deferred": core["fetch_deferred"],
                "cancel_deferred": core["cancel_deferred"],
            },
        )
    )

    # 返回兼容 TS ``FauxProviderHandle`` 的结构
    result = {
        "provider": provider_obj,
        "api": core["api"],
        "models": core["models"],
        "get_model": core["get_model"],
        "state": core["state"],
        "set_responses": core["set_responses"],
        "append_responses": core["append_responses"],
        "get_pending_response_count": core["get_pending_response_count"],
    }
    return type("FauxProviderHandle", (), result)()
