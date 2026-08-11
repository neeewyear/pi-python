"""Mistral Conversations API 消息格式转换与流式传输（对应 ``mistral-conversations.ts``）。"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, TypeAlias, cast

import httpx

from ..models import calculate_cost, clamp_thinking_level
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
    ImageContent,
    Message,
    StreamOptions,
    TextContent,
    Tool,
    ToolCallContent,
    Usage,
)
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.hash import short_hash
from ..utils.json_parse import parse_streaming_json
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import resolve_json_schema_strict_sampling
from .simple_options import build_base_options
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MISTRAL_TOOL_CALL_ID_LENGTH = 9
MAX_MISTRAL_ERROR_BODY_CHARS = 4000

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

MistralReasoningEffort: TypeAlias = Literal["none", "high"]


class MistralOptions(StreamOptions):
    """Mistral API 特定选项（对应 TS ``MistralOptions``）。"""

    tool_choice: Literal["auto", "none", "any", "required"] | dict[str, Any] | None = (
        None
    )
    prompt_mode: Literal["reasoning"] | None = None
    reasoning_effort: MistralReasoningEffort | None = None
    signal: Any | None = None  # CancellationToken
    on_payload: (
        Callable[[dict[str, Any], Any], Awaitable[dict[str, Any] | None]] | None
    ) = None
    on_response: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def has_header(headers: dict[str, str | None] | None, name: str) -> bool:
    """检查请求头是否存在且非空。"""
    if not headers:
        return False
    expected = name.lower()
    for key, value in headers.items():
        if key.lower() == expected and value is not None and value.strip():
            return True
    return False


def get_client_api_key(
    provider: str,
    api_key: str | None,
    headers: dict[str, str | None] | None,
) -> str:
    """获取客户端 API key。"""
    if api_key:
        return api_key
    if has_header(headers, "authorization") or has_header(
        headers, "cf-aig-authorization"
    ):
        return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def format_mistral_error(error: object) -> str:
    """格式化 Mistral 错误（对应 TS ``formatMistralError``）。"""
    if isinstance(error, BaseException):
        status = _extract_error_status(error)
        body = _extract_error_body(error)
        if status is not None and body is not None:
            return (
                f"Mistral API error ({status}): "
                f"{_truncate_error_text(body, MAX_MISTRAL_ERROR_BODY_CHARS)}"
            )
        if status is not None:
            return f"Mistral API error ({status}): {error}"
        return str(error)
    return _safe_json_stringify(error)


def _extract_error_status(error: BaseException) -> int | None:
    """从异常中提取 HTTP 状态码。"""
    for attr in ("status_code", "statusCode", "status"):
        val = getattr(error, attr, None)
        if isinstance(val, int):
            return val
    return None


def _extract_error_body(error: BaseException) -> str | None:
    """从异常中提取响应体。"""
    body = getattr(error, "body", None)
    if isinstance(body, str) and body.strip():
        return body.strip()
    return None


def _truncate_error_text(text: str, max_chars: int) -> str:
    """截断错误文本。"""
    if len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}... [truncated {len(text) - max_chars} chars]"


def _safe_json_stringify(value: object) -> str:
    """安全地 JSON 序列化。"""
    try:
        serialized = json.dumps(value)
        return serialized if serialized != "undefined" else str(value)
    except (TypeError, ValueError, OverflowError):
        return str(value)


def create_mistral_tool_call_id_normalizer() -> Callable[[str], str]:
    """创建 Mistral tool call ID 归一化器（对应 TS ``createMistralToolCallIdNormalizer``）。

    Mistral 要求 tool call ID 为 9 字符。
    """
    id_map: dict[str, str] = {}
    reverse_map: dict[str, str] = {}

    def normalize(id: str) -> str:
        existing = id_map.get(id)
        if existing is not None:
            return existing

        attempt = 0
        while True:
            candidate = derive_mistral_tool_call_id(id, attempt)
            owner = reverse_map.get(candidate)
            if owner is None or owner == id:
                id_map[id] = candidate
                reverse_map[candidate] = id
                return candidate
            attempt += 1

    return normalize


def derive_mistral_tool_call_id(id: str, attempt: int) -> str:
    """派生 Mistral tool call ID（对应 TS ``deriveMistralToolCallId``）。"""
    import re

    normalized = re.sub(r"[^a-zA-Z0-9]", "", id)
    if attempt == 0 and len(normalized) == MISTRAL_TOOL_CALL_ID_LENGTH:
        return normalized
    seed_base = normalized or id
    seed = seed_base if attempt == 0 else f"{seed_base}:{attempt}"
    return re.sub(
        r"[^a-zA-Z0-9]",
        "",
        short_hash(seed),
    )[:MISTRAL_TOOL_CALL_ID_LENGTH]


def should_use_prompt_caching(
    options: MistralOptions | None,
) -> bool:
    """检查是否应该使用 prompt 缓存（对应 TS ``shouldUsePromptCaching``）。"""
    if not options:
        return False
    return options.cache_retention != "none" and bool(options.session_id)


def get_mistral_cached_prompt_tokens(usage: dict[str, Any], prompt_tokens: int) -> int:
    """获取 Mistral 缓存的 prompt token 数（对应 TS ``getMistralCachedPromptTokens``）。"""
    raw_cached_tokens: Any = (
        _deep_get(usage, "promptTokensDetails", "cachedTokens")
        or _deep_get(usage, "prompt_tokens_details", "cached_tokens")
        or _deep_get(usage, "promptTokenDetails", "cachedTokens")
        or _deep_get(usage, "prompt_token_details", "cached_tokens")
        or usage.get("numCachedTokens")
        or usage.get("num_cached_tokens")
        or 0
    )
    cached_tokens = (
        raw_cached_tokens
        if isinstance(raw_cached_tokens, (int, float))
        and raw_cached_tokens == raw_cached_tokens  # not NaN
        else 0
    )
    return min(prompt_tokens, max(0, int(cached_tokens)))


def _deep_get(d: dict[str, Any], *keys: str) -> Any:
    """深度获取嵌套字典值。"""
    current: Any = d
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
        if current is None:
            return None
    return current


def uses_reasoning_effort(model: Any) -> bool:
    """检查模型是否使用 reasoning_effort（对应 TS ``usesReasoningEffort``）。"""
    model_id = getattr(model, "model_id", "")
    return model_id in (
        "mistral-small-2603",
        "mistral-small-latest",
        "mistral-medium-3.5",
    )


def uses_prompt_mode_reasoning(model: Any) -> bool:
    """检查模型是否使用 prompt_mode reasoning（对应 TS ``usesPromptModeReasoning``）。"""
    return bool(getattr(model, "reasoning", False)) and not uses_reasoning_effort(model)


def map_reasoning_effort(model: Any, level: str) -> MistralReasoningEffort:
    """映射 reasoning effort（对应 TS ``mapReasoningEffort``）。"""
    thinking_level_map = getattr(model, "thinking_level_map", None) or {}
    mapped = thinking_level_map.get(level)
    if isinstance(mapped, str) and mapped in ("none", "high"):
        return cast(MistralReasoningEffort, mapped)
    return "high"


def map_tool_choice(choice: Any) -> Any:
    """映射 tool_choice（对应 TS ``mapToolChoice``）。"""
    if not choice:
        return None
    if choice in ("auto", "none", "any", "required"):
        return choice
    if isinstance(choice, dict) and choice.get("type") == "function":
        return {
            "type": "function",
            "function": {"name": choice["function"]["name"]},
        }
    return choice


def map_chat_stop_reason(reason: str | None) -> dict[str, Any]:
    """映射 Mistral finish_reason 到内部 stop reason（对应 TS ``mapChatStopReason``）。"""
    if reason is None:
        return {"stop_reason": "stop"}
    if reason == "stop":
        return {"stop_reason": "stop"}
    if reason in ("length", "model_length"):
        return {"stop_reason": "length"}
    if reason == "tool_calls":
        return {"stop_reason": "tool_use"}
    if reason == "error":
        return {
            "stop_reason": "error",
            "error_message": "Provider stopped with: error",
        }
    return {
        "stop_reason": "error",
        "error_message": f"Provider stopped with: {reason}",
    }


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: MistralOptions | None = None,
) -> AssistantMessageEventStream:
    """Mistral Chat Completions API 流式生成函数（对应 TS ``stream``）。"""
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
                cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
            ),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )
        # 额外字段（Pydantic 不验证的额外属性）
        object.__setattr__(output, "response_id", "")
        object.__setattr__(output, "raw_stop_reason", None)

        try:
            provider = getattr(model, "provider", "")
            api_key = get_client_api_key(
                provider,
                options.api_key if options else None,
                options.headers if options else None,
            )

            # 创建 tool call ID 归一化器
            normalize_mistral_tool_call_id = create_mistral_tool_call_id_normalizer()

            def normalize_tool_call_id_cb(
                tc_id: str, _target_model: Any, _source: Any
            ) -> str:
                return normalize_mistral_tool_call_id(tc_id)

            transformed_messages = transform_messages(
                context.messages, model, normalize_tool_call_id_cb
            )

            params = build_params(model, context, transformed_messages, options)

            # onPayload 回调允许修改参数
            if options and options.on_payload:
                next_params = await options.on_payload(dict(params), model)
                if next_params is not None:
                    params = next_params

            # 构建请求头和 URL
            base_url = getattr(model, "base_url", "") or ""
            url = f"{base_url.rstrip('/')}/chat/completions"
            request_headers = build_request_headers(model, options)
            request_headers["Authorization"] = f"Bearer {api_key}"
            request_headers["Content-Type"] = "application/json"

            timeout_ms = options.timeout_ms if options else None
            timeout = httpx.Timeout(timeout_ms / 1000.0 if timeout_ms else 120.0)

            event_stream.push(AssistantMessageSnapshot(message=output))

            # 发起 HTTP 流式请求
            async with httpx.AsyncClient(timeout=timeout) as client:
                async with client.stream(
                    "POST",
                    url,
                    json=params,
                    headers=request_headers,
                ) as response:
                    # onResponse 回调
                    if options and options.on_response:
                        await options.on_response(
                            {
                                "status": response.status_code,
                                "headers": dict(response.headers),
                            },
                            model,
                        )

                    response.raise_for_status()
                    await consume_chat_stream(model, output, event_stream, response)

            # 检查 abort
            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Mistral stream ended without a finish reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # 清理临时字段
            for content_block in output.content:
                for field in ("partial_args",):
                    if hasattr(content_block, field):
                        try:
                            delattr(content_block, field)
                        except (AttributeError, TypeError):
                            pass

            output.stop_reason = (
                "aborted"
                if (
                    options
                    and options.signal
                    and getattr(options.signal, "aborted", False)
                )
                else "error"
            )
            output.error_message = format_mistral_error(error)

            event_stream.push(
                AssistantErrorEvent(reason=output.stop_reason, error=output)
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
    """简化的流式接口（对应 TS ``streamSimple``）。"""
    provider = getattr(model, "provider", "")
    api_key = get_client_api_key(
        provider,
        options.api_key if options else None,
        options.headers if options else None,
    )

    base = build_base_options(model, context, options, api_key)
    clamped_reasoning = (
        clamp_thinking_level(model, options.reasoning)
        if options and options.reasoning
        else None
    )
    reasoning: str | None = None if clamped_reasoning == "off" else clamped_reasoning
    should_use_reasoning = (
        bool(getattr(model, "reasoning", False)) and reasoning is not None
    )

    mistral_opts = MistralOptions(
        **base.model_dump(),
        api_key=api_key,
        prompt_mode=(
            "reasoning"
            if should_use_reasoning and uses_prompt_mode_reasoning(model)
            else None
        ),
        reasoning_effort=(
            map_reasoning_effort(model, cast(str, reasoning))
            if should_use_reasoning and uses_reasoning_effort(model)
            else None
        ),
    )

    return stream(model, context, mistral_opts)


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------


def build_params(
    model: Any,
    context: Context,
    messages: list[Message],
    options: MistralOptions | None = None,
) -> dict[str, Any]:
    """构建请求参数（对应 TS ``buildChatPayload``）。"""
    params: dict[str, Any] = {
        "model": getattr(model, "model_id", ""),
        "stream": True,
        "messages": convert_messages(model, messages),
    }

    if context.tools:
        params["tools"] = convert_tools(context.tools)
    if options is not None:
        if options.temperature is not None:
            params["temperature"] = options.temperature
        if options.max_tokens is not None:
            params["max_tokens"] = options.max_tokens
        if options.tool_choice is not None:
            params["tool_choice"] = map_tool_choice(options.tool_choice)
        if options.prompt_mode is not None:
            params["prompt_mode"] = options.prompt_mode
        if options.reasoning_effort is not None:
            params["reasoning_effort"] = options.reasoning_effort
        if should_use_prompt_caching(options):
            assert options.session_id is not None
            params["prompt_cache_key"] = options.session_id
        if options.sampling_params:
            params.update(options.sampling_params)

    return params


def build_request_headers(
    model: Any,
    options: MistralOptions | None = None,
) -> dict[str, str]:
    """构建请求头（对应 TS ``buildRequestOptions`` 中的头部逻辑）。"""
    headers: dict[str, str] = {}

    # 复制模型级请求头
    model_headers = getattr(model, "headers", None) or {}
    if isinstance(model_headers, dict):
        for key, value in model_headers.items():
            if value is not None:
                headers[key] = value

    # 选项级请求头
    if options and options.headers:
        for key, value in options.headers.items():
            if value is not None:
                headers[key] = value

    # Mistral 使用 x-affinity 进行 KV-cache 重用（前缀缓存）
    if (
        options is not None
        and should_use_prompt_caching(options)
        and "x-affinity" not in {k.lower(): v for k, v in headers.items()}
    ):
        assert options.session_id is not None
        headers["x-affinity"] = options.session_id

    return headers


def convert_messages(
    model: Any,
    messages: list[Message],
) -> list[dict[str, Any]]:
    """将内部消息格式转换为 Mistral Chat Completions API 格式（对应 TS ``toChatMessages``）。"""
    result: list[dict[str, Any]] = []
    supports_images = "image" in (
        getattr(model, "input_types", None) or getattr(model, "input", []) or []
    )

    for msg in messages:
        if msg.role == "user":
            user_msg = msg
            if isinstance(user_msg.content, list):
                # 检查是否有图片
                had_images = any(isinstance(b, ImageContent) for b in user_msg.content)
                content: list[dict[str, Any]] = []
                for item in user_msg.content:
                    if isinstance(item, TextContent):
                        content.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    elif isinstance(item, ImageContent) and supports_images:
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.mime_type};base64,{item.data}"
                                },
                            }
                        )
                if content:
                    result.append({"role": "user", "content": content})
                elif had_images and not supports_images:
                    result.append(
                        {
                            "role": "user",
                            "content": "(image omitted: model does not support images)",
                        }
                    )
            else:
                result.append(
                    {
                        "role": "user",
                        "content": sanitize_surrogates(str(user_msg.content)),
                    }
                )

        elif msg.role == "assistant":
            assistant_msg = msg
            content_parts: list[dict[str, Any]] = []
            tool_calls: list[dict[str, Any]] = []

            for block in cast("list[Any]", assistant_msg.content):
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    text = getattr(block, "text", "")
                    if text.strip():
                        content_parts.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(text),
                            }
                        )
                elif block_type == "thinking":
                    thinking_text = getattr(block, "text", "")
                    if thinking_text.strip():
                        content_parts.append(
                            {
                                "type": "thinking",
                                "thinking": [
                                    {
                                        "type": "text",
                                        "text": sanitize_surrogates(thinking_text),
                                    }
                                ],
                            }
                        )
                elif block_type == "toolCall":
                    tc = cast(ToolCallContent, block)
                    tool_calls.append(
                        {
                            "id": tc.tool_call_id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.args, separators=(",", ":")),
                            },
                        }
                    )

            assistant_entry: dict[str, Any] = {"role": "assistant"}
            if content_parts:
                assistant_entry["content"] = content_parts
            if tool_calls:
                assistant_entry["tool_calls"] = tool_calls
            if content_parts or tool_calls:
                result.append(assistant_entry)

        elif msg.role == "toolResult":
            tool_result_msg = msg
            text_result = "\n".join(
                getattr(b, "text", "")
                for b in tool_result_msg.content
                if isinstance(b, TextContent)
            )
            has_images = any(
                isinstance(b, ImageContent) for b in tool_result_msg.content
            )
            tool_text = build_tool_result_text(
                text_result, has_images, supports_images, tool_result_msg.is_error
            )

            tool_content: list[dict[str, Any]] = [{"type": "text", "text": tool_text}]
            for part in tool_result_msg.content:
                if not supports_images:
                    break
                if isinstance(part, ImageContent):
                    tool_content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{part.mime_type};base64,{part.data}"
                            },
                        }
                    )

            result.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_result_msg.tool_call_id,
                    "name": tool_result_msg.tool_name or "",
                    "content": tool_content,
                }
            )

    return result


def build_tool_result_text(
    text: str, has_images: bool, supports_images: bool, is_error: bool
) -> str:
    """构建工具结果文本（对应 TS ``buildToolResultText``）。"""
    trimmed = text.strip()
    error_prefix = "[tool error] " if is_error else ""

    if trimmed:
        image_suffix = (
            "\n[tool image omitted: model does not support images]"
            if has_images and not supports_images
            else ""
        )
        return f"{error_prefix}{trimmed}{image_suffix}"

    if has_images:
        if supports_images:
            return (
                "[tool error] (see attached image)"
                if is_error
                else "(see attached image)"
            )
        return (
            "[tool error] (image omitted: model does not support images)"
            if is_error
            else "(image omitted: model does not support images)"
        )

    return "[tool error] (no tool output)" if is_error else "(no tool output)"


def convert_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """将内部工具格式转换为 Mistral 工具格式（对应 TS ``toFunctionTools``）。"""
    result: list[dict[str, Any]] = []

    for tool in tools:
        strict = resolve_json_schema_strict_sampling(tool, True)
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": strict if strict is not None else False,
                },
            }
        )

    return result


async def consume_chat_stream(
    model: Any,
    output: AssistantMessage,
    event_stream: AssistantMessageEventStream,
    response: httpx.Response,
) -> None:
    """消费 Mistral 流式响应（对应 TS ``consumeChatStream``）。"""
    current_text_block: TextContent | None = None
    current_thinking_block: Any | None = None
    blocks = cast("list[Any]", output.content)
    tool_blocks_by_key: dict[str, int] = {}

    def finish_current_text_block(block: TextContent | None) -> None:
        if block is not None:
            event_stream.push(AssistantTextDelta(delta=block.text))

    def finish_current_thinking_block(block: Any | None) -> None:
        if block is not None:
            event_stream.push(AssistantThinkingDelta(delta=block.thinking))

    async for line in response.aiter_lines():
        line = line.strip()
        if not line.startswith("data: "):
            continue

        data = line[6:].strip()
        if data == "[DONE]":
            break

        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue

        # Mistral 的 streamed chunk 携带 id 字段
        chunk_id = chunk.get("id")
        if chunk_id:
            if not getattr(output, "response_id", None):
                object.__setattr__(output, "response_id", chunk_id)

        # 处理 usage
        usage_data = chunk.get("usage")
        if usage_data and isinstance(usage_data, dict):
            prompt_tokens = usage_data.get("promptTokens", 0) or 0
            cached_prompt_tokens = get_mistral_cached_prompt_tokens(
                usage_data, prompt_tokens
            )
            assert output.usage is not None

            output.usage.input = max(0, prompt_tokens - cached_prompt_tokens)
            output.usage.output = usage_data.get("completionTokens", 0) or 0
            output.usage.cache_read = cached_prompt_tokens
            output.usage.cache_write = 0
            output.usage.total_tokens = usage_data.get("totalTokens", 0) or (
                output.usage.input
                + output.usage.output
                + output.usage.cache_read
                + output.usage.cache_write
            )
            calculate_cost(model, output.usage)

        choices = chunk.get("choices", [])
        if not choices:
            continue
        choice = choices[0]
        if not isinstance(choice, dict):
            continue

        # 处理 finish_reason
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None:
            object.__setattr__(output, "raw_stop_reason", finish_reason)
            stop_reason_result = map_chat_stop_reason(finish_reason)
            output.stop_reason = stop_reason_result["stop_reason"]
            if stop_reason_result.get("error_message"):
                output.error_message = stop_reason_result["error_message"]

        delta = choice.get("delta", {})
        if not isinstance(delta, dict):
            continue

        # 处理文本增量
        delta_content = delta.get("content")
        if delta_content is not None:
            content_items: list[Any] = (
                [delta_content] if isinstance(delta_content, str) else delta_content
            )
            for item in content_items:
                if isinstance(item, str):
                    text_delta = sanitize_surrogates(item)
                    if current_thinking_block is not None:
                        finish_current_thinking_block(current_thinking_block)
                        current_thinking_block = None
                    if current_text_block is None:
                        current_text_block = TextContent(text="")
                        blocks.append(current_text_block)
                    current_text_block.text += text_delta
                    event_stream.push(AssistantTextDelta(delta=text_delta))
                    continue

                if isinstance(item, dict):
                    item_type = item.get("type")
                    if item_type == "thinking":
                        thinking_parts = item.get("thinking", [])
                        delta_text = "".join(
                            p.get("text", "")
                            for p in thinking_parts
                            if isinstance(p, dict) and "text" in p
                        )
                        thinking_delta = sanitize_surrogates(delta_text)
                        if not thinking_delta:
                            continue
                        if current_text_block is not None:
                            finish_current_text_block(current_text_block)
                            current_text_block = None
                        if current_thinking_block is None:
                            current_thinking_block = {
                                "type": "thinking",
                                "thinking": "",
                            }
                            blocks.append(current_thinking_block)
                        current_thinking_block["thinking"] += thinking_delta
                        event_stream.push(AssistantThinkingDelta(delta=thinking_delta))
                        continue

                    if item_type == "text":
                        text_delta = sanitize_surrogates(item.get("text", ""))
                        if current_thinking_block is not None:
                            finish_current_thinking_block(current_thinking_block)
                            current_thinking_block = None
                        if current_text_block is None:
                            current_text_block = TextContent(text="")
                            blocks.append(current_text_block)
                        current_text_block.text += text_delta
                        event_stream.push(AssistantTextDelta(delta=text_delta))

        # 处理 tool_calls
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            if current_text_block is not None:
                finish_current_text_block(current_text_block)
                current_text_block = None
            if current_thinking_block is not None:
                finish_current_thinking_block(current_thinking_block)
                current_thinking_block = None

            for tool_call in tool_calls:
                if not isinstance(tool_call, dict):
                    continue

                tc_id = tool_call.get("id", "")
                tc_index = tool_call.get("index", 0)
                tc_function = tool_call.get("function", {})

                call_id = (
                    tc_id
                    if tc_id and tc_id != "null"
                    else derive_mistral_tool_call_id(f"toolcall:{tc_index}", 0)
                )
                key = f"{call_id}:{tc_index}"
                existing_index = tool_blocks_by_key.get(key)
                block: Any = None

                if existing_index is not None:
                    existing = blocks[existing_index]
                    if (
                        isinstance(existing, dict)
                        or getattr(existing, "type", None) == "toolCall"
                    ):
                        block = existing

                if block is None:
                    func_name = ""
                    if isinstance(tc_function, dict):
                        func_name = tc_function.get("name", "") or ""
                    block = ToolCallContent(
                        tool_call_id=call_id,
                        name=func_name,
                        args={},
                    )
                    object.__setattr__(block, "partial_args", "")
                    blocks.append(block)
                    tool_blocks_by_key[key] = len(blocks) - 1

                # 处理 arguments delta
                args_delta = ""
                if isinstance(tc_function, dict):
                    raw_args = tc_function.get("arguments")
                    if isinstance(raw_args, str):
                        args_delta = raw_args
                    elif raw_args is not None:
                        args_delta = json.dumps(raw_args)

                if args_delta:
                    partial_args = getattr(block, "partial_args", "")
                    partial_args += args_delta
                    block.partial_args = partial_args
                    block.args = parse_streaming_json(partial_args)

                    event_stream.push(
                        AssistantToolCallUpdate(
                            tool_call_id=block.tool_call_id,
                            args=block.args,
                        )
                    )

    # 流结束：清理临时字段并推送结束事件
    finish_current_text_block(current_text_block)
    finish_current_thinking_block(current_thinking_block)

    for index in tool_blocks_by_key.values():
        if index >= len(blocks):
            continue
        block = blocks[index]
        if getattr(block, "type", None) != "toolCall":
            continue
        partial_args = getattr(block, "partial_args", None)
        if partial_args is not None:
            block.args = parse_streaming_json(partial_args)
        # 清理临时字段
        if hasattr(block, "partial_args"):
            try:
                delattr(block, "partial_args")
            except (AttributeError, TypeError):
                pass
        event_stream.push(
            AssistantToolCallEnd(
                tool_call_id=block.tool_call_id,
                content=[block],
            )
        )
