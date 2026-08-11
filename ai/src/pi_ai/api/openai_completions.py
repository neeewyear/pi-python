"""OpenAI Chat Completions API 消息格式转换与流式传输（对应 ``openai-completions.ts``）。"""

from __future__ import annotations

import json
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from openai import AsyncOpenAI

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
    ThinkingBlock,
    Tool,
    ToolCallContent,
    ToolResultMessage,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.hash import short_hash
from ..utils.headers import headers_to_record
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import (
    GrammarToolInputJsonBuffer,
    append_grammar_tool_input_json_delta,
    create_grammar_tool_input_properties,
    get_grammar_tool_input,
    resolve_grammar_constrained_sampling,
    resolve_json_schema_strict_sampling,
)
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .simple_options import build_base_options
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})

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


def has_tool_history(messages: list[Message]) -> bool:
    """检查消息中是否有工具调用历史。"""
    for msg in messages:
        if msg.role == "toolResult":
            return True
        if msg.role == "assistant":
            for block in msg.content:
                if block.type == "toolCall":
                    return True
    return False


def get_deferred_tool_names(messages: list[Message]) -> set[str]:
    """获取延迟工具名称集合。"""
    names: set[str] = set()
    for message in messages:
        if message.role == "toolResult":
            for name in message.added_tool_names or []:
                names.add(name)
    return names


def get_tools_by_name(tools: list[Tool] | None, names: set[str]) -> list[Tool]:
    """按名称获取工具列表。"""
    if not tools:
        return []
    tools_by_name = {tool.name: tool for tool in tools}
    result: list[Tool] = []
    for name in names:
        tool = tools_by_name.get(name)
        if tool is not None:
            result.append(tool)
    return result


def is_text_content_block(block: Any) -> bool:
    """检查是否为文本内容块。"""
    return getattr(block, "type", None) == "text"


def is_thinking_content_block(block: Any) -> bool:
    """检查是否为思考内容块。"""
    return getattr(block, "type", None) == "thinking"


def is_tool_call_block(block: Any) -> bool:
    """检查是否为工具调用块。"""
    return getattr(block, "type", None) == "toolCall"


def is_image_content_block(block: Any) -> bool:
    """检查是否为图片内容块。"""
    return getattr(block, "type", None) == "image"


def is_encrypted_reasoning_detail(detail: Any) -> bool:
    """检查是否为加密推理详情。"""
    if not isinstance(detail, dict):
        return False
    return (
        detail.get("type") == "reasoning.encrypted"
        and isinstance(detail.get("id"), str)
        and len(detail["id"]) > 0
        and isinstance(detail.get("data"), str)
        and len(detail["data"]) > 0
    )


def format_openai_error(error: object) -> str:
    """格式化 OpenAI 错误。"""
    return format_provider_error(normalize_provider_error(error), "OpenAI API error")


# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------


class OpenAICompletionsOptions(StreamOptions):
    """OpenAI Chat Completions API 特定选项（对应 TS ``OpenAICompletionsOptions``）。"""

    reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    service_tier: str | None = None
    tool_choice: Any | None = None
    signal: Any | None = None  # CancellationToken
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


class ConvertCompletionsMessagesOptions:
    """消息转换选项（对应 TS ``ConvertCompletionsMessagesOptions``）。"""

    def __init__(
        self,
        include_system_prompt: bool = True,
        grammar_tool_input_properties: dict[str, str] | None = None,
        tool_options: dict[str, Any] | None = None,
    ) -> None:
        self.include_system_prompt = include_system_prompt
        self.grammar_tool_input_properties = grammar_tool_input_properties
        self.tool_options = tool_options


# 内部类型
class OpenAICompatCacheControl:
    """OpenAI 兼容缓存控制。"""

    def __init__(self, type: str = "ephemeral", ttl: str | None = None) -> None:
        self.type = type
        self.ttl = ttl


# 流式工具调用增量
class StreamingToolCallDelta:
    """流式工具调用增量。"""

    def __init__(
        self,
        index: int | None = None,
        id: str | None = None,
        type: str | None = None,
        function: dict[str, str] | None = None,
        custom: dict[str, str] | None = None,
    ) -> None:
        self.index = index
        self.id = id
        self.type = type
        self.function = function
        self.custom = custom


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
) -> AssistantMessageEventStream:
    """OpenAI Chat Completions API 流式生成函数（对应 TS ``stream``）。"""
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
        object.__setattr__(output, "response_model", "")
        object.__setattr__(output, "raw_stop_reason", None)
        # error_message 是 AssistantMessage 的合法字段，无需额外设置

        try:
            provider = getattr(model, "provider", "")
            api_key = get_client_api_key(
                provider,
                options.api_key if options else None,
                options.headers if options else None,
            )
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools,
                compat.get("supports_openai_grammar_tools", False),
            )
            cache_retention = resolve_cache_retention(
                options.cache_retention if options else None,
                options.env if options else None,
            )
            cache_session_id = (
                options.session_id if options and cache_retention != "none" else None
            )
            client = create_client(
                model,
                context,
                api_key,
                options.headers if options else None,
                options.fetch if options else None,
                cache_session_id,
                compat,
            )
            params = build_params(
                model,
                context,
                options,
                compat,
                cache_retention,
                grammar_tool_input_properties,
            )

            # onPayload 回调允许修改参数
            if options and options.on_payload:
                next_params = await options.on_payload(dict(params), model)
                if next_params is not None:
                    params = next_params

            # 请求选项
            request_options: dict[str, Any] = {
                "max_retries": 0,
            }
            if options and options.signal is not None:
                request_options["signal"] = options.signal
            if options and options.timeout_ms is not None:
                request_options["timeout"] = options.timeout_ms

            # 发起请求（带重试）
            raw_response = await retry_provider_request(
                lambda: client.chat.completions.with_raw_response.create(
                    **params, **request_options
                ),
                ProviderRetryOptions(
                    max_retries=cast(int, options.max_retries if options else 0),
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=options.signal if options else None,
                ),
            )

            # 解析出流和响应头
            raw_response_obj = cast("Any", raw_response)
            openai_stream = raw_response_obj.parse()
            response_headers = dict(raw_response_obj.headers)

            # onResponse 回调
            if options and options.on_response:
                await options.on_response(
                    {
                        "status": getattr(raw_response, "status_code", 0),
                        "headers": headers_to_record(response_headers),
                    },
                    model,
                )

            event_stream.push(AssistantMessageSnapshot(message=output))

            # 流式处理状态
            text_block: TextContent | None = None
            thinking_block: ThinkingBlock | None = None
            has_finish_reason = False
            tool_call_blocks_by_index: dict[int, Any] = {}
            tool_call_blocks_by_id: dict[str, Any] = {}
            pending_reasoning_details_by_tool_call_id: dict[str, str] = {}
            blocks = cast("list[Any]", output.content)

            def get_content_index(block: Any) -> int:
                try:
                    return blocks.index(block)
                except ValueError:
                    return -1

            def get_custom_tool_call_input(block: Any) -> str:
                custom_input = getattr(block, "custom_input", None)
                if custom_input is None:
                    return ""
                property_name = custom_input.get("property", "")
                if not property_name:
                    return ""
                value = block.args.get(property_name)
                return str(value) if isinstance(value, str) else ""

            def append_custom_tool_call_input(
                block: Any, next_input: str, close: bool
            ) -> str | None:
                custom_input = getattr(block, "custom_input", None)
                if not custom_input:
                    return None
                delta = append_grammar_tool_input_json_delta(
                    custom_input["jsonBuffer"],
                    custom_input["property"],
                    next_input,
                    close,
                )
                block.args = {custom_input["property"]: next_input}
                return delta

            def finish_block(block: Any) -> None:
                content_index = get_content_index(block)
                if content_index == -1:
                    return
                block_type = getattr(block, "type", None)
                if block_type == "text":
                    event_stream.push(AssistantTextDelta(delta=block.text))
                elif block_type == "thinking":
                    event_stream.push(AssistantThinkingDelta(delta=block.thinking))
                elif block_type == "toolCall":
                    custom_input = getattr(block, "custom_input", None)
                    if custom_input:
                        delta = append_custom_tool_call_input(
                            block, get_custom_tool_call_input(block), True
                        )
                        if delta is not None:
                            event_stream.push(
                                AssistantToolCallUpdate(
                                    tool_call_id=block.tool_call_id,
                                    args={
                                        custom_input["property"]: block.args.get(
                                            custom_input["property"], ""
                                        )
                                    },
                                )
                            )
                    else:
                        partial_args = getattr(block, "partial_args", None)
                        if partial_args is not None:
                            block.args = parse_streaming_json(partial_args)
                    # 清理临时字段
                    for field in ("partial_args", "custom_input", "stream_index"):
                        if hasattr(block, field):
                            try:
                                delattr(block, field)
                            except (AttributeError, TypeError):
                                pass
                    event_stream.push(
                        AssistantToolCallEnd(
                            tool_call_id=block.tool_call_id,
                            content=[block],
                        )
                    )

            def ensure_text_block() -> TextContent:
                nonlocal text_block
                if text_block is None:
                    text_block = TextContent(text="")
                    blocks.append(text_block)
                return text_block

            def ensure_thinking_block(thinking_signature: str) -> ThinkingBlock:
                nonlocal thinking_block
                if thinking_block is None:
                    thinking_block = ThinkingBlock(
                        text="",
                        signature=thinking_signature,
                    )
                    blocks.append(thinking_block)
                return thinking_block

            def apply_pending_reasoning_detail(block: Any) -> None:
                block_id = getattr(block, "tool_call_id", None)
                if not block_id:
                    return
                pending = pending_reasoning_details_by_tool_call_id.get(block_id)
                if pending:
                    block.thought_signature = pending
                    pending_reasoning_details_by_tool_call_id.pop(block_id)

            def ensure_tool_call_block(tool_call: StreamingToolCallDelta) -> Any:
                stream_index = tool_call.index
                name = ""
                if tool_call.function:
                    name = tool_call.function.get("name", "")
                elif tool_call.custom:
                    name = tool_call.custom.get("name", "")

                block: Any = None
                if stream_index is not None:
                    block = tool_call_blocks_by_index.get(stream_index)
                if block is None and tool_call.id:
                    block = tool_call_blocks_by_id.get(tool_call.id)

                if block is None:
                    custom_input_property: str | None = None
                    if tool_call.custom and not tool_call.function:
                        custom_input_property = grammar_tool_input_properties.get(
                            name, "input"
                        )
                    has_custom_input = custom_input_property is not None

                    block = ToolCallContent(
                        tool_call_id=tool_call.id or "",
                        name=name,
                        args={custom_input_property: ""} if has_custom_input else {},
                    )
                    if not has_custom_input:
                        block.partial_args = ""
                    if has_custom_input:
                        block.custom_input = {
                            "property": custom_input_property,
                            "jsonBuffer": GrammarToolInputJsonBuffer(),
                        }
                    if stream_index is not None:
                        block.stream_index = stream_index

                    if stream_index is not None:
                        tool_call_blocks_by_index[stream_index] = block
                    if tool_call.id:
                        tool_call_blocks_by_id[tool_call.id] = block
                    blocks.append(block)

                if (
                    stream_index is not None
                    and getattr(block, "stream_index", None) is None
                ):
                    block.stream_index = stream_index
                    tool_call_blocks_by_index[stream_index] = block
                if tool_call.id:
                    tool_call_blocks_by_id[tool_call.id] = block
                if not getattr(block, "name", None) and name:
                    block.name = name
                if (
                    tool_call.custom
                    and not tool_call.function
                    and not hasattr(block, "custom_input")
                ):
                    custom_input_property = grammar_tool_input_properties.get(
                        block.name, "input"
                    )
                    block.args = {custom_input_property: ""}
                    block.custom_input = {
                        "property": custom_input_property,
                        "jsonBuffer": GrammarToolInputJsonBuffer(),
                    }
                    if hasattr(block, "partial_args"):
                        try:
                            delattr(block, "partial_args")
                        except (AttributeError, TypeError):
                            pass
                apply_pending_reasoning_detail(block)
                return block

            # 主流式循环
            async for chunk in openai_stream:
                if not chunk or not isinstance(chunk, object):
                    continue

                # OpenAI 的 ChatCompletionChunk.id 是唯一的聊天完成标识符
                chunk_id = getattr(chunk, "id", None)
                if chunk_id:
                    output.response_id = output.response_id or chunk_id

                chunk_model = getattr(chunk, "model", None)
                if (
                    isinstance(chunk_model, str)
                    and len(chunk_model) > 0
                    and chunk_model != getattr(model, "model_id", "")
                ):
                    if not output.response_model:  # type: ignore[attr-defined]
                        output.response_model = chunk_model  # type: ignore[attr-defined]

                chunk_usage = getattr(chunk, "usage", None)
                if chunk_usage:
                    output.usage = parse_chunk_usage(chunk_usage, model)

                choices = getattr(chunk, "choices", None)
                choice = (
                    choices[0]
                    if isinstance(choices, list) and len(choices) > 0
                    else None
                )
                if not choice:
                    continue

                # 回退：有些 provider（如 Moonshot）在 choice.usage 中返回用量
                if not chunk_usage:
                    choice_usage = getattr(choice, "usage", None)
                    if choice_usage:
                        output.usage = parse_chunk_usage(choice_usage, model)

                finish_reason = getattr(choice, "finish_reason", None)
                if finish_reason:
                    object.__setattr__(output, "raw_stop_reason", finish_reason)
                    finish_reason_result = map_stop_reason(finish_reason)
                    output.stop_reason = finish_reason_result["stop_reason"]
                    if finish_reason_result.get("error_message"):
                        output.error_message = finish_reason_result["error_message"]
                    has_finish_reason = True

                delta = getattr(choice, "delta", None)
                if delta:
                    delta_content = getattr(delta, "content", None)
                    if delta_content is not None and len(str(delta_content)) > 0:
                        block = ensure_text_block()
                        block.text += str(delta_content)
                        event_stream.push(AssistantTextDelta(delta=str(delta_content)))

                    # 推理内容处理
                    reasoning_fields = [
                        "reasoning_content",
                        "reasoning",
                        "reasoning_text",
                    ]
                    delta_dict = (
                        delta
                        if isinstance(delta, dict)
                        else delta.model_dump()
                        if hasattr(delta, "model_dump")
                        else {}
                    )
                    found_reasoning_field: str | None = None
                    for field in reasoning_fields:
                        value = (
                            delta_dict.get(field)
                            if isinstance(delta_dict, dict)
                            else getattr(delta, field, None)
                        )
                        if isinstance(value, str) and len(value) > 0:
                            found_reasoning_field = field
                            break

                    if found_reasoning_field:
                        raw_value = (
                            delta_dict.get(found_reasoning_field)
                            if isinstance(delta_dict, dict)
                            else getattr(delta, found_reasoning_field, None)
                        )
                        if isinstance(raw_value, str) and len(raw_value) > 0:
                            thinking_signature = (
                                "reasoning_content"
                                if getattr(model, "provider", "") == "opencode-go"
                                and found_reasoning_field == "reasoning"
                                else found_reasoning_field
                            )
                            t_block = ensure_thinking_block(thinking_signature)
                            t_block.text += raw_value
                            event_stream.push(AssistantThinkingDelta(delta=raw_value))

                    # 工具调用处理
                    delta_tool_calls = getattr(delta, "tool_calls", None)
                    if delta_tool_calls:
                        for tc_delta in delta_tool_calls:
                            tc_index = getattr(tc_delta, "index", None)
                            tc_id = getattr(tc_delta, "id", None)
                            tc_function = getattr(tc_delta, "function", None)
                            tc_type = getattr(tc_delta, "type", None)

                            func_name = (
                                getattr(tc_function, "name", None)
                                if tc_function is not None
                                else None
                            )
                            func_args = (
                                getattr(tc_function, "arguments", None)
                                if tc_function is not None
                                else None
                            )

                            function_dict: dict[str, str] | None = None
                            if func_name is not None:
                                function_dict = {"name": func_name}
                                if func_args is not None:
                                    function_dict["arguments"] = func_args
                            elif func_args is not None:
                                function_dict = {"arguments": func_args}

                            tool_call = StreamingToolCallDelta(
                                index=tc_index,
                                id=tc_id,
                                type=tc_type,
                                function=function_dict,
                            )
                            block = ensure_tool_call_block(tool_call)
                            if not getattr(block, "tool_call_id", None) and tc_id:
                                block.tool_call_id = tc_id
                                tool_call_blocks_by_id[tc_id] = block
                            if not getattr(block, "name", None) and func_name:
                                block.name = func_name

                            delta_str = ""
                            if func_args:
                                delta_str = func_args
                                partial_args = getattr(block, "partial_args", None)
                                if partial_args is None:
                                    partial_args = ""
                                partial_args += func_args
                                block.partial_args = partial_args
                                block.args = parse_streaming_json(partial_args)

                            if delta_str:
                                event_stream.push(
                                    AssistantToolCallUpdate(
                                        tool_call_id=block.tool_call_id,
                                        args=block.args,
                                    )
                                )

                    # reasoning_details 处理
                    reasoning_details = (
                        delta_dict.get("reasoning_details")
                        if isinstance(delta_dict, dict)
                        else getattr(delta, "reasoning_details", None)
                    )
                    if isinstance(reasoning_details, list):
                        for detail in reasoning_details:
                            if is_encrypted_reasoning_detail(detail):
                                serialized_detail = json.dumps(
                                    detail, separators=(",", ":")
                                )
                                matching_tool_call = tool_call_blocks_by_id.get(
                                    detail["id"]
                                )
                                if matching_tool_call is not None:
                                    matching_tool_call.thought_signature = (
                                        serialized_detail
                                    )
                                else:
                                    pending_reasoning_details_by_tool_call_id[
                                        detail["id"]
                                    ] = serialized_detail

            # 流结束处理
            for block in blocks:
                finish_block(block)

            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "aborted":
                raise RuntimeError("Request was aborted")

            if not has_finish_reason and not compat.get("supports_finish_reason", True):
                output.stop_reason = (
                    "tool_use"
                    if any(
                        getattr(b, "type", None) == "toolCall" for b in output.content
                    )
                    else "stop"
                )

            if output.stop_reason == "error":
                raise RuntimeError(
                    output.error_message or "Provider returned an error stop reason"
                )

            if (
                compat.get("supports_finish_reason", True) and not has_finish_reason
            ) or output.stop_reason == "pending":
                raise RuntimeError("Stream ended without finish_reason")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # 清理临时字段
            for content_block in output.content:
                for field in ("index", "partial_args", "custom_input", "stream_index"):
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
            output.error_message = format_openai_error(error)

            # 一些通过 OpenRouter 的 provider 在这个字段中提供额外信息
            raw_metadata = None
            try:
                raw_metadata = getattr(error, "error", None)
                if raw_metadata is not None:
                    raw_metadata = getattr(raw_metadata, "metadata", None)
                    if raw_metadata is not None:
                        raw_metadata = getattr(raw_metadata, "raw", None)
            except Exception:
                raw_metadata = None

            if (
                raw_metadata is not None
                and output.error_message is not None
                and str(raw_metadata) not in output.error_message
            ):
                output.error_message = f"{output.error_message}\n{raw_metadata}"

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
    """简化的流式接口（对应 TS ``streamSimple``）。"""
    get_client_api_key(
        getattr(model, "provider", ""),
        options.api_key if options else None,
        options.headers if options else None,
    )

    base = build_base_options(
        model, context, options, options.api_key if options else None
    )
    clamped_reasoning = (
        clamp_thinking_level(model, options.reasoning)
        if options and options.reasoning
        else None
    )
    reasoning_effort: str | None = (
        None if clamped_reasoning == "off" else clamped_reasoning
    )
    tool_choice = getattr(options, "tool_choice", None) if options else None

    return stream(
        model,
        context,
        OpenAICompletionsOptions(
            **base.model_dump() if hasattr(base, "model_dump") else {},
            reasoning_effort=reasoning_effort,
            tool_choice=tool_choice,
        ),
    )


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------


def create_client(
    model: Any,
    context: Context,
    api_key: str,
    options_headers: dict[str, str | None] | None = None,
    fetch: Any = None,
    session_id: str | None = None,
    compat: dict[str, Any] | None = None,
) -> Any:
    """创建 OpenAI 客户端（对应 TS ``createClient``）。"""
    if compat is None:
        compat = get_compat(model)
    headers: dict[str, str | None] = {}

    # 复制模型级请求头
    model_headers = getattr(model, "headers", None) or {}
    if isinstance(model_headers, dict):
        headers.update(model_headers)

    if getattr(model, "provider", "") == "github-copilot":
        has_images = has_copilot_vision_input(context.messages)
        copilot_headers = build_copilot_dynamic_headers(
            {
                "messages": context.messages,
                "has_images": has_images,
            }
        )
        headers.update(copilot_headers)

    if session_id:
        if compat.get("session_affinity_format") == "openrouter":
            headers["x-session-id"] = session_id
        else:
            if compat.get("session_affinity_format") == "openai":
                headers["session_id"] = session_id
            headers["x-client-request-id"] = session_id
            headers["x-session-affinity"] = session_id

    # 选项级请求头最后合并，可覆盖默认值
    if options_headers:
        headers.update(options_headers)

    # 过滤掉 None 值
    filtered_headers: dict[str, str] = {
        k: v for k, v in headers.items() if v is not None
    }

    return AsyncOpenAI(
        api_key=api_key,
        base_url=getattr(model, "base_url", ""),
        dangerously_allow_browser=True,  # type: ignore[call-arg]
        default_headers=filtered_headers,
        http_client=fetch,
    )


def build_params(
    model: Any,
    context: Context,
    options: OpenAICompletionsOptions | None = None,
    compat: dict[str, Any] | None = None,
    cache_retention: str | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    """构建请求参数（对应 TS ``buildParams``）。"""
    if compat is None:
        compat = get_compat(model)
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools,
            compat.get("supports_openai_grammar_tools", False),
        )
    if cache_retention is None:
        cache_retention = resolve_cache_retention(
            options.cache_retention if options else None,
            options.env if options else None,
        )

    messages = convert_messages(
        model,
        context,
        compat,
        ConvertCompletionsMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
        ),
    )
    cache_control = get_compat_cache_control(compat, cache_retention)

    params: dict[str, Any] = {
        "model": getattr(model, "model_id", ""),
        "messages": messages,
        "stream": True,
    }

    # Prompt cache key
    base_url = getattr(model, "base_url", "") or ""
    if (base_url and "api.openai.com" in base_url and cache_retention != "none") or (
        cache_retention == "long" and compat.get("supports_long_cache_retention", True)
    ):
        prompt_cache_key = clamp_openai_prompt_cache_key(
            options.session_id if options else None
        )
        if prompt_cache_key is not None:
            params["prompt_cache_key"] = prompt_cache_key

    if cache_retention == "long" and compat.get("supports_long_cache_retention", True):
        params["prompt_cache_retention"] = "24h"

    # stream_options 包含用量信息
    if compat.get("supports_usage_in_streaming", True) is not False:
        params["stream_options"] = {"include_usage": True}

    if compat.get("supports_store", True):
        params["store"] = False

    # max_tokens / max_completion_tokens
    if options and options.max_tokens:
        if compat.get("max_tokens_field") == "max_tokens":
            params["max_tokens"] = options.max_tokens
        else:
            params["max_completion_tokens"] = options.max_tokens

    # temperature
    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    # tools
    deferred_tool_names = (
        get_deferred_tool_names(context.messages)
        if compat.get("deferred_tools_mode") == "kimi"
        else set()
    )
    active_tools = (
        [t for t in (context.tools or []) if t.name not in deferred_tool_names]
        if context.tools
        else None
    )
    if active_tools and len(active_tools) > 0:
        params["tools"] = convert_tools(active_tools, compat)
        if compat.get("zai_tool_stream", False):
            params["tool_stream"] = True
    elif has_tool_history(context.messages):
        # Anthropic（通过 LiteLLM/proxy）要求消息中有 tool_calls/tool_results 时提供 tools 参数
        params["tools"] = []

    # 缓存控制（Anthropic 风格）
    if cache_control is not None:
        apply_anthropic_cache_control(messages, params.get("tools"), cache_control)

    # tool_choice
    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    # thinking/reasoning 处理
    model_reasoning = getattr(model, "reasoning", False)
    thinking_format = compat.get("thinking_format", "openai")
    supports_reasoning_effort = compat.get("supports_reasoning_effort", True)

    if thinking_format == "zai" and model_reasoning:
        if options and options.reasoning_effort:
            params["thinking"] = {"type": "enabled", "clear_thinking": False}
        elif model_reasoning:
            params["thinking"] = {"type": "disabled"}
        if options and options.reasoning_effort and supports_reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            mapped_effort = thinking_level_map.get(
                options.reasoning_effort, options.reasoning_effort
            )
            if isinstance(mapped_effort, str):
                params["reasoning_effort"] = mapped_effort
    elif thinking_format == "qwen" and model_reasoning:
        params["enable_thinking"] = bool(options and options.reasoning_effort)
        if options and options.reasoning_effort and supports_reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            effort = thinking_level_map.get(
                options.reasoning_effort, options.reasoning_effort
            )
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif thinking_format == "qwen-chat-template" and model_reasoning:
        params["chat_template_kwargs"] = {
            "enable_thinking": bool(options and options.reasoning_effort),
            "preserve_thinking": True,
        }
    elif thinking_format == "chat-template" and model_reasoning:
        chat_template_kwargs = build_chat_template_values(
            model, options, compat.get("chat_template_kwargs", {})
        )
        if chat_template_kwargs:
            params["chat_template_kwargs"] = chat_template_kwargs
    elif thinking_format == "baseten" and model_reasoning:
        chat_template_args = build_chat_template_values(
            model, options, compat.get("chat_template_args", {})
        )
        if chat_template_args:
            params["chat_template_args"] = chat_template_args
        if supports_reasoning_effort:
            requested_effort = options.reasoning_effort if options else None
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            mapped_effort = (
                thinking_level_map.get(requested_effort)
                if requested_effort
                else thinking_level_map.get("off")
            )
            effort = mapped_effort if mapped_effort is not None else requested_effort
            if isinstance(effort, str):
                params["reasoning_effort"] = effort
    elif thinking_format == "deepseek" and model_reasoning:
        if options and options.reasoning_effort:
            params["thinking"] = {"type": "enabled"}
        elif (
            getattr(model, "thinking_level_map", None)
            and getattr(model, "thinking_level_map", {}).get("off") is not None
        ):
            params["thinking"] = {"type": "disabled"}
        if options and options.reasoning_effort and supports_reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            params["reasoning_effort"] = thinking_level_map.get(
                options.reasoning_effort, options.reasoning_effort
            )
    elif thinking_format == "openrouter" and model_reasoning:
        if options and options.reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            params["reasoning"] = {
                "effort": thinking_level_map.get(
                    options.reasoning_effort, options.reasoning_effort
                ),
            }
        elif (
            getattr(model, "thinking_level_map", None)
            and getattr(model, "thinking_level_map", {}).get("off") is not None
        ):
            params["reasoning"] = {
                "effort": getattr(model, "thinking_level_map", {}).get("off", "none")
            }
    elif (
        thinking_format == "ant-ling"
        and model_reasoning
        and options
        and options.reasoning_effort
    ):
        thinking_level_map = getattr(model, "thinking_level_map", None) or {}
        effort = thinking_level_map.get(options.reasoning_effort)
        if isinstance(effort, str):
            params["reasoning"] = {"effort": effort}
    elif thinking_format == "together" and model_reasoning:
        params["reasoning"] = {"enabled": bool(options and options.reasoning_effort)}
        if options and options.reasoning_effort and supports_reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            params["reasoning_effort"] = thinking_level_map.get(
                options.reasoning_effort, options.reasoning_effort
            )
    elif thinking_format == "string-thinking" and model_reasoning:
        if options and options.reasoning_effort:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            params["thinking"] = thinking_level_map.get(
                options.reasoning_effort, options.reasoning_effort
            )
        elif (
            getattr(model, "thinking_level_map", None)
            and getattr(model, "thinking_level_map", {}).get("off") is not None
        ):
            params["thinking"] = getattr(model, "thinking_level_map", {}).get(
                "off", "none"
            )
    elif (
        options
        and options.reasoning_effort
        and model_reasoning
        and supports_reasoning_effort
    ):
        # OpenAI 风格的 reasoning_effort
        thinking_level_map = getattr(model, "thinking_level_map", None) or {}
        params["reasoning_effort"] = thinking_level_map.get(
            options.reasoning_effort, options.reasoning_effort
        )
    elif (
        not (options and options.reasoning_effort)
        and model_reasoning
        and supports_reasoning_effort
    ):
        thinking_level_map = getattr(model, "thinking_level_map", None) or {}
        off_value = thinking_level_map.get("off")
        if isinstance(off_value, str):
            params["reasoning_effort"] = off_value

    # OpenRouter provider routing preferences
    model_compat = getattr(model, "compat", None) or {}
    if isinstance(model_compat, dict):
        open_router_routing = model_compat.get("openRouterRouting")
        if open_router_routing:
            params["provider"] = open_router_routing

    # Vercel AI Gateway provider routing preferences
    vercel_gateway_routing = (
        model_compat.get("vercelGatewayRouting")
        if isinstance(model_compat, dict)
        else None
    )
    if vercel_gateway_routing:
        only = vercel_gateway_routing.get("only")
        order = vercel_gateway_routing.get("order")
        if only or order:
            gateway_options: dict[str, list[str]] = {}
            if only:
                gateway_options["only"] = only
            if order:
                gateway_options["order"] = order
            params["providerOptions"] = {"gateway": gateway_options}

    # 自定义采样参数最后设置，覆盖命名请求字段
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


def build_chat_template_values(
    model: Any,
    options: OpenAICompletionsOptions | None,
    values: dict[str, Any],
) -> dict[str, Any] | None:
    """构建聊天模板值（对应 TS ``buildChatTemplateValues``）。"""
    resolved_values: dict[str, Any] = {}

    for key, value in values.items():
        resolved = resolve_chat_template_kwarg_value(model, options, value)
        if resolved is not None:
            resolved_values[key] = resolved

    return resolved_values if len(resolved_values) > 0 else None


def resolve_chat_template_kwarg_value(
    model: Any,
    options: OpenAICompletionsOptions | None,
    value: Any,
) -> Any:
    """解析聊天模板关键字参数值（对应 TS ``resolveChatTemplateKwargValue``）。"""
    if not isinstance(value, dict):
        return value

    reasoning_effort = options.reasoning_effort if options else None
    if not reasoning_effort and value.get("omitWhenOff"):
        return None
    if value.get("$var") == "thinking.enabled":
        return bool(reasoning_effort)

    thinking_level_map = getattr(model, "thinking_level_map", None) or {}
    mapped_value = (
        thinking_level_map.get(reasoning_effort)
        if reasoning_effort
        else thinking_level_map.get("off")
    )
    return mapped_value if mapped_value is not None else reasoning_effort


def get_compat_cache_control(
    compat: dict[str, Any],
    cache_retention: str,
) -> OpenAICompatCacheControl | None:
    """获取兼容缓存控制对象（对应 TS ``getCompatCacheControl``）。"""
    if compat.get("cache_control_format") != "anthropic" or cache_retention == "none":
        return None

    ttl = (
        "1h"
        if cache_retention == "long"
        and compat.get("supports_long_cache_retention", False)
        else None
    )
    return OpenAICompatCacheControl(type="ephemeral", ttl=ttl)


def apply_anthropic_cache_control(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    cache_control: OpenAICompatCacheControl,
) -> None:
    """应用 Anthropic 风格的缓存控制（对应 TS ``applyAnthropicCacheControl``）。"""
    _add_cache_control_to_system_prompt(messages, cache_control)
    _add_cache_control_to_last_tool(tools, cache_control)
    _add_cache_control_to_last_conversation_message(messages, cache_control)


def _add_cache_control_to_system_prompt(
    messages: list[dict[str, Any]],
    cache_control: OpenAICompatCacheControl,
) -> None:
    """为系统提示词添加缓存控制。"""
    for message in messages:
        if message.get("role") in ("system", "developer"):
            _add_cache_control_to_instruction_message(message, cache_control)
            return


def _add_cache_control_to_last_conversation_message(
    messages: list[dict[str, Any]],
    cache_control: OpenAICompatCacheControl,
) -> None:
    """为最后一条对话消息添加缓存控制。"""
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if message.get("role") in ("user", "assistant", "tool"):
            if _add_cache_control_to_message(message, cache_control):
                return


def _add_cache_control_to_last_tool(
    tools: list[dict[str, Any]] | None,
    cache_control: OpenAICompatCacheControl,
) -> None:
    """为最后一条工具定义添加缓存控制。"""
    if not tools or len(tools) == 0:
        return
    last_tool = tools[-1]
    cc_dict = {"type": cache_control.type}
    if cache_control.ttl:
        cc_dict["ttl"] = cache_control.ttl
    last_tool["cache_control"] = cc_dict


def _add_cache_control_to_instruction_message(
    message: dict[str, Any],
    cache_control: OpenAICompatCacheControl,
) -> bool:
    """为指令消息添加缓存控制。"""
    return _add_cache_control_to_text_content(message, cache_control)


def _add_cache_control_to_message(
    message: dict[str, Any],
    cache_control: OpenAICompatCacheControl,
) -> bool:
    """为消息添加缓存控制。"""
    if message.get("role") in ("user", "assistant", "tool"):
        return _add_cache_control_to_text_content(message, cache_control)
    return False


def _add_cache_control_to_text_content(
    message: dict[str, Any],
    cache_control: OpenAICompatCacheControl,
) -> bool:
    """为文本内容添加缓存控制。"""
    content = message.get("content")
    if isinstance(content, str):
        if len(content) == 0:
            return False
        cc_dict = {"type": cache_control.type}
        if cache_control.ttl:
            cc_dict["ttl"] = cache_control.ttl
        message["content"] = [
            {
                "type": "text",
                "text": content,
                "cache_control": cc_dict,
            }
        ]
        return True

    if not isinstance(content, list):
        return False

    for i in range(len(content) - 1, -1, -1):
        part = content[i]
        if isinstance(part, dict) and part.get("type") == "text":
            cc_dict = {"type": cache_control.type}
            if cache_control.ttl:
                cc_dict["ttl"] = cache_control.ttl
            part["cache_control"] = cc_dict
            return True

    return False


# ---------------------------------------------------------------------------
# convert_messages() - 消息格式转换
# ---------------------------------------------------------------------------


def convert_messages(
    model: Any,
    context: Context,
    compat: dict[str, Any],
    options: ConvertCompletionsMessagesOptions | None = None,
) -> list[dict[str, Any]]:
    """将内部消息格式转换为 OpenAI Chat Completions API 格式（对应 TS ``convertMessages``）。"""
    if options is None:
        options = ConvertCompletionsMessagesOptions()

    params: list[dict[str, Any]] = []

    def normalize_tool_call_id(id: str) -> str:
        # 处理来自 OpenAI Responses API 的管道分隔 ID
        # 格式: {call_id}|{id}
        if "|" in id:
            separator_index = id.index("|")
            call_id = id[:separator_index]
            # 只保留字母数字、_、-
            import re

            sanitized_call_id = re.sub(r"[^a-zA-Z0-9_-]", "_", call_id)
            item_id = id[separator_index + 1 :]
            sanitized_item_id = re.sub(r"[^a-zA-Z0-9_-]", "_", item_id)
            combined_id = (
                f"{sanitized_call_id}_{sanitized_item_id}"
                if len(sanitized_item_id) > 0
                else sanitized_call_id
            )
            if len(combined_id) <= 40:
                return combined_id
            hash_val = short_hash(id)[:8]
            prefix = sanitized_call_id[: max(1, 40 - len(hash_val) - 1)]
            return f"{prefix}_{hash_val}"

        if getattr(model, "provider", "") == "openai":
            return id[:40] if len(id) > 40 else id
        return id

    # 使用 transform_messages 转换消息
    def normalize_tool_call_id_cb(tc_id: str, _target_model: Any, _source: Any) -> str:
        return normalize_tool_call_id(tc_id)

    transformed_messages = transform_messages(
        context.messages, model, normalize_tool_call_id_cb
    )

    # 系统提示词
    if context.system_prompt:
        model_reasoning = getattr(model, "reasoning", False)
        supports_developer_role = compat.get("supports_developer_role", False)
        use_developer_role = model_reasoning and supports_developer_role
        role = "developer" if use_developer_role else "system"
        params.append(
            {"role": role, "content": sanitize_surrogates(context.system_prompt)}
        )

    last_role: str | None = None

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        # 有些 provider 不允许 tool result 后直接跟 user 消息
        # 插入一条合成 assistant 消息来桥接
        if (
            compat.get("requires_assistant_after_tool_result", False)
            and last_role == "toolResult"
            and msg.role == "user"
        ):
            params.append(
                {
                    "role": "assistant",
                    "content": "I have processed the tool results.",
                }
            )

        if msg.role == "user":
            user_msg = msg
            if isinstance(user_msg.content, list):
                content: list[dict[str, Any]] = []
                for item in user_msg.content:
                    if isinstance(item, TextContent):
                        content.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    elif isinstance(item, ImageContent):
                        content.append(
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{item.mime_type};base64,{item.data}",
                                },
                            }
                        )
                if len(content) == 0:
                    i += 1
                    last_role = msg.role
                    continue
                params.append(
                    {
                        "role": "user",
                        "content": content,
                    }
                )
            else:
                params.append(
                    {
                        "role": "user",
                        "content": sanitize_surrogates(str(user_msg.content)),
                    }
                )

        elif msg.role == "assistant":
            assistant_msg: AssistantMessage = msg
            requires_assistant_content = compat.get(
                "requires_assistant_after_tool_result", False
            )
            assistant_entry: dict[str, Any] = {
                "role": "assistant",
                "content": "" if requires_assistant_content else None,
            }

            assistant_text_parts = [
                block
                for block in assistant_msg.content
                if is_text_content_block(block) and getattr(block, "text", "").strip()
            ]
            assistant_text = "".join(
                getattr(b, "text", "") for b in assistant_text_parts
            )

            non_empty_thinking_blocks = [
                block
                for block in assistant_msg.content
                if is_thinking_content_block(block)
                and getattr(block, "text", "").strip()
            ]

            if non_empty_thinking_blocks:
                if compat.get("requires_thinking_as_text", False):
                    thinking_text = "\n\n".join(
                        sanitize_surrogates(getattr(b, "text", ""))
                        for b in non_empty_thinking_blocks
                    )
                    assistant_entry["content"] = [
                        {"type": "text", "text": thinking_text},
                        *[
                            {
                                "type": "text",
                                "text": sanitize_surrogates(getattr(b, "text", "")),
                            }
                            for b in assistant_text_parts
                        ],
                    ]
                else:
                    if assistant_text:
                        assistant_entry["content"] = assistant_text

                    signature = getattr(non_empty_thinking_blocks[0], "signature", None)
                    if (
                        getattr(model, "provider", "") == "opencode-go"
                        and signature == "reasoning"
                    ):
                        signature = "reasoning_content"
                    if signature:
                        thinking_text = "\n".join(
                            sanitize_surrogates(getattr(b, "text", ""))
                            for b in non_empty_thinking_blocks
                        )
                        assistant_entry[signature] = thinking_text
            elif assistant_text:
                assistant_entry["content"] = assistant_text

            tool_calls = [
                block for block in assistant_msg.content if is_tool_call_block(block)
            ]
            if tool_calls:
                assistant_entry["tool_calls"] = []
                for tc in tool_calls:
                    tc_block = cast(ToolCallContent, tc)
                    custom_input_property = (
                        options.grammar_tool_input_properties.get(tc_block.name)
                        if options.grammar_tool_input_properties
                        else None
                    )
                    if custom_input_property is not None:
                        assistant_entry["tool_calls"].append(
                            {
                                "id": tc_block.tool_call_id,
                                "type": "custom",
                                "custom": {
                                    "name": tc_block.name,
                                    "input": sanitize_surrogates(
                                        get_grammar_tool_input(
                                            tc_block.name,
                                            tc_block.args,
                                            custom_input_property,
                                        )
                                    ),
                                },
                            }
                        )
                    else:
                        assistant_entry["tool_calls"].append(
                            {
                                "id": tc_block.tool_call_id,
                                "type": "function",
                                "function": {
                                    "name": tc_block.name,
                                    "arguments": json.dumps(
                                        tc_block.args, separators=(",", ":")
                                    ),
                                },
                            }
                        )

                # 处理 reasoning_details
                reasoning_details = [
                    json.loads(getattr(tc, "thought_signature", "null"))
                    for tc in tool_calls
                    if getattr(tc, "thought_signature", None)
                ]
                reasoning_details = [rd for rd in reasoning_details if rd is not None]
                if reasoning_details:
                    assistant_entry["reasoning_details"] = reasoning_details

            if (
                compat.get("requires_reasoning_content_on_assistant_messages", False)
                and getattr(model, "reasoning", False)
                and assistant_entry.get("reasoning_content") is None
            ):
                assistant_entry["reasoning_content"] = ""

            # 跳过没有内容和工具调用的 assistant 消息
            entry_content = assistant_entry.get("content")
            has_content = entry_content is not None and (
                (isinstance(entry_content, str) and len(entry_content) > 0)
                or (isinstance(entry_content, list) and len(entry_content) > 0)
            )
            if not has_content and "tool_calls" not in assistant_entry:
                i += 1
                last_role = msg.role
                continue

            params.append(assistant_entry)

        elif msg.role == "toolResult":
            tool_result_msg: ToolResultMessage = msg
            image_blocks: list[dict[str, Any]] = []
            deferred_tool_names_local: set[str] = set()

            # 收集连续的 toolResult 消息
            j = i
            while (
                j < len(transformed_messages)
                and transformed_messages[j].role == "toolResult"
            ):
                tool_msg = cast(ToolResultMessage, transformed_messages[j])

                text_result = "\n".join(
                    getattr(b, "text", "")
                    for b in tool_msg.content
                    if is_text_content_block(b)
                )
                has_images = any(is_image_content_block(b) for b in tool_msg.content)

                has_text = len(text_result) > 0
                tool_result_text = (
                    text_result
                    if has_text
                    else ("(see attached image)" if has_images else "(no tool output)")
                )

                tool_result_entry: dict[str, Any] = {
                    "role": "tool",
                    "content": sanitize_surrogates(tool_result_text),
                    "tool_call_id": tool_msg.tool_call_id,
                }
                if (
                    compat.get("requires_tool_result_name", False)
                    and tool_msg.tool_name
                ):
                    tool_result_entry["name"] = tool_msg.tool_name
                params.append(tool_result_entry)

                if compat.get("deferred_tools_mode") == "kimi":
                    for name in tool_msg.added_tool_names or []:
                        deferred_tool_names_local.add(name)

                if has_images and "image" in (
                    getattr(model, "input_types", None) or getattr(model, "input", [])
                ):
                    for block in tool_msg.content:
                        if is_image_content_block(block):
                            image_block = cast(ImageContent, block)
                            image_blocks.append(
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:{image_block.mime_type};base64,{image_block.data}",
                                    },
                                }
                            )

                j += 1

            i = j - 1

            if image_blocks:
                if compat.get("requires_assistant_after_tool_result", False):
                    params.append(
                        {
                            "role": "assistant",
                            "content": "I have processed the tool results.",
                        }
                    )

                params.append(
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": "Attached image(s) from tool result:",
                            },
                            *image_blocks,
                        ],
                    }
                )
                last_role = "user"
            else:
                last_role = "toolResult"

            if deferred_tool_names_local:
                deferred_tools = get_tools_by_name(
                    context.tools, deferred_tool_names_local
                )
                if deferred_tools:
                    kimi_tool_message: dict[str, Any] = {
                        "role": "system",
                        "tools": convert_tools(deferred_tools, compat),
                    }
                    params.append(kimi_tool_message)

            i += 1
            continue

        last_role = msg.role
        i += 1

    return params


# ---------------------------------------------------------------------------
# convert_tools() - 工具格式转换
# ---------------------------------------------------------------------------


def convert_tools(
    tools: list[Tool],
    compat: dict[str, Any],
) -> list[dict[str, Any]]:
    """将内部工具格式转换为 OpenAI Chat Completions API 工具格式（对应 TS ``convertTools``）。"""
    result: list[dict[str, Any]] = []
    supports_openai_grammar_tools = compat.get("supports_openai_grammar_tools", False)
    supports_strict_mode = compat.get("supports_strict_mode", True) is not False

    for tool in tools:
        grammar = resolve_grammar_constrained_sampling(
            tool, supports_openai_grammar_tools
        )
        if grammar is not None:
            result.append(
                {
                    "type": "custom",
                    "custom": {
                        "name": tool.name,
                        "description": tool.description,
                        "format": {
                            "type": "grammar",
                            "grammar": {
                                "syntax": grammar.format,
                                "definition": grammar.definition,
                            },
                        },
                    },
                }
            )
            continue

        strict = resolve_json_schema_strict_sampling(tool, supports_strict_mode)
        tool_entry: dict[str, Any] = {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters,
            },
        }
        if supports_strict_mode:
            tool_entry["function"]["strict"] = strict if strict is not None else False
        result.append(tool_entry)

    return result


# ---------------------------------------------------------------------------
# parse_chunk_usage() - 解析用量
# ---------------------------------------------------------------------------


def parse_chunk_usage(raw_usage: Any, model: Any) -> Usage:
    """解析流式 chunk 中的用量信息（对应 TS ``parseChunkUsage``）。"""
    prompt_tokens = getattr(raw_usage, "prompt_tokens", None) or 0
    completion_tokens = getattr(raw_usage, "completion_tokens", None) or 0

    prompt_tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
    if prompt_tokens_details is not None:
        cache_read_tokens = getattr(prompt_tokens_details, "cached_tokens", None) or 0
        cache_write_tokens = (
            getattr(prompt_tokens_details, "cache_write_tokens", None) or 0
        )
    else:
        cache_read_tokens = getattr(raw_usage, "prompt_cache_hit_tokens", None) or 0
        cache_write_tokens = 0

    completion_tokens_details = getattr(raw_usage, "completion_tokens_details", None)
    reasoning_tokens = 0
    if completion_tokens_details is not None:
        reasoning_tokens = (
            getattr(completion_tokens_details, "reasoning_tokens", None) or 0
        )

    # OpenAI 在 prompt_tokens 中包含了缓存 token，需要减去
    input = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = completion_tokens

    usage = Usage(
        input=input,
        output=output_tokens,
        cache_read=cache_read_tokens,
        cache_write=cache_write_tokens,
        total_tokens=input + output_tokens + cache_read_tokens + cache_write_tokens,
        cost=Cost(input=0, output=0, cache_read=0, cache_write=0, total=0),
    )
    # 额外字段
    usage.reasoning = reasoning_tokens  # type: ignore[attr-defined]

    calculate_cost(model, usage)
    return usage


# ---------------------------------------------------------------------------
# map_stop_reason() - 映射停止原因
# ---------------------------------------------------------------------------


def map_stop_reason(reason: str | None) -> dict[str, Any]:
    """映射 OpenAI finish_reason 到内部 stop reason（对应 TS ``mapStopReason``）。"""
    if reason is None:
        return {"stop_reason": "stop"}

    if reason in ("stop", "end"):
        return {"stop_reason": "stop"}
    if reason == "length":
        return {"stop_reason": "length"}
    if reason in ("function_call", "tool_calls"):
        return {"stop_reason": "tool_use"}
    if reason == "content_filter":
        return {
            "stop_reason": "error",
            "error_message": "Provider finish_reason: content_filter",
        }
    if reason == "network_error":
        return {
            "stop_reason": "error",
            "error_message": "Provider finish_reason: network_error",
        }
    return {
        "stop_reason": "error",
        "error_message": f"Provider finish_reason: {reason}",
    }


# ---------------------------------------------------------------------------
# detect_compat() / get_compat() - 兼容性检测
# ---------------------------------------------------------------------------


def detect_compat(model: Any) -> dict[str, Any]:
    """自动检测兼容性设置（对应 TS ``detectCompat``）。"""
    provider = getattr(model, "provider", "")
    base_url = getattr(model, "base_url", "") or ""

    is_zai = (
        provider in ("zai", "zai-coding-cn")
        or "api.z.ai" in base_url
        or "open.bigmodel.cn" in base_url
    )
    is_together = (
        provider == "together"
        or "api.together.ai" in base_url
        or "api.together.xyz" in base_url
    )
    is_moonshot = (
        provider in ("moonshotai", "moonshotai-cn") or "api.moonshot." in base_url
    )
    is_openrouter = provider == "openrouter" or "openrouter.ai" in base_url
    is_cloudflare_workers_ai = (
        provider == "cloudflare-workers-ai" or "api.cloudflare.com" in base_url
    )
    is_cloudflare_ai_gateway = (
        provider == "cloudflare-ai-gateway" or "gateway.ai.cloudflare.com" in base_url
    )
    is_nvidia = provider == "nvidia" or "integrate.api.nvidia.com" in base_url
    is_ant_ling = provider == "ant-ling" or "api.ant-ling.com" in base_url

    is_non_standard = (
        is_nvidia
        or provider == "cerebras"
        or "cerebras.ai" in base_url
        or provider == "xai"
        or "api.x.ai" in base_url
        or is_together
        or "chutes.ai" in base_url
        or "deepseek.com" in base_url
        or is_zai
        or is_moonshot
        or provider == "opencode"
        or "opencode.ai" in base_url
        or is_cloudflare_workers_ai
        or is_cloudflare_ai_gateway
        or is_ant_ling
    )

    use_max_tokens = (
        "chutes.ai" in base_url
        or is_moonshot
        or is_cloudflare_ai_gateway
        or is_together
        or is_nvidia
        or is_ant_ling
        or is_zai
    )

    is_grok = provider == "xai" or "api.x.ai" in base_url
    is_deepseek = provider == "deepseek" or "deepseek.com" in base_url

    model_id = getattr(model, "model_id", "")
    is_openrouter_developer_role_model = is_openrouter and (
        model_id.startswith("anthropic/") or model_id.startswith("openai/")
    )

    cache_control_format: str | None = (
        "anthropic" if (is_openrouter and model_id.startswith("anthropic/")) else None
    )

    # 确定 thinking_format
    if is_deepseek:
        thinking_format = "deepseek"
    elif is_zai:
        thinking_format = "zai"
    elif is_together:
        thinking_format = "together"
    elif is_ant_ling:
        thinking_format = "ant-ling"
    elif is_openrouter:
        thinking_format = "openrouter"
    else:
        thinking_format = "openai"

    # session_affinity_format
    session_affinity_format = "openrouter" if is_openrouter else "openai"

    return {
        "supports_store": not is_non_standard,
        "supports_developer_role": is_openrouter_developer_role_model
        or (not is_non_standard and not is_openrouter),
        "supports_reasoning_effort": not is_grok
        and not is_zai
        and not is_moonshot
        and not is_together
        and not is_cloudflare_ai_gateway
        and not is_nvidia
        and not is_ant_ling,
        "supports_usage_in_streaming": True,
        "supports_finish_reason": True,
        "max_tokens_field": "max_tokens" if use_max_tokens else "max_completion_tokens",
        "requires_tool_result_name": False,
        "requires_assistant_after_tool_result": False,
        "requires_thinking_as_text": False,
        "requires_reasoning_content_on_assistant_messages": is_deepseek,
        "thinking_format": thinking_format,
        "open_router_routing": {},
        "vercel_gateway_routing": {},
        "chat_template_kwargs": {},
        "chat_template_args": {},
        "zai_tool_stream": False,
        "supports_strict_mode": not is_moonshot
        and not is_together
        and not is_cloudflare_ai_gateway
        and not is_nvidia,
        "supports_openai_grammar_tools": False,
        "cache_control_format": cache_control_format,
        "send_session_affinity_headers": False,
        "deferred_tools_mode": None,
        "session_affinity_format": session_affinity_format,
        "supports_long_cache_retention": not (
            is_together
            or is_cloudflare_workers_ai
            or is_cloudflare_ai_gateway
            or is_nvidia
            or is_ant_ling
        ),
    }


def get_compat(model: Any) -> dict[str, Any]:
    """获取解析后的兼容性设置（对应 TS ``getCompat``）。

    自动检测 provider/URL，然后用显式的 ``model.compat`` 覆盖。
    """
    detected = detect_compat(model)
    model_compat = getattr(model, "compat", None)
    if not model_compat:
        return detected

    if not isinstance(model_compat, dict):
        return detected

    return {
        "supports_store": model_compat.get(
            "supports_store",
            model_compat.get("supportsStore", detected["supports_store"]),
        ),
        "supports_developer_role": model_compat.get(
            "supports_developer_role",
            model_compat.get(
                "supportsDeveloperRole", detected["supports_developer_role"]
            ),
        ),
        "supports_reasoning_effort": model_compat.get(
            "supports_reasoning_effort",
            model_compat.get(
                "supportsReasoningEffort", detected["supports_reasoning_effort"]
            ),
        ),
        "supports_usage_in_streaming": model_compat.get(
            "supports_usage_in_streaming",
            model_compat.get(
                "supportsUsageInStreaming", detected["supports_usage_in_streaming"]
            ),
        ),
        "supports_finish_reason": model_compat.get(
            "supports_finish_reason",
            model_compat.get(
                "supportsFinishReason", detected["supports_finish_reason"]
            ),
        ),
        "max_tokens_field": model_compat.get(
            "max_tokens_field",
            model_compat.get("maxTokensField", detected["max_tokens_field"]),
        ),
        "requires_tool_result_name": model_compat.get(
            "requires_tool_result_name",
            model_compat.get(
                "requiresToolResultName", detected["requires_tool_result_name"]
            ),
        ),
        "requires_assistant_after_tool_result": model_compat.get(
            "requires_assistant_after_tool_result",
            model_compat.get(
                "requiresAssistantAfterToolResult",
                detected["requires_assistant_after_tool_result"],
            ),
        ),
        "requires_thinking_as_text": model_compat.get(
            "requires_thinking_as_text",
            model_compat.get(
                "requiresThinkingAsText", detected["requires_thinking_as_text"]
            ),
        ),
        "requires_reasoning_content_on_assistant_messages": model_compat.get(
            "requires_reasoning_content_on_assistant_messages",
            model_compat.get(
                "requiresReasoningContentOnAssistantMessages",
                detected["requires_reasoning_content_on_assistant_messages"],
            ),
        ),
        "thinking_format": model_compat.get(
            "thinking_format",
            model_compat.get("thinkingFormat", detected["thinking_format"]),
        ),
        "open_router_routing": model_compat.get(
            "open_router_routing", model_compat.get("openRouterRouting", {})
        ),
        "vercel_gateway_routing": model_compat.get(
            "vercel_gateway_routing",
            model_compat.get(
                "vercelGatewayRouting", detected["vercel_gateway_routing"]
            ),
        ),
        "chat_template_kwargs": model_compat.get(
            "chat_template_kwargs",
            model_compat.get("chatTemplateKwargs", detected["chat_template_kwargs"]),
        ),
        "chat_template_args": model_compat.get(
            "chat_template_args",
            model_compat.get("chatTemplateArgs", detected["chat_template_args"]),
        ),
        "zai_tool_stream": model_compat.get(
            "zai_tool_stream",
            model_compat.get("zaiToolStream", detected["zai_tool_stream"]),
        ),
        "supports_strict_mode": model_compat.get(
            "supports_strict_mode",
            model_compat.get("supportsStrictMode", detected["supports_strict_mode"]),
        ),
        "supports_openai_grammar_tools": model_compat.get(
            "supports_openai_grammar_tools",
            model_compat.get(
                "supportsOpenAIGrammarTools", detected["supports_openai_grammar_tools"]
            ),
        ),
        "cache_control_format": model_compat.get(
            "cache_control_format",
            model_compat.get("cacheControlFormat", detected["cache_control_format"]),
        ),
        "send_session_affinity_headers": model_compat.get(
            "send_session_affinity_headers",
            model_compat.get(
                "sendSessionAffinityHeaders", detected["send_session_affinity_headers"]
            ),
        ),
        "deferred_tools_mode": model_compat.get(
            "deferred_tools_mode",
            model_compat.get("deferredToolsMode", detected["deferred_tools_mode"]),
        ),
        "session_affinity_format": model_compat.get(
            "session_affinity_format",
            model_compat.get(
                "sessionAffinityFormat", detected["session_affinity_format"]
            ),
        ),
        "supports_long_cache_retention": model_compat.get(
            "supports_long_cache_retention",
            model_compat.get(
                "supportsLongCacheRetention", detected["supports_long_cache_retention"]
            ),
        ),
    }


def resolve_cache_retention(
    cache_retention: str | None,
    env: dict[str, str] | None,
) -> str:
    """解析缓存保留策略（对应 TS ``resolveCacheRetention``）。

    默认使用 "short"，通过 ``PI_CACHE_RETENTION`` 环境变量向后兼容。
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"
