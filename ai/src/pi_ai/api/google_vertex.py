"""Google Vertex AI API 主入口。"""

from __future__ import annotations

import asyncio
import json
import re
import time
from collections.abc import AsyncIterable, Awaitable, Callable
from typing import Any, Literal, cast

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
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    Context,
    Cost,
    StreamOptions,
    TextContent,
    ThinkingBlock,
    ToolCallContent,
    Usage,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from ..utils.sanitize_unicode import sanitize_surrogates
from .google_shared import (
    GoogleThinkingLevel,
    convert_messages,
    convert_tools,
    is_thinking_part,
    map_stop_reason,
    resolve_google_function_calling_mode,
    retain_thought_signature,
    retry_google_request,
    supports_google_strict_tool_sampling,
)
from .simple_options import build_base_options

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_GCP_VERTEX_CREDENTIALS_MARKER = "gcp-vertex-credentials"

# ---------------------------------------------------------------------------
# GoogleVertexOptions
# ---------------------------------------------------------------------------


class GoogleVertexOptions(StreamOptions):
    """Google Vertex AI 流式选项。"""

    tool_choice: Literal["auto", "none", "any"] | None = None
    thinking: dict[str, Any] | None = None
    signal: Any = None
    on_payload: (
        Callable[[dict[str, Any], Any], Awaitable[dict[str, Any] | None]] | None
    ) = None
    on_response: Callable[[dict[str, Any], Any], Awaitable[Any]] | None = None
    fetch: Any = None
    env: Any = None
    session_id: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    sampling_params: dict[str, Any] | None = None
    headers: dict[str, str | None] | None = None
    api_key: str | None = None
    project: str | None = None
    location: str | None = None


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
    if has_header(headers, "authorization") or has_header(headers, "x-goog-api-key"):
        return "unused"
    raise ValueError(f"No API key for provider: {provider}")


def format_vertex_error(error: object) -> str:
    """格式化 Vertex AI 错误。"""
    return format_provider_error(normalize_provider_error(error), "Vertex AI API error")


def _get_model_id(model: Any) -> str:
    """获取模型 ID。"""
    return getattr(model, "model_id", "") or getattr(model, "id", "")


def _get_provider(model: Any) -> str:
    """获取 provider。"""
    return getattr(model, "provider", "")


def _is_placeholder_api_key(api_key: str) -> bool:
    """检查是否为占位符 API key（如 ``<your-api-key>``）。"""
    return bool(re.match(r"^<[^>]+>$", api_key))


def resolve_api_key(options: GoogleVertexOptions | None = None) -> str | None:
    """解析 Vertex AI API key。

    如果未提供 API key 或为占位符/标记值，返回 ``None``（表示使用 ADC 认证）。
    """
    if options is None:
        return None
    api_key = options.api_key
    if not api_key:
        return None
    api_key = api_key.strip()
    if (
        not api_key
        or api_key == _GCP_VERTEX_CREDENTIALS_MARKER
        or _is_placeholder_api_key(api_key)
    ):
        return None
    return api_key


def resolve_project(options: GoogleVertexOptions | None = None) -> str:
    """解析 GCP 项目 ID。"""
    if options is None:
        project = get_provider_env_value(
            "GOOGLE_CLOUD_PROJECT"
        ) or get_provider_env_value("GCLOUD_PROJECT")
    else:
        project = (
            options.project
            or get_provider_env_value("GOOGLE_CLOUD_PROJECT", options.env)
            or get_provider_env_value("GCLOUD_PROJECT", options.env)
        )
    if not project:
        raise ValueError(
            "Vertex AI requires a project ID. "
            "Set GOOGLE_CLOUD_PROJECT/GCLOUD_PROJECT or pass project in options."
        )
    return project


def resolve_location(options: GoogleVertexOptions | None = None) -> str:
    """解析 GCP 区域。"""
    if options is None:
        location = get_provider_env_value("GOOGLE_CLOUD_LOCATION")
    else:
        location = options.location or get_provider_env_value(
            "GOOGLE_CLOUD_LOCATION", options.env
        )
    if not location:
        raise ValueError(
            "Vertex AI requires a location. "
            "Set GOOGLE_CLOUD_LOCATION or pass location in options."
        )
    return location


def resolve_vertex_config(
    model: Any,
    options: GoogleVertexOptions | None = None,
) -> dict[str, Any]:
    """解析 Vertex AI 配置（project_id, location, publisher, model）。"""
    model_id = _get_model_id(model)
    project = resolve_project(options)
    location = resolve_location(options)
    publisher = "google"
    return {
        "project_id": project,
        "location": location,
        "publisher": publisher,
        "model": model_id,
    }


def build_vertex_url(config: dict[str, Any]) -> str:
    """构建 Vertex AI endpoint URL。"""
    project = config["project_id"]
    location = config["location"]
    publisher = config["publisher"]
    model = config["model"]
    return (
        f"https://{location}-aiplatform.googleapis.com/v1/"
        f"projects/{project}/locations/{location}/"
        f"publishers/{publisher}/models/{model}:streamGenerateContent"
    )


def build_vertex_headers(
    config: dict[str, Any],
    api_key: str | None,
    options_headers: dict[str, str | None] | None = None,
) -> dict[str, str]:
    """构建 Vertex AI 请求头（不含 Bearer token，token 在请求时动态获取）。"""
    headers: dict[str, str] = {
        "Content-Type": "application/json",
    }
    if api_key:
        headers["x-goog-api-key"] = api_key
    # 合并选项级 headers
    if options_headers:
        for key, value in options_headers.items():
            if value is not None:
                headers[key] = value
    return headers


# ---------------------------------------------------------------------------
# convert_thinking_config
# ---------------------------------------------------------------------------


def convert_thinking_config(
    thinking: dict[str, Any] | None,
    model: Any,
) -> dict[str, Any] | None:
    """转换思考配置为 Google API 格式。"""
    if not thinking or not getattr(model, "reasoning", False):
        return None
    enabled = thinking.get("enabled", False)
    if enabled:
        thinking_config: dict[str, Any] = {"includeThoughts": True}
        level = thinking.get("level")
        if level is not None:
            thinking_config["thinkingLevel"] = level
        else:
            budget_tokens = thinking.get("budgetTokens")
            if budget_tokens is not None:
                thinking_config["thinkingBudget"] = budget_tokens
        return thinking_config
    # thinking 被禁用
    return _get_disabled_thinking_config(model)


# ---------------------------------------------------------------------------
# 模型检测辅助函数
# ---------------------------------------------------------------------------

_ClampedThinkingLevel = Literal["minimal", "low", "medium", "high"]


def _is_gemini_3_pro_model(model_id: str) -> bool:
    """检查是否为 Gemini 3 Pro 模型。"""
    return bool(re.search(r"gemini-3(?:\.\d+)?-pro", model_id.lower()))


def _is_gemini_3_flash_model(model_id: str) -> bool:
    """检查是否为 Gemini 3 Flash 模型。"""
    id_lower = model_id.lower()
    return (
        bool(re.search(r"gemini-3(?:\.\d+)?-flash", id_lower))
        or id_lower == "gemini-flash-latest"
        or id_lower == "gemini-flash-lite-latest"
    )


def _get_disabled_thinking_config(model: Any) -> dict[str, Any] | None:
    """获取禁用思考的配置。

    Google docs: Gemini 3.1 Pro 无法禁用思考，Gemini 3 Flash / Flash-Lite
    也不支持完全关闭思考。对于 Gemini 3 模型，使用最低的 thinkingLevel 且
    不包含 includeThoughts，使隐藏思考对 pi 不可见。
    """
    model_id = _get_model_id(model)
    if _is_gemini_3_pro_model(model_id):
        return {"thinkingLevel": "LOW"}
    if _is_gemini_3_flash_model(model_id):
        return {"thinkingLevel": "MINIMAL"}
    # Gemini 2.x 支持通过 thinkingBudget = 0 禁用
    return {"thinkingBudget": 0}


def _get_thinking_level(
    effort: _ClampedThinkingLevel,
    model: Any,
) -> GoogleThinkingLevel:
    """获取思考级别。"""
    model_id = _get_model_id(model)
    if _is_gemini_3_pro_model(model_id):
        if effort in ("minimal", "low"):
            return "LOW"
        return "HIGH"
    mapping: dict[_ClampedThinkingLevel, GoogleThinkingLevel] = {
        "minimal": "MINIMAL",
        "low": "LOW",
        "medium": "MEDIUM",
        "high": "HIGH",
    }
    return mapping.get(effort, "HIGH")


def _get_google_budget(
    model: Any,
    effort: _ClampedThinkingLevel,
    custom_budgets: Any | None = None,
) -> int:
    """获取 Google 思考预算。"""
    if custom_budgets is not None:
        budget = getattr(custom_budgets, effort, None)
        if budget is not None:
            return int(budget)

    model_id = _get_model_id(model)

    if "2.5-pro" in model_id:
        budgets: dict[_ClampedThinkingLevel, int] = {
            "minimal": 128,
            "low": 2048,
            "medium": 8192,
            "high": 32768,
        }
        return budgets[effort]

    if "2.5-flash-lite" in model_id:
        budgets = {
            "minimal": 512,
            "low": 2048,
            "medium": 8192,
            "high": 24576,
        }
        return budgets[effort]

    if "2.5-flash" in model_id:
        budgets = {
            "minimal": 128,
            "low": 2048,
            "medium": 8192,
            "high": 24576,
        }
        return budgets[effort]

    return -1


# ---------------------------------------------------------------------------
# build_params
# ---------------------------------------------------------------------------


def build_params(
    model: Any,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> dict[str, Any]:
    """构建请求参数。"""
    model_id = _get_model_id(model)
    contents = convert_messages(model, context)

    generation_config: dict[str, Any] = {}
    if options and options.temperature is not None:
        generation_config["temperature"] = options.temperature
    if options and options.max_tokens is not None:
        generation_config["maxOutputTokens"] = options.max_tokens

    params: dict[str, Any] = {
        "model": model_id,
        "contents": contents,
    }

    if generation_config:
        params["generationConfig"] = generation_config

    # System prompt
    if context.system_prompt:
        params["systemInstruction"] = {
            "parts": [{"text": sanitize_surrogates(context.system_prompt)}]
        }

    # Tools
    if context.tools and len(context.tools) > 0:
        params["tools"] = convert_tools(context.tools)

    # Tool config
    if context.tools and len(context.tools) > 0:
        function_calling_mode = resolve_google_function_calling_mode(
            context.tools,
            options.tool_choice if options else None,
            supports_google_strict_tool_sampling(model_id),
        )
        if function_calling_mode is not None:
            params["toolConfig"] = {
                "functionCallingConfig": {"mode": function_calling_mode}
            }

    # Thinking config
    thinking_config = convert_thinking_config(
        options.thinking if options else None, model
    )
    if thinking_config is not None:
        params["thinkingConfig"] = thinking_config

    return params


# ---------------------------------------------------------------------------
# Vertex AI 流式请求
# ---------------------------------------------------------------------------


async def _get_oauth_token() -> str:
    """获取 Google OAuth token（使用 ADC - Application Default Credentials）。"""
    try:
        import google.auth  # type: ignore[import-not-found]
        import google.auth.transport.requests  # type: ignore[import-not-found]

        credentials, _ = await asyncio.get_event_loop().run_in_executor(
            None,
            google.auth.default,
            ["https://www.googleapis.com/auth/cloud-platform"],
        )
        if not credentials.valid:
            request = google.auth.transport.requests.Request()
            await asyncio.get_event_loop().run_in_executor(
                None, credentials.refresh, request
            )
        token = credentials.token
        if not token:
            raise RuntimeError("Failed to obtain OAuth token")
        return cast(str, token)
    except ImportError:
        raise RuntimeError(
            "google-auth library is required for Vertex AI ADC authentication. "
            "Install with: pip install google-auth google-auth-httplib2"
        )


async def _make_vertex_request(
    config: dict[str, Any],
    api_key: str | None,
    contents: list[dict[str, Any]],
    params: dict[str, Any],
    options_headers: dict[str, str | None] | None = None,
    timeout: float | None = None,
) -> AsyncIterable[dict[str, Any]]:
    """发起 Vertex AI Gemini API 流式请求并返回解析后的 JSON 块。"""
    url = build_vertex_url(config)
    request_params = {"alt": "sse"}

    # 构建请求体
    payload: dict[str, Any] = {
        "contents": contents,
    }
    # 添加其他参数（不含 model，model 已在 URL 中）
    if "generationConfig" in params:
        payload["generationConfig"] = params["generationConfig"]
    if "systemInstruction" in params:
        payload["systemInstruction"] = params["systemInstruction"]
    if "tools" in params:
        payload["tools"] = params["tools"]
    if "toolConfig" in params:
        payload["toolConfig"] = params["toolConfig"]
    if "thinkingConfig" in params:
        payload["thinkingConfig"] = params["thinkingConfig"]

    # 构建请求头
    headers = build_vertex_headers(config, api_key, options_headers)

    # 如果没有 API key，使用 ADC OAuth token
    if not api_key:
        token = await _get_oauth_token()
        headers["Authorization"] = f"Bearer {token}"

    client = httpx.AsyncClient(timeout=timeout)
    response = await client.post(
        url, headers=headers, json=payload, params=request_params
    )
    response.raise_for_status()

    async def _iterate() -> AsyncIterable[dict[str, Any]]:
        try:
            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue
                if line.startswith("data: "):
                    data_str = line[6:].strip()
                    if data_str and data_str != "[DONE]":
                        yield json.loads(data_str)
        finally:
            await client.aclose()

    return _iterate()


# 工具调用 ID 计数器
_tool_call_counter = 0


def _next_tool_call_id() -> int:
    global _tool_call_counter
    _tool_call_counter += 1
    return _tool_call_counter


# ---------------------------------------------------------------------------
# stream
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: GoogleVertexOptions | None = None,
) -> AssistantMessageEventStream:
    """Google Vertex AI 流式生成函数。"""
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

            # 自定义 fetch 不支持
            if options and options.fetch:
                raise RuntimeError(
                    "Custom fetch is not supported by the Google Vertex AI adapter"
                )

            # 解析 API key（None 表示使用 ADC 认证）
            api_key = resolve_api_key(options)

            # 解析 Vertex AI 配置
            config = resolve_vertex_config(model, options)

            # 构建请求参数
            params = build_params(model, context, options)

            # onPayload 回调
            if options and options.on_payload:
                next_params = await options.on_payload(dict(params), model)
                if next_params is not None:
                    params = next_params

            # 从 params 中提取 contents
            contents = params.pop("contents", [])
            model_id = params.pop("model", config["model"])

            # 超时设置
            timeout = None
            if options and options.timeout_ms is not None:
                timeout = options.timeout_ms / 1000.0

            # 发起流式请求（带重试）
            vertex_stream = await retry_google_request(
                lambda: _make_vertex_request(
                    config=config,
                    api_key=api_key,
                    contents=contents,
                    params=params,
                    options_headers=options.headers if options else None,
                    timeout=timeout,
                ),
                options,
            )

            # 事件开始
            event_stream.push(AssistantMessageSnapshot(message=output))

            # 流式处理
            current_block_type: str | None = None
            current_block_index: int = -1

            async for chunk in vertex_stream:
                # 保留 responseId
                response_id = chunk.get("responseId")
                if response_id:
                    object.__setattr__(output, "response_id", response_id)

                candidate = chunk.get("candidates", [None])[0]
                if candidate and candidate.get("content", {}).get("parts"):
                    content_parts = candidate["content"]["parts"]
                    for part in content_parts:
                        part_text = part.get("text")
                        if part_text is not None:
                            is_thinking = is_thinking_part(part)
                            if (
                                current_block_type is None
                                or (is_thinking and current_block_type != "thinking")
                                or (not is_thinking and current_block_type != "text")
                            ):
                                if current_block_type is not None:
                                    event_stream.push(
                                        AssistantMessageSnapshot(message=output)
                                    )
                                if is_thinking:
                                    think_block = ThinkingBlock(text="", signature=None)
                                    output.thinking = (output.thinking or []) + [
                                        think_block
                                    ]
                                    current_block_type = "thinking"
                                    current_block_index = len(output.thinking) - 1
                                    event_stream.push(
                                        AssistantMessageSnapshot(message=output)
                                    )
                                else:
                                    text_block = TextContent(text="")
                                    output.content.append(text_block)
                                    current_block_type = "text"
                                    current_block_index = len(output.content) - 1
                                    event_stream.push(
                                        AssistantMessageSnapshot(message=output)
                                    )

                            if current_block_type == "thinking":
                                assert output.thinking is not None
                                think_block = output.thinking[current_block_index]
                                think_block.text += part_text
                                new_sig = retain_thought_signature(
                                    think_block.signature,
                                    part.get("thoughtSignature"),
                                )
                                if new_sig != think_block.signature:
                                    object.__setattr__(
                                        think_block, "signature", new_sig
                                    )
                                event_stream.push(
                                    AssistantThinkingDelta(delta=part_text)
                                )
                            else:
                                text_block = cast(
                                    TextContent,
                                    output.content[current_block_index],
                                )
                                text_block.text += part_text
                                new_sig = retain_thought_signature(
                                    getattr(text_block, "text_signature", None),
                                    part.get("thoughtSignature"),
                                )
                                if new_sig is not None:
                                    object.__setattr__(
                                        text_block, "text_signature", new_sig
                                    )
                                event_stream.push(AssistantTextDelta(delta=part_text))

                        # 工具调用
                        function_call = part.get("functionCall")
                        if function_call:
                            if current_block_type is not None:
                                event_stream.push(
                                    AssistantMessageSnapshot(message=output)
                                )
                                current_block_type = None
                                current_block_index = -1

                            # 生成唯一 ID
                            provided_id = function_call.get("id")
                            needs_new_id = not provided_id or any(
                                b.type == "toolCall"
                                and getattr(b, "tool_call_id", None) == provided_id
                                for b in output.content
                            )
                            tool_call_id = (
                                f"{function_call.get('name', '')}_"
                                f"{int(time.time() * 1000)}_"
                                f"{_next_tool_call_id()}"
                                if needs_new_id
                                else provided_id
                            )

                            tool_call = ToolCallContent(
                                tool_call_id=tool_call_id,
                                name=function_call.get("name", ""),
                                args=function_call.get("args", {}) or {},
                            )
                            # 设置 thought_signature（如果存在）
                            thought_sig = part.get("thoughtSignature")
                            if thought_sig:
                                object.__setattr__(
                                    tool_call, "thought_signature", thought_sig
                                )

                            output.content.append(tool_call)
                            event_stream.push(
                                AssistantToolCallStart(
                                    tool_call_id=tool_call_id,
                                    name=function_call.get("name", ""),
                                )
                            )
                            event_stream.push(
                                AssistantToolCallUpdate(
                                    tool_call_id=tool_call_id,
                                    args=function_call.get("args", {}) or {},
                                )
                            )
                            event_stream.push(
                                AssistantToolCallEnd(
                                    tool_call_id=tool_call_id,
                                    content=[tool_call],
                                )
                            )

                # 停止原因
                if candidate and candidate.get("finishReason"):
                    finish_reason = candidate["finishReason"]
                    object.__setattr__(output, "raw_stop_reason", finish_reason)
                    output.stop_reason = map_stop_reason(finish_reason)
                    if any(b.type == "toolCall" for b in output.content):
                        output.stop_reason = "tool_use"

                # 用量
                usage_metadata = chunk.get("usageMetadata")
                if usage_metadata:
                    prompt_tokens = usage_metadata.get("promptTokenCount", 0) or 0
                    cached_tokens = (
                        usage_metadata.get("cachedContentTokenCount", 0) or 0
                    )
                    candidates_tokens = (
                        usage_metadata.get("candidatesTokenCount", 0) or 0
                    )
                    thoughts_tokens = usage_metadata.get("thoughtsTokenCount", 0) or 0
                    total_tokens = usage_metadata.get("totalTokenCount", 0) or 0

                    usage.input = prompt_tokens - cached_tokens
                    usage.output = candidates_tokens + thoughts_tokens
                    usage.cache_read = cached_tokens
                    usage.cache_write = 0
                    usage.total_tokens = total_tokens
                    if thoughts_tokens > 0:
                        object.__setattr__(usage, "reasoning", thoughts_tokens)
                    calculate_cost(model, usage)

            # 结束当前块
            if current_block_type is not None:
                event_stream.push(AssistantMessageSnapshot(message=output))

            # 检查中止
            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError(
                    "Google Vertex AI stream ended without a finish reason"
                )
            if output.stop_reason in ("aborted", "error"):
                error_message = (
                    getattr(output, "raw_stop_reason", None)
                    or "An unknown error occurred"
                )
                if isinstance(error_message, str):
                    raise RuntimeError(f"Provider stopped with: {error_message}")
                raise RuntimeError("An unknown error occurred")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # 清理临时字段
            for block in output.content:
                if hasattr(block, "thought_signature") and block.type != "toolCall":
                    try:
                        delattr(block, "thought_signature")
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
            output.error_message = format_vertex_error(error)
            event_stream.push(
                AssistantErrorEvent(reason=output.stop_reason, error=output)
            )
            event_stream.end()

    asyncio.ensure_future(_run())
    return event_stream


# ---------------------------------------------------------------------------
# stream_simple - 简化接口
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
            GoogleVertexOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                thinking={"enabled": False},
            ),
        )

    clamped_reasoning = clamp_thinking_level(
        model, getattr(options, "reasoning", "off")
    )
    effort = cast(
        "_ClampedThinkingLevel",
        "high" if clamped_reasoning == "off" else clamped_reasoning,
    )

    model_id = _get_model_id(model)

    if _is_gemini_3_pro_model(model_id) or _is_gemini_3_flash_model(model_id):
        return stream(
            model,
            context,
            GoogleVertexOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                thinking={
                    "enabled": True,
                    "level": _get_thinking_level(effort, model),
                },
            ),
        )

    return stream(
        model,
        context,
        GoogleVertexOptions(
            **base.model_dump() if hasattr(base, "model_dump") else dict(base),
            thinking={
                "enabled": True,
                "budgetTokens": _get_google_budget(
                    model,
                    effort,
                    getattr(options, "thinking_budgets", None),
                ),
            },
        ),
    )
