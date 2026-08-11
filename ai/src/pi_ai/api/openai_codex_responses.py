"""OpenAI Codex Responses API 实现（对应 ``openai-codex-responses.ts``）。

提供 ``stream``、``stream_simple`` 函数，以及 WebSocket 相关占位实现。
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import AsyncGenerator, AsyncIterable, Callable, Coroutine
from typing import Any, Literal, cast

import httpx

from ..types import (
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    CacheRetention,
    Context,
    Cost,
    ProviderEnv,
    ProviderHeaders,
    StreamOptions,
    Transport,
    Usage,
)
from ..utils.abort import CancellationToken
from ..utils.deferred_tools import split_deferred_tools
from ..utils.diagnostics import (
    format_thrown_value,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from .constrained_sampling import create_grammar_tool_input_properties
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options

# ============================================================================
# Configuration
# ============================================================================

DEFAULT_CODEX_BASE_URL = "https://chatgpt.com/backend-api"
JWT_CLAIM_PATH = "https://api.openai.com/auth"
DEFAULT_MAX_RETRIES = 0
BASE_DELAY_MS = 1000
DEFAULT_MAX_RETRY_DELAY_MS = 60_000
DEFAULT_WEBSOCKET_CONNECT_TIMEOUT_MS = 15_000
CODEX_TOOL_CALL_PROVIDERS: set[str] = {"openai", "openai-codex", "opencode"}
WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE = "websocket_connection_limit_reached"
PREVIOUS_RESPONSE_NOT_FOUND_CODE = "previous_response_not_found"

CODEX_RESPONSE_STATUSES = frozenset(
    {
        "completed",
        "incomplete",
        "failed",
        "cancelled",
        "queued",
        "in_progress",
    }
)

CodexResponseStatus = Literal[
    "completed", "incomplete", "failed", "cancelled", "queued", "in_progress"
]

# ============================================================================
# Types
# ============================================================================


class OpenAICodexResponsesOptions(StreamOptions):
    """OpenAI Codex Responses 选项（对应 TS ``OpenAICodexResponsesOptions``）。"""

    reasoning_effort: (
        Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    reasoning_summary: Literal["auto", "concise", "detailed", "off", "on"] | None = None
    service_tier: str | None = None
    text_verbosity: Literal["low", "medium", "high"] | None = None
    tool_choice: Literal["auto", "none", "required"] | None = None
    signal: CancellationToken | None = None
    on_payload: (
        Callable[[dict[str, Any], Any], Coroutine[Any, Any, dict[str, Any] | None]]
        | None
    ) = None
    on_response: Callable[[dict[str, Any], Any], Coroutine[Any, Any, None]] | None = (
        None
    )
    fetch: Any = None
    env: ProviderEnv | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    sampling_params: dict[str, Any] | None = None
    headers: ProviderHeaders | None = None
    transport: Transport | None = None
    websocket_connect_timeout_ms: int | None = None


class RequestBody(dict[str, Any]):
    """Codex 请求体（对应 TS ``RequestBody``）。"""


SuccessfulAssistantMessage = AssistantMessage
"""type alias for runtime checking — we use ``assert_successful_output`` instead."""


# ============================================================================
# Retry Helpers
# ============================================================================


def is_terminal_rate_limit_error(error_text: str) -> bool:
    """检查是否为终态速率限制错误。"""
    import re

    return bool(
        re.search(
            r"GoUsageLimitError|FreeUsageLimitError|Monthly usage limit reached|"
            r"available balance|insufficient_quota|out of budget|quota exceeded|billing",
            error_text,
            re.IGNORECASE,
        )
    )


def is_retryable_error(status: int, error_text: str) -> bool:
    """检查是否可重试。"""
    if status == 429 and is_terminal_rate_limit_error(error_text):
        return False
    if status in (429, 500, 502, 503, 504):
        return True
    import re

    return bool(
        re.search(
            r"rate.?limit|overloaded|service.?unavailable|upstream.?connect|connection.?refused",
            error_text,
            re.IGNORECASE,
        )
    )


def get_retry_after_delay_ms(headers: httpx.Headers) -> int | None:
    """从响应头中获取重试延迟。"""
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            millis = float(retry_after_ms)
            if millis >= 0:
                return int(millis)
        except (ValueError, TypeError):
            pass

    retry_after = headers.get("retry-after")
    if not retry_after:
        return None

    try:
        seconds = float(retry_after)
        if seconds >= 0:
            return int(seconds * 1000)
    except (ValueError, TypeError):
        pass

    try:
        from email.utils import parsedate_to_datetime

        parsed = parsedate_to_datetime(retry_after)
        delay = int((parsed.timestamp() * 1000) - time.time() * 1000)
        return max(0, delay)
    except (ValueError, TypeError, OverflowError):
        pass

    return None


class RetryDelayExceededError(Exception):
    """重试延迟超出限制。"""


def validate_retry_delay_ms(delay_ms: int, options: StreamOptions | None = None) -> int:
    """验证重试延迟是否在允许范围内。"""
    max_retry_delay_ms = (
        options.max_retry_delay_ms if options else DEFAULT_MAX_RETRY_DELAY_MS
    )
    if max_retry_delay_ms and max_retry_delay_ms > 0 and delay_ms > max_retry_delay_ms:
        raise RetryDelayExceededError(
            f"Server requested {delay_ms // 1000}s retry delay "
            f"(max: {max_retry_delay_ms // 1000}s)"
        )
    return delay_ms


async def sleep(ms: int, signal: CancellationToken | None = None) -> None:
    """异步睡眠，支持取消信号。"""
    if signal and signal.aborted:
        raise RuntimeError("Request was aborted")
    try:
        await asyncio.wait_for(asyncio.sleep(ms / 1000.0), timeout=None)
    except asyncio.CancelledError:
        if signal and signal.aborted:
            raise RuntimeError("Request was aborted") from None
        raise


def normalize_timeout_ms(value: int | None) -> int | None:
    """规范化超时时间（毫秒）。"""
    if value is None:
        return None
    if value < 0:
        raise ValueError(f"Invalid timeoutMs: {value}")
    return int(value)


# ============================================================================
# Main Stream Function
# ============================================================================


def stream(
    model: Any,
    context: Context,
    options: OpenAICodexResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Codex Responses API 流式生成（对应 TS ``stream``）。"""
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

        try:
            api_key = options.api_key if options else None
            if not api_key:
                raise ValueError(
                    f"No API key for provider: {getattr(model, 'provider', '')}"
                )

            account_id = extract_account_id(api_key)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools,
                getattr(
                    getattr(model, "compat", None), "supportsOpenAIGrammarTools", False
                )
                if hasattr(model, "compat") and model.compat is not None
                else False,
            )
            cache_session_id = (
                None
                if (options and options.cache_retention == "none")
                else (options.session_id if options else None)
            )
            codex_session_id = clamp_openai_prompt_cache_key(cache_session_id)
            body = build_request_body(
                model, context, options, codex_session_id, grammar_tool_input_properties
            )

            if options and options.on_payload:
                next_body = await options.on_payload(dict(body), model)
                if next_body is not None:
                    body = RequestBody(next_body)

            body_json = json.dumps(body)
            http_timeout_ms = normalize_timeout_ms(
                options.timeout_ms if options else None
            )
            sse_headers = build_sse_headers(
                getattr(model, "headers", None),
                options.headers if options else None,
                account_id,
                api_key,
                codex_session_id,
            )

            # Fetch with retry logic
            response: httpx.Response | None = None
            last_error: Exception | None = None
            max_retries = options.max_retries if options else DEFAULT_MAX_RETRIES
            base_url = getattr(model, "base_url", None) or ""
            codex_url = resolve_codex_url(base_url)

            max_retries_val: int = (
                max_retries if max_retries is not None else DEFAULT_MAX_RETRIES
            )
            for attempt in range(max_retries_val + 1):
                if options and options.signal and options.signal.aborted:
                    raise RuntimeError("Request was aborted")

                try:
                    timeout_sec = (
                        http_timeout_ms / 1000.0
                        if http_timeout_ms is not None
                        else None
                    )
                    async with httpx.AsyncClient(timeout=timeout_sec) as client:
                        response = await client.post(
                            codex_url,
                            headers=dict(sse_headers),
                            content=body_json,
                        )

                    if options and options.on_response:
                        await options.on_response(
                            {
                                "status": response.status_code,
                                "headers": dict(response.headers),
                            },
                            model,
                        )

                    if response.is_success:
                        break

                    error_text = response.text
                    if attempt < max_retries_val and is_retryable_error(
                        response.status_code, error_text
                    ):
                        retry_after_delay_ms = get_retry_after_delay_ms(
                            response.headers
                        )
                        delay_ms = (
                            retry_after_delay_ms
                            if retry_after_delay_ms is not None
                            else BASE_DELAY_MS * (2**attempt)
                        )
                        if retry_after_delay_ms is not None:
                            validate_retry_delay_ms(delay_ms, options)
                        await sleep(delay_ms, options.signal if options else None)
                        continue

                    # Parse error for friendly message
                    info = await parse_error_response(
                        error_text, response.status_code, response.reason_phrase or ""
                    )
                    raise RuntimeError(info.get("friendlyMessage") or info["message"])

                except Exception as error:
                    if (
                        isinstance(error, RuntimeError)
                        and str(error) == "Request was aborted"
                    ):
                        raise
                    last_error = (
                        error
                        if isinstance(error, Exception)
                        else RuntimeError(str(error))
                    )
                    if (
                        attempt < max_retries_val
                        and not isinstance(last_error, RetryDelayExceededError)
                        and "usage limit" not in str(last_error).lower()
                    ):
                        delay_ms = BASE_DELAY_MS * (2**attempt)
                        await sleep(delay_ms, options.signal if options else None)
                        continue
                    raise last_error

            if not response or not response.is_success:
                raise last_error or RuntimeError("Failed after retries")

            event_stream.push(AssistantMessageSnapshot(message=output))

            await process_stream(
                response,
                output,
                event_stream,
                model,
                grammar_tool_input_properties,
                options,
            )

            if options and options.signal and options.signal.aborted:
                raise RuntimeError("Request was aborted")

            assert_successful_output(output)
            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            for block in output.content:
                for field in (
                    "partialJson",
                    "customInput",
                    "partial_json",
                    "custom_input",
                ):
                    if hasattr(block, field):
                        try:
                            delattr(block, field)
                        except (AttributeError, TypeError):
                            pass

            output.stop_reason = (
                "aborted"
                if (options and options.signal and options.signal.aborted)
                else "error"
            )
            output.error_message = format_provider_error(
                normalize_provider_error(error)
            )
            event_stream.push(
                AssistantErrorEvent(reason=output.stop_reason, error=output)
            )
            event_stream.end()

    asyncio.ensure_future(_run())
    return event_stream


def stream_simple(
    model: Any,
    context: Context,
    options: Any | None = None,
) -> AssistantMessageEventStream:
    """简化的流式接口（对应 TS ``streamSimple``）。"""
    api_key = options.api_key if options else None
    if not api_key:
        raise ValueError(f"No API key for provider: {getattr(model, 'provider', '')}")

    base = build_base_options(model, context, options, api_key)

    from ..models import clamp_thinking_level

    clamped_reasoning = (
        clamp_thinking_level(model, options.reasoning)
        if options and getattr(options, "reasoning", None)
        else None
    )
    reasoning_effort = None if clamped_reasoning == "off" else clamped_reasoning

    return stream(
        model,
        context,
        OpenAICodexResponsesOptions(
            **base.model_dump(),
            reasoning_effort=reasoning_effort,
        ),
    )


# ============================================================================
# Request Building
# ============================================================================


def build_request_body(
    model: Any,
    context: Context,
    options: OpenAICodexResponsesOptions | None = None,
    cache_session_id: str | None = None,
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    """构建 Codex 请求体（对应 TS ``buildRequestBody``）。"""
    compat = getattr(model, "compat", None) or {}
    supports_strict_mode = (
        compat.get("supportsStrictMode", True) if isinstance(compat, dict) else True
    )
    supports_openai_grammar_tools = (
        compat.get("supportsOpenAIGrammarTools", False)
        if isinstance(compat, dict)
        else False
    )
    immediate_tools, deferred_tools_map = split_deferred_tools(
        context,
        compat.get("supportsToolSearch", False) if isinstance(compat, dict) else False,
    )

    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools, supports_openai_grammar_tools
        )

    messages = convert_responses_messages(
        model,
        context,
        CODEX_TOOL_CALL_PROVIDERS,
        ConvertResponsesMessagesOptions(
            include_system_prompt=False,
            grammar_tool_input_properties=grammar_tool_input_properties,
            deferred_tools=deferred_tools_map if deferred_tools_map else None,
            tool_options=ConvertResponsesToolsOptions(
                strict=None,
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_openai_grammar_tools,
            ),
        ),
    )

    body: dict[str, Any] = {
        "model": getattr(model, "model_id", ""),
        "store": False,
        "stream": True,
        "instructions": context.system_prompt or "You are a helpful assistant.",
        "input": messages,
        "text": {"verbosity": options.text_verbosity if options else "low"},
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": cache_session_id,
        "tool_choice": options.tool_choice if options else "auto",
        "parallel_tool_calls": True,
    }

    if options and options.temperature is not None:
        body["temperature"] = options.temperature

    if options and options.service_tier is not None:
        body["service_tier"] = options.service_tier

    if immediate_tools:
        body["tools"] = convert_responses_tools(
            immediate_tools,
            ConvertResponsesToolsOptions(
                strict=None,
                supports_strict_mode=supports_strict_mode,
                supports_openai_grammar_tools=supports_openai_grammar_tools,
            ),
        )

    if options and options.reasoning_effort is not None:
        effort = options.reasoning_effort
        thinking_level_map = getattr(model, "thinking_level_map", None) or {}
        if effort == "none":
            mapped_effort = thinking_level_map.get("off", "none")
        else:
            mapped_effort = thinking_level_map.get(effort, effort)
        if mapped_effort is not None:
            body["reasoning"] = {
                "effort": mapped_effort,
                "summary": options.reasoning_summary or "auto",
            }

    return body


def get_service_tier_cost_multiplier(
    model: Any,
    service_tier: str | None,
) -> float:
    """获取服务层成本乘数（对应 TS ``getServiceTierCostMultiplier``）。"""
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        model_id = getattr(model, "model_id", "")
        return 2.5 if model_id == "gpt-5.5" else 2.0
    return 1.0


def apply_service_tier_pricing(
    usage: Usage,
    service_tier: str | None,
    model: Any,
) -> None:
    """应用服务层定价调整（对应 TS ``applyServiceTierPricing``）。"""
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1.0:
        return
    if usage.cost is not None:
        usage.cost.input *= multiplier
        usage.cost.output *= multiplier
        usage.cost.cache_read *= multiplier
        usage.cost.cache_write *= multiplier
        usage.cost.total = (
            usage.cost.input
            + usage.cost.output
            + usage.cost.cache_read
            + usage.cost.cache_write
        )


def resolve_codex_service_tier(
    response_service_tier: str | None,
    request_service_tier: str | None,
) -> str | None:
    """解析 Codex 服务层（对应 TS ``resolveCodexServiceTier``）。"""
    if response_service_tier == "default" and request_service_tier in (
        "flex",
        "priority",
    ):
        return request_service_tier
    return response_service_tier or request_service_tier


def resolve_codex_url(base_url: str | None = None) -> str:
    """解析 Codex URL（对应 TS ``resolveCodexUrl``）。"""
    raw = base_url.strip() if base_url and base_url.strip() else DEFAULT_CODEX_BASE_URL
    normalized = raw.rstrip("/")
    if normalized.endswith("/codex/responses"):
        return normalized
    if normalized.endswith("/codex"):
        return f"{normalized}/responses"
    return f"{normalized}/codex/responses"


def resolve_codex_websocket_url(base_url: str | None = None) -> str:
    """解析 Codex WebSocket URL（对应 TS ``resolveCodexWebSocketUrl``）。"""
    url = resolve_codex_url(base_url)
    if url.startswith("https:"):
        url = "wss:" + url[6:]
    elif url.startswith("http:"):
        url = "ws:" + url[5:]
    return url


# ============================================================================
# Response Processing
# ============================================================================


async def process_stream(
    response: httpx.Response,
    output: AssistantMessage,
    stream: AssistantMessageEventStream,
    model: Any,
    grammar_tool_input_properties: dict[str, str],
    options: OpenAICodexResponsesOptions | None = None,
) -> None:
    """处理 Codex 流式响应（对应 TS ``processStream``）。"""
    await process_responses_stream(
        map_codex_events(parse_sse(response)),
        output,
        stream,
        model,
        OpenAIResponsesStreamOptions(
            service_tier=options.service_tier if options else None,
            grammar_tool_input_properties=grammar_tool_input_properties,
            resolve_service_tier=resolve_codex_service_tier,
            apply_service_tier_pricing=lambda usage, service_tier: (
                apply_service_tier_pricing(usage, service_tier, model)
            ),
        ),
    )


class CodexApiError(Exception):
    """Codex API 错误（对应 TS ``CodexApiError``）。"""

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        payload: dict[str, Any] | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.code = code
        self.payload = payload
        self.cause = cause
        super().__init__(message)


class CodexProtocolError(Exception):
    """Codex 协议错误（对应 TS ``CodexProtocolError``）。"""

    def __init__(
        self,
        message: str,
        *,
        payload: Any = None,
        cause: BaseException | None = None,
    ) -> None:
        self.payload = payload
        self.cause = cause
        super().__init__(message)


def is_codex_non_transport_error(error: object) -> bool:
    """检查是否为 Codex 非传输层错误。"""
    return isinstance(error, (CodexApiError, CodexProtocolError))


def is_websocket_connection_limit_reached_error(error: object) -> bool:
    """检查是否为 WebSocket 连接限制错误。"""
    return (
        isinstance(error, CodexApiError)
        and error.code == WEBSOCKET_CONNECTION_LIMIT_REACHED_CODE
    )


def is_previous_response_not_found_error(error: object) -> bool:
    """检查是否为前一个响应未找到错误。"""
    return (
        isinstance(error, CodexApiError)
        and error.code == PREVIOUS_RESPONSE_NOT_FOUND_CODE
    )


def extract_codex_event_error(event: dict[str, Any]) -> dict[str, str | None]:
    """提取 Codex 事件错误信息。"""
    nested = event.get("error")
    if isinstance(nested, dict):
        return {
            "code": (
                event.get("code")
                if isinstance(event.get("code"), str)
                else nested.get("code")
            ),
            "message": (
                event.get("message")
                if isinstance(event.get("message"), str)
                else nested.get("message")
            ),
        }
    return {
        "code": event.get("code") if isinstance(event.get("code"), str) else None,
        "message": event.get("message")
        if isinstance(event.get("message"), str)
        else None,
    }


async def map_codex_events(
    events: AsyncIterable[dict[str, Any]],
) -> AsyncGenerator[Any, None]:
    """映射 Codex 事件到标准 Responses 事件（对应 TS ``mapCodexEvents``）。"""
    async for event in events:
        event_type = event.get("type")
        if not isinstance(event_type, str):
            continue

        if event_type == "error":
            info = extract_codex_event_error(event)
            raise CodexApiError(
                f"Codex error: {info.get('message') or info.get('code') or json.dumps(event)}",
                code=info.get("code"),
                payload=event,
            )

        if event_type == "response.failed":
            response = event.get("response")
            if isinstance(response, dict):
                err = response.get("error")
                code = err.get("code") if isinstance(err, dict) else None
                message = err.get("message") if isinstance(err, dict) else None
            else:
                code = None
                message = None
            raise CodexApiError(
                message or "Codex response failed",
                code=code,
                payload=event,
            )

        if event_type in ("response.done", "response.completed", "response.incomplete"):
            response = event.get("response")
            normalized_response = None
            if isinstance(response, dict):
                normalized_response = {
                    **response,
                    "status": normalize_codex_status(response.get("status")),
                }
            yield {
                **event,
                "type": "response.completed",
                "response": normalized_response,
            }
            return

        yield event


def normalize_codex_status(status: Any) -> CodexResponseStatus | None:
    """规范化 Codex 响应状态。"""
    if not isinstance(status, str):
        return None
    if status in CODEX_RESPONSE_STATUSES:
        return cast(CodexResponseStatus, status)
    return None


# ============================================================================
# SSE Parsing
# ============================================================================


async def parse_sse(response: httpx.Response) -> AsyncGenerator[dict[str, Any], None]:
    """解析 SSE 流（对应 TS ``parseSSE``）。

    使用 httpx 的 ``aiter_lines`` 逐行读取，按 ``\\n\\n`` 分隔事件。
    """
    buffer = ""
    async for line in response.aiter_lines():
        buffer += line + "\n"
        idx = buffer.find("\n\n")
        while idx != -1:
            chunk = buffer[:idx]
            buffer = buffer[idx + 2 :]

            data_lines = [
                l[5:].strip() for l in chunk.split("\n") if l.startswith("data:")
            ]
            if data_lines:
                data = "\n".join(data_lines).strip()
                if data and data != "[DONE]":
                    try:
                        yield json.loads(data)
                    except Exception as cause:
                        raise CodexProtocolError(
                            f"Invalid Codex SSE JSON: {format_thrown_value(cause)}",
                            cause=cause,
                            payload=data,
                        ) from cause
            idx = buffer.find("\n\n")


# ============================================================================
# WebSocket（占位实现）
# ============================================================================

OPENAI_BETA_RESPONSES_WEBSOCKETS = "responses_websockets=2026-02-06"
SESSION_WEBSOCKET_CACHE_TTL_MS = 5 * 60 * 1000
SESSION_WEBSOCKET_MAX_AGE_MS = 55 * 60 * 1000

OpenAICodexWebSocketDebugStats: type = dict
"""WebSocket 调试统计（对应 TS ``OpenAICodexWebSocketDebugStats``）。"""

# WebSocket 会话缓存（占位）
_websocket_session_cache: dict[str, dict[str, Any]] = {}
_websocket_debug_stats: dict[str, dict[str, Any]] = {}
_websocket_sse_fallback_sessions: set[str] = set()


def get_or_create_websocket_debug_stats(session_id: str) -> dict[str, Any]:
    """获取或创建 WebSocket 调试统计。"""
    stats = _websocket_debug_stats.get(session_id)
    if stats is None:
        stats = {
            "requests": 0,
            "connectionsCreated": 0,
            "connectionsReused": 0,
            "cachedContextRequests": 0,
            "storeTrueRequests": 0,
            "fullContextRequests": 0,
            "deltaRequests": 0,
            "lastInputItems": 0,
            "websocketFailures": 0,
            "sseFallbacks": 0,
        }
        _websocket_debug_stats[session_id] = stats
    return stats


def get_openai_codex_websocket_debug_stats(session_id: str) -> dict[str, Any] | None:
    """获取 WebSocket 调试统计（占位）。"""
    stats = _websocket_debug_stats.get(session_id)
    return dict(stats) if stats else None


def reset_openai_codex_websocket_debug_stats(session_id: str | None = None) -> None:
    """重置 WebSocket 调试统计（占位）。"""
    if session_id:
        _websocket_debug_stats.pop(session_id, None)
        _websocket_sse_fallback_sessions.discard(session_id)
    else:
        _websocket_debug_stats.clear()
        _websocket_sse_fallback_sessions.clear()


def close_openai_codex_websocket_sessions(session_id: str | None = None) -> None:
    """关闭 WebSocket 会话（占位）。"""
    if session_id:
        _websocket_session_cache.pop(session_id, None)
    else:
        _websocket_session_cache.clear()


# ============================================================================
# Auth & Headers
# ============================================================================


def extract_account_id(token: str) -> str:
    """从 JWT token 中提取 account ID（对应 TS ``extractAccountId``）。"""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("Invalid token")
        # Base64 decode payload
        import base64

        padded = parts[1] + "=" * (4 - len(parts[1]) % 4)
        try:
            decoded = base64.urlsafe_b64decode(padded)
        except Exception:
            decoded = base64.b64decode(padded)
        payload = json.loads(decoded)
        auth_claim = payload.get(JWT_CLAIM_PATH, {})
        if not isinstance(auth_claim, dict):
            raise ValueError("No auth claim in token")
        account_id = auth_claim.get("chatgpt_account_id")
        if not account_id:
            raise ValueError("No account ID in token")
        return str(account_id)
    except Exception as e:
        raise ValueError(f"Failed to extract accountId from token: {e}") from e


def build_base_codex_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
) -> dict[str, str]:
    """构建基础 Codex 请求头（对应 TS ``buildBaseCodexHeaders``）。"""
    headers: dict[str, str] = dict(init_headers or {})

    if additional_headers:
        for key, value in additional_headers.items():
            if value is None:
                headers.pop(key, None)
            else:
                headers[key] = value

    headers["Authorization"] = f"Bearer {token}"
    headers["chatgpt-account-id"] = account_id
    headers["originator"] = "pi"
    headers["User-Agent"] = (
        f"pi ({os.uname().sysname} {os.uname().release}; {os.uname().machine})"
    )
    return headers


def build_sse_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
    session_id: str | None = None,
) -> dict[str, str]:
    """构建 SSE 请求头（对应 TS ``buildSSEHeaders``）。"""
    headers = build_base_codex_headers(
        init_headers, additional_headers, account_id, token
    )
    headers["OpenAI-Beta"] = "responses=experimental"
    headers["accept"] = "text/event-stream"
    headers["content-type"] = "application/json"

    if session_id:
        headers["session-id"] = session_id
        headers["x-client-request-id"] = session_id

    return headers


def build_websocket_headers(
    init_headers: dict[str, str] | None,
    additional_headers: ProviderHeaders | None,
    account_id: str,
    token: str,
    request_id: str,
) -> dict[str, str]:
    """构建 WebSocket 请求头（对应 TS ``buildWebSocketHeaders``）。"""
    headers = build_base_codex_headers(
        init_headers, additional_headers, account_id, token
    )
    headers.pop("accept", None)
    headers.pop("content-type", None)
    headers.pop("OpenAI-Beta", None)
    headers.pop("openai-beta", None)
    headers["OpenAI-Beta"] = OPENAI_BETA_RESPONSES_WEBSOCKETS
    headers["x-client-request-id"] = request_id
    headers["session-id"] = request_id
    return headers


# ============================================================================
# Error Handling
# ============================================================================


async def parse_error_response(
    text: str,
    status: int,
    status_text: str,
) -> dict[str, str]:
    """解析错误响应（对应 TS ``parseErrorResponse``）。"""
    message = text or status_text or "Request failed"
    friendly_message: str | None = None

    try:
        parsed = json.loads(text)
        err = parsed.get("error") if isinstance(parsed, dict) else None
        if isinstance(err, dict):
            code = err.get("code") or err.get("type") or ""
            import re

            if (
                re.search(
                    r"usage_limit_reached|usage_not_included|rate_limit_exceeded",
                    code,
                    re.IGNORECASE,
                )
                or status == 429
            ):
                plan = err.get("plan_type")
                plan_text = (
                    f" ({plan.lower()} plan)" if isinstance(plan, str) and plan else ""
                )
                resets_at = err.get("resets_at")
                mins = None
                if isinstance(resets_at, (int, float)):
                    mins = max(
                        0, round((resets_at * 1000 - time.time() * 1000) / 60000)
                    )
                when = f" Try again in ~{mins} min." if mins is not None else ""
                friendly_message = (
                    f"You have hit your ChatGPT usage limit{plan_text}.{when}".strip()
                )
            message = err.get("message") or friendly_message or message
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    return {"message": message, "friendlyMessage": friendly_message or message}


# ============================================================================
# Validation
# ============================================================================


def assert_successful_output(output: AssistantMessage) -> None:
    """断言输出成功（对应 TS ``assertSuccessfulOutput``）。"""
    if output.stop_reason == "pending":
        raise RuntimeError("Codex stream ended without a stop reason")
    if output.stop_reason in ("error", "aborted"):
        raise RuntimeError(output.error_message or "An unknown error occurred")
