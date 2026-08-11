"""OpenAI Responses API 主入口（对应 ``openai-responses.ts``）。"""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

from ..types import (
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    Context,
    Cost,
    Usage,
)
from ..utils.provider_retry import ProviderRetryOptions
from .openai_responses_shared import OpenAIResponsesStreamOptions

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

OPENAI_TOOL_CALL_PROVIDERS = frozenset({"openai", "openai-codex", "opencode"})

# OpenAI Responses 拒绝 max_output_tokens 低于 16
# https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

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


def detect_session_affinity_format(model: Any) -> str:
    """检测会话亲和性格式。"""
    provider = getattr(model, "provider", "")
    base_url = getattr(model, "base_url", "") or ""
    if provider == "openrouter" or "openrouter.ai" in base_url:
        return "openrouter"
    return "openai"


def resolve_cache_retention(
    cache_retention: str | None,
    env: dict[str, str] | None,
) -> str:
    """解析缓存保留策略。

    默认使用 "short"，通过 PI_CACHE_RETENTION 环境变量向后兼容。
    """
    if cache_retention:
        return cache_retention
    if get_provider_env_value("PI_CACHE_RETENTION", env) == "long":
        return "long"
    return "short"


def get_compat(model: Any) -> dict[str, Any]:
    """获取兼容性设置。"""
    compat = getattr(model, "compat", None) or {}
    if isinstance(compat, dict):
        return {
            "supports_developer_role": compat.get("supportsDeveloperRole", True),
            "session_affinity_format": compat.get(
                "sessionAffinityFormat", detect_session_affinity_format(model)
            ),
            "supports_long_cache_retention": compat.get(
                "supportsLongCacheRetention", True
            ),
            "supports_strict_mode": compat.get("supportsStrictMode", False),
            "supports_openai_grammar_tools": compat.get(
                "supportsOpenAIGrammarTools", False
            ),
            "supports_tool_search": compat.get("supportsToolSearch", False),
            "supports_explicit_prompt_cache_mode": compat.get(
                "supportsExplicitPromptCacheMode", False
            ),
        }
    return {
        "supports_developer_role": True,
        "session_affinity_format": detect_session_affinity_format(model),
        "supports_long_cache_retention": True,
        "supports_strict_mode": False,
        "supports_openai_grammar_tools": False,
        "supports_tool_search": False,
        "supports_explicit_prompt_cache_mode": False,
    }


def get_prompt_cache_retention(
    compat: dict[str, Any], cache_retention: str
) -> str | None:
    """获取 prompt cache retention。"""
    if cache_retention == "long" and compat.get("supports_long_cache_retention", True):
        return "24h"
    return None


def format_openai_responses_error(error: object) -> str:
    """格式化 OpenAI Responses 错误。"""
    return format_provider_error(normalize_provider_error(error), "OpenAI API error")


# ---------------------------------------------------------------------------
# 延迟导入（避免循环依赖）
# ---------------------------------------------------------------------------

from openai import AsyncOpenAI

from ..models import clamp_thinking_level
from ..types import (
    StreamOptions,
)
from ..utils.deferred_tools import split_deferred_tools
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from .constrained_sampling import create_grammar_tool_input_properties
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .simple_options import build_base_options

# ---------------------------------------------------------------------------
# OpenAIResponsesOptions
# ---------------------------------------------------------------------------


class OpenAIResponsesOptions(StreamOptions):
    """OpenAI Responses API 特定选项（对应 TS ``OpenAIResponsesOptions``）。"""

    reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = None
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


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: OpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """OpenAI Responses API 流式生成函数（对应 TS ``stream``）。"""
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
            # 创建 OpenAI 客户端
            provider = getattr(model, "provider", "")
            api_key = get_client_api_key(
                provider,
                options.api_key if options else None,
                options.headers if options else None,
            )
            cache_retention = resolve_cache_retention(
                options.cache_retention if options else None,
                options.env if options else None,
            )
            cache_session_id = (
                options.session_id if options and cache_retention != "none" else None
            )
            compat = get_compat(model)
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools,
                compat.get("supports_openai_grammar_tools", False),
            )
            client = create_client(
                model,
                context,
                api_key,
                options.headers if options else None,
                options.fetch if options else None,
                cache_session_id,
            )
            params = build_params(
                model, context, options, compat, grammar_tool_input_properties
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
            if options and options.timeout_ms is not None:
                request_options["timeout"] = options.timeout_ms / 1000.0

            # 发起请求（带重试）
            raw_response = await retry_provider_request(
                lambda: client.responses.with_raw_response.create(
                    **params, **request_options
                ),
                ProviderRetryOptions(
                    max_retries=cast(int, options.max_retries if options else 0),
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=options.signal if options else None,
                ),
            )

            # 解析出流和响应头
            raw_response_parsed = cast(Any, raw_response)
            openai_stream = raw_response_parsed.parse()
            response_headers = dict(raw_response_parsed.headers)

            # onResponse 回调
            if options and options.on_response:
                await options.on_response(
                    {
                        "status": getattr(raw_response, "status_code", 0),
                        "headers": response_headers,
                    },
                    model,
                )

            event_stream.push(AssistantMessageSnapshot(message=output))

            stream_options = OpenAIResponsesStreamOptions(
                service_tier=options.service_tier if options else None,
                grammar_tool_input_properties=grammar_tool_input_properties,
                apply_service_tier_pricing=lambda usage, service_tier: (
                    apply_service_tier_pricing(usage, service_tier, model)
                ),
            )
            await process_responses_stream(
                openai_stream,
                output,
                event_stream,
                model,
                stream_options,
            )

            # 检查中止
            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError(
                    "OpenAI Responses stream ended without a stop reason"
                )
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # 清理临时字段
            for block in output.content:
                for field in ("index", "partial_json", "custom_input"):
                    if hasattr(block, field):
                        try:
                            delattr(block, field)
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
            output.error_message = format_openai_responses_error(error)
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

    return stream(
        model,
        context,
        OpenAIResponsesOptions(
            **base.model_dump() if hasattr(base, "model_dump") else dict(base),
            reasoning_effort=reasoning_effort,
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
) -> Any:
    """创建 OpenAI 客户端（对应 TS ``createClient``）。"""
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
    options: OpenAIResponsesOptions | None = None,
    compat: dict[str, Any] | None = None,
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

    tool_placement = split_deferred_tools(
        context, compat.get("supports_tool_search", False)
    )
    immediate_tools, deferred_tools = (
        tool_placement if isinstance(tool_placement, tuple) else (tool_placement, {})
    )

    messages = convert_responses_messages(
        model,
        context,
        set(OPENAI_TOOL_CALL_PROVIDERS),
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            deferred_tools=deferred_tools,
            tool_options=ConvertResponsesToolsOptions(
                supports_strict_mode=compat.get("supports_strict_mode", False),
                supports_openai_grammar_tools=compat.get(
                    "supports_openai_grammar_tools", False
                ),
            ),
        ),
    )

    cache_retention = resolve_cache_retention(
        options.cache_retention if options else None,
        options.env if options else None,
    )
    disable_implicit_prompt_cache = cache_retention == "none" and compat.get(
        "supports_explicit_prompt_cache_mode", False
    )

    params: dict[str, Any] = {
        "model": getattr(model, "model_id", ""),
        "input": messages,
        "stream": True,
        "store": False,
    }

    # Prompt cache
    if cache_retention != "none":
        params["prompt_cache_key"] = clamp_openai_prompt_cache_key(
            options.session_id if options else None
        )
        prompt_cache_retention = get_prompt_cache_retention(compat, cache_retention)
        if prompt_cache_retention:
            params["prompt_cache_retention"] = prompt_cache_retention

    if disable_implicit_prompt_cache:
        params["prompt_cache_options"] = {"mode": "explicit"}

    # max_output_tokens
    if options and options.max_tokens:
        params["max_output_tokens"] = max(
            options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS
        )

    # temperature
    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    # service_tier
    if options and options.service_tier is not None:
        params["service_tier"] = options.service_tier

    # tools
    if immediate_tools:
        params["tools"] = convert_responses_tools(
            immediate_tools,
            ConvertResponsesToolsOptions(
                supports_strict_mode=compat.get("supports_strict_mode", False),
                supports_openai_grammar_tools=compat.get(
                    "supports_openai_grammar_tools", False
                ),
            ),
        )

    # tool_choice
    if options and options.tool_choice is not None:
        params["tool_choice"] = options.tool_choice

    # reasoning
    if getattr(model, "reasoning", False):
        if (options and options.reasoning_effort) or (
            options and options.reasoning_summary
        ):
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            effort = (
                thinking_level_map.get(
                    options.reasoning_effort, options.reasoning_effort
                )
                if options and options.reasoning_effort
                else "medium"
            )
            reasoning: dict[str, Any] = {
                "effort": effort,
                "summary": options.reasoning_summary
                if options and options.reasoning_summary
                else "auto",
            }
            params["reasoning"] = reasoning
            params["include"] = ["reasoning.encrypted_content"]
        elif (
            getattr(model, "provider", "") != "github-copilot"
            and cast(
                "dict[str, object] | None",
                getattr(model, "thinking_level_map", None) or None,
            )
            is not None
        ):
            thinking_level_map = cast(
                "dict[str, object]", getattr(model, "thinking_level_map", None) or {}
            )
            if thinking_level_map.get("off") is not None:
                params["reasoning"] = {"effort": thinking_level_map}

        if getattr(model, "provider", "") == "xai":
            params["include"] = ["reasoning.encrypted_content"]

    # 自定义采样参数最后设置，覆盖命名请求字段
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params


def get_service_tier_cost_multiplier(
    model: Any,
    service_tier: str | None,
) -> float:
    """获取服务层级成本倍数（对应 TS ``getServiceTierCostMultiplier``）。"""
    if service_tier == "flex":
        return 0.5
    if service_tier == "priority":
        return 2.5 if getattr(model, "model_id", "") == "gpt-5.5" else 2.0
    return 1.0


def apply_service_tier_pricing(
    usage: Usage,
    service_tier: str | None,
    model: Any,
) -> None:
    """应用服务层级定价（对应 TS ``applyServiceTierPricing``）。"""
    multiplier = get_service_tier_cost_multiplier(model, service_tier)
    if multiplier == 1.0:
        return

    if usage.cost:
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
