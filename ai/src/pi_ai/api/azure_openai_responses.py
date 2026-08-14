"""Azure OpenAI Responses API 主入口"""

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

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

DEFAULT_AZURE_API_VERSION = "v1"
AZURE_TOOL_CALL_PROVIDERS = frozenset(
    {"openai", "openai-codex", "opencode", "azure-openai-responses"}
)
# OpenAI Responses 拒绝 max_output_tokens 低于 16
# https://github.com/earendil-works/pi/issues/6265
OPENAI_RESPONSES_MIN_OUTPUT_TOKENS = 16

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def parse_deployment_name_map(value: str | None) -> dict[str, str]:
    """解析部署名称映射字符串。

    格式：``modelId=deploymentName,modelId2=deploymentName2``
    """
    result: dict[str, str] = {}
    if not value:
        return result
    for entry in value.split(","):
        trimmed = entry.strip()
        if not trimmed:
            continue
        parts = trimmed.split("=", 1)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            continue
        result[parts[0].strip()] = parts[1].strip()
    return result


def resolve_deployment_name(
    model: Any,
    options: AzureOpenAIResponsesOptions | None = None,
) -> str:
    """解析部署名称。"""
    if options and options.azure_deployment_name:
        return options.azure_deployment_name
    mapped_deployment = parse_deployment_name_map(
        get_provider_env_value(
            "AZURE_OPENAI_DEPLOYMENT_NAME_MAP",
            options.env if options else None,
        )
    ).get(getattr(model, "model_id", ""))
    return mapped_deployment or cast(str, getattr(model, "model_id", ""))


def format_azure_openai_error(error: object) -> str:
    """格式化 Azure OpenAI 错误。"""
    return format_provider_error(
        normalize_provider_error(error), "Azure OpenAI API error"
    )


def normalize_azure_base_url(base_url: str) -> str:
    """规范化 Azure OpenAI 基础 URL。"""
    trimmed = base_url.strip().rstrip("/")
    try:
        from urllib.parse import urlparse, urlunparse

        parsed = urlparse(trimmed)
    except Exception:
        raise ValueError(f"Invalid Azure OpenAI base URL: {base_url}")

    is_azure_host = parsed.hostname is not None and (
        parsed.hostname.endswith(".openai.azure.com")
        or parsed.hostname.endswith(".cognitiveservices.azure.com")
        or parsed.hostname.endswith(".ai.azure.com")
    )
    normalized_path = parsed.path.rstrip("/")

    # 确保 Azure 主机有 /openai/v1 作为基础路径
    if is_azure_host and normalized_path in (
        "",
        "/",
        "/openai",
        "/openai/v1/responses",
    ):
        parsed = parsed._replace(path="/openai/v1", query="")
        return urlunparse(parsed).rstrip("/")

    return trimmed


def build_default_base_url(resource_name: str) -> str:
    """构建默认的 Azure OpenAI 基础 URL。"""
    return f"https://{resource_name}.openai.azure.com/openai/v1"


def resolve_azure_config(
    model: Any,
    options: AzureOpenAIResponsesOptions | None = None,
) -> dict[str, str]:
    """解析 Azure 配置。"""
    api_version = (
        (options.azure_api_version if options else None)
        or get_provider_env_value(
            "AZURE_OPENAI_API_VERSION",
            options.env if options else None,
        )
        or DEFAULT_AZURE_API_VERSION
    )

    base_url = (
        (options.azure_base_url.strip() if options and options.azure_base_url else None)
        or (
            get_provider_env_value(
                "AZURE_OPENAI_BASE_URL",
                options.env if options else None,
            )
            or ""
        ).strip()
        or None
    )
    resource_name = (
        options.azure_resource_name if options else None
    ) or get_provider_env_value(
        "AZURE_OPENAI_RESOURCE_NAME",
        options.env if options else None,
    )

    resolved_base_url = base_url

    if not resolved_base_url and resource_name:
        resolved_base_url = build_default_base_url(resource_name)

    if not resolved_base_url:
        resolved_base_url = getattr(model, "base_url", None) or ""

    if not resolved_base_url:
        raise ValueError(
            "Azure OpenAI base URL is required. "
            "Set AZURE_OPENAI_BASE_URL or AZURE_OPENAI_RESOURCE_NAME, "
            "or pass azure_base_url, azure_resource_name, or model.base_url."
        )

    return {
        "base_url": normalize_azure_base_url(resolved_base_url),
        "api_version": api_version,
    }


# ---------------------------------------------------------------------------
# 延迟导入（避免循环依赖）
# ---------------------------------------------------------------------------

from openai import AsyncAzureOpenAI

from ..models import clamp_thinking_level
from ..types import (
    StreamOptions,
)
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import retry_provider_request
from .constrained_sampling import create_grammar_tool_input_properties
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
# AzureOpenAIResponsesOptions
# ---------------------------------------------------------------------------


class AzureOpenAIResponsesOptions(StreamOptions):
    """Azure OpenAI Responses API 特定选项。"""

    reasoning_effort: (
        Literal["minimal", "low", "medium", "high", "xhigh", "max"] | None
    ) = None
    reasoning_summary: Literal["auto", "detailed", "concise"] | None = None
    azure_api_version: str | None = None
    azure_resource_name: str | None = None
    azure_base_url: str | None = None
    azure_deployment_name: str | None = None
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


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: AzureOpenAIResponsesOptions | None = None,
) -> AssistantMessageEventStream:
    """Azure OpenAI Responses API 流式生成函数。"""
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        deployment_name = resolve_deployment_name(model, options)

        output = AssistantMessage(
            role="assistant",
            content=[],
            api="azure-openai-responses",
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
            # 获取 API key
            api_key = options.api_key if options else None
            if not api_key:
                raise ValueError(
                    f"No API key for provider: {getattr(model, 'provider', '')}"
                )

            # 创建 Azure OpenAI 客户端
            client = create_client(model, api_key, options)
            compat = getattr(model, "compat", None) or {}
            supports_grammar = (
                compat.get("supportsOpenAIGrammarTools", False)
                if isinstance(compat, dict)
                else False
            )
            grammar_tool_input_properties = create_grammar_tool_input_properties(
                context.tools,
                supports_grammar,
            )
            params = build_params(
                model, context, options, deployment_name, grammar_tool_input_properties
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

            await process_responses_stream(
                openai_stream,
                output,
                event_stream,
                model,
                None,  # 无需 service_tier 选项
            )

            # 检查中止
            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError(
                    "Azure OpenAI Responses stream ended without a stop reason"
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
            output.error_message = format_azure_openai_error(error)
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
    """简化的 Azure OpenAI 流式接口。"""
    api_key = options.api_key if options else None
    if not api_key:
        raise ValueError(f"No API key for provider: {getattr(model, 'provider', '')}")

    base = build_base_options(model, context, options, api_key)
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
        AzureOpenAIResponsesOptions(
            **base.model_dump() if hasattr(base, "model_dump") else dict(base),
            reasoning_effort=reasoning_effort,
            api_key=api_key,
        ),
    )


# ---------------------------------------------------------------------------
# 内部函数
# ---------------------------------------------------------------------------


def create_client(
    model: Any,
    api_key: str,
    options: AzureOpenAIResponsesOptions | None = None,
) -> Any:
    """创建 Azure OpenAI 客户端。"""
    headers: dict[str, str | None] = {}

    # 复制模型级请求头
    model_headers = getattr(model, "headers", None) or {}
    if isinstance(model_headers, dict):
        headers.update(model_headers)

    # 选项级请求头最后合并，可覆盖默认值
    if options and options.headers:
        headers.update(options.headers)

    # 过滤掉 None 值
    filtered_headers: dict[str, str] = {
        k: v for k, v in headers.items() if v is not None
    }

    azure_config = resolve_azure_config(model, options)

    return AsyncAzureOpenAI(
        api_key=api_key,
        api_version=azure_config["api_version"],
        default_headers=filtered_headers,
        base_url=azure_config["base_url"],
        http_client=options.fetch if options else None,
    )


def build_params(
    model: Any,
    context: Context,
    options: AzureOpenAIResponsesOptions | None = None,
    deployment_name: str = "",
    grammar_tool_input_properties: dict[str, str] | None = None,
) -> dict[str, Any]:
    """构建请求参数。"""    
    if grammar_tool_input_properties is None:
        grammar_tool_input_properties = create_grammar_tool_input_properties(
            context.tools,
            False,
        )

    messages = convert_responses_messages(
        model,
        context,
        set(AZURE_TOOL_CALL_PROVIDERS),
        ConvertResponsesMessagesOptions(
            grammar_tool_input_properties=grammar_tool_input_properties,
            tool_options=ConvertResponsesToolsOptions(
                supports_strict_mode=getattr(
                    getattr(model, "compat", None) or {}, "supportsStrictMode", True
                ),
                supports_openai_grammar_tools=getattr(
                    getattr(model, "compat", None) or {},
                    "supportsOpenAIGrammarTools",
                    False,
                ),
            ),
        ),
    )

    params: dict[str, Any] = {
        "model": deployment_name,
        "input": messages,
        "stream": True,
        "prompt_cache_key": clamp_openai_prompt_cache_key(
            options.session_id if options else None
        ),
        "store": False,
    }

    # max_output_tokens
    if options and options.max_tokens:
        params["max_output_tokens"] = max(
            options.max_tokens, OPENAI_RESPONSES_MIN_OUTPUT_TOKENS
        )

    # temperature
    if options and options.temperature is not None:
        params["temperature"] = options.temperature

    # tools
    if context.tools:
        params["tools"] = convert_responses_tools(
            context.tools,
            ConvertResponsesToolsOptions(
                supports_strict_mode=getattr(
                    getattr(model, "compat", None) or {},
                    "supportsStrictMode",
                    True,
                ),
                supports_openai_grammar_tools=getattr(
                    getattr(model, "compat", None) or {},
                    "supportsOpenAIGrammarTools",
                    False,
                ),
            ),
        )

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
            cast(
                "dict[str, object] | None",
                getattr(model, "thinking_level_map", None) or None,
            )
            is not None
        ):
            thinking_level_map = cast(
                "dict[str, object]",
                getattr(model, "thinking_level_map", None) or {},
            )
            if thinking_level_map.get("off") is not None:
                params["reasoning"] = {"effort": thinking_level_map.get("off")}

    # 自定义采样参数最后设置，覆盖命名请求字段
    if options and options.sampling_params:
        params.update(options.sampling_params)

    return params
