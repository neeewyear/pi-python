"""Anthropic Messages API 主入口。"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any, Literal, cast

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
    Context,
    Cost,
    ImageContent,
    StreamOptions,
    TextContent,
    ThinkingBlock,
    ToolCallContent,
    Usage,
)
from ..utils.deferred_tools import split_deferred_tools
from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.event_stream import AssistantMessageEventStream
from ..utils.json_parse import parse_streaming_json
from ..utils.provider_env import get_provider_env_value
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates
from .constrained_sampling import resolve_json_schema_strict_sampling
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .simple_options import (
    adjust_max_tokens_for_thinking,
    build_base_options,
    clamp_max_tokens_to_context,
)
from .transform_messages import transform_messages

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

ANTHROPIC_TOOL_CALL_PROVIDERS = frozenset({"anthropic", "anthropic-codex"})

FINE_GRAINED_TOOL_STREAMING_BETA = "fine-grained-tool-streaming-2025-05-14"
INTERLEAVED_THINKING_BETA = "interleaved-thinking-2025-05-14"

# Stealth mode: Mimic Claude Code's tool naming exactly
CLAUDE_CODE_VERSION = "2.1.75"

# Claude Code 2.x tool names (canonical casing)
CLAUDE_CODE_TOOLS = [
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Grep",
    "Glob",
    "AskUserQuestion",
    "EnterPlanMode",
    "ExitPlanMode",
    "KillShell",
    "NotebookEdit",
    "Skill",
    "Task",
    "TaskOutput",
    "TodoWrite",
    "WebFetch",
    "WebSearch",
]

CC_TOOL_LOOKUP: dict[str, str] = {t.lower(): t for t in CLAUDE_CODE_TOOLS}

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

AnthropicEffort = Literal["low", "medium", "high", "xhigh", "max"]
AnthropicThinkingDisplay = Literal["summarized", "omitted"]

# ---------------------------------------------------------------------------
# AnthropicOptions
# ---------------------------------------------------------------------------


class AnthropicOptions(StreamOptions):
    """Anthropic Messages API 流式选项。"""

    thinking_enabled: bool | None = None
    thinking_budget_tokens: int | None = None
    effort: AnthropicEffort | None = None
    thinking_display: AnthropicThinkingDisplay | None = None
    interleaved_thinking: bool | None = None
    tool_choice: Any | None = None
    client: Any | None = None  # Pre-built Anthropic client instance
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
    anthropic_beta: str | None = None


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _to_claude_code_name(name: str) -> str:
    """Convert tool name to CC canonical casing if it matches (case-insensitive)."""
    return CC_TOOL_LOOKUP.get(name.lower(), name)


def _from_claude_code_name(name: str, tools: list[Any] | None = None) -> str:
    """Convert from CC canonical casing back to original tool name."""
    if tools:
        lower_name = name.lower()
        for tool in tools:
            if tool.name.lower() == lower_name:
                return cast(str, tool.name)
    return name


def _resolve_cache_retention(
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


def _get_cache_control(
    model: Any,
    cache_retention: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    """获取缓存控制配置。"""    
    retention = _resolve_cache_retention(cache_retention, env)
    if retention == "none":
        return {"retention": retention}
    compat = _get_anthropic_compat(model)
    ttl = (
        "1h"
        if retention == "long" and compat.get("supports_long_cache_retention")
        else None
    )
    cache_control: dict[str, Any] = {"type": "ephemeral"}
    if ttl:
        cache_control["ttl"] = ttl
    return {"retention": retention, "cache_control": cache_control}


def _has_header(
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


def _assert_request_auth(
    provider: str,
    api_key: str | None,
    headers: dict[str, str | None] | None,
    ) -> None:
    """断言请求认证。"""
    if api_key:
        return
    if (
        _has_header(headers, "authorization")
        or _has_header(headers, "x-api-key")
        or _has_header(headers, "cf-aig-authorization")
    ):
        return
    raise ValueError(f"No API key for provider: {provider}")


def _get_client_api_key(
    provider: str,
    api_key: str | None,
    headers: dict[str, str | None] | None,
) -> str:
    """获取客户端 API key。"""
    _assert_request_auth(provider, api_key, headers)
    if api_key:
        return api_key
    return "unused"


def _is_oauth_token(api_key: str) -> bool:
    """检查是否是 OAuth token。"""
    return "sk-ant-oat" in api_key


def _merge_headers(
    *header_sources: dict[str, str | None] | None,
) -> dict[str, str | None]:
    """合并请求头。"""
    merged: dict[str, str | None] = {}
    for headers in header_sources:
        if headers:
            merged.update(headers)
    return merged


def _default_supports_tool_references(model: Any) -> bool:
    """默认的 ``supportsToolReferences`` 实现。"""
    provider = getattr(model, "provider", "")
    model_id = getattr(model, "model_id", "") or getattr(model, "id", "")
    if provider != "anthropic" or "haiku" in model_id:
        return False
    import re

    version_match = re.match(
        r"^claude-(?:opus|sonnet|fable)-(\d+)(?:-(\d+))?(?:-|$)", model_id
    )
    if not version_match:
        return False
    major = int(version_match.group(1))
    minor_str = version_match.group(2)
    minor = int(minor_str) if minor_str and len(minor_str) < 8 else 0
    return major > 4 or (major == 4 and minor >= 5)


def _get_anthropic_compat(model: Any) -> dict[str, Any]:
    """获取 Anthropic 兼容性设置。"""
    compat = getattr(model, "compat", None) or {}
    if isinstance(compat, dict):
        return {
            "supports_eager_tool_input_streaming": compat.get(
                "supportsEagerToolInputStreaming", True
            ),
            "supports_long_cache_retention": compat.get(
                "supportsLongCacheRetention", True
            ),
            "send_session_affinity_headers": compat.get(
                "sendSessionAffinityHeaders", False
            ),
            "supports_cache_control_on_tools": compat.get(
                "supportsCacheControlOnTools", True
            ),
            "supports_temperature": compat.get("supportsTemperature", True),
            "allow_empty_signature": compat.get("allowEmptySignature", False),
            "supports_strict_tools": compat.get("supportsStrictTools", False),
            "supports_tool_references": compat.get(
                "supportsToolReferences", _default_supports_tool_references(model)
            ),
            "force_adaptive_thinking": compat.get("forceAdaptiveThinking", None),
        }
    return {
        "supports_eager_tool_input_streaming": True,
        "supports_long_cache_retention": True,
        "send_session_affinity_headers": False,
        "supports_cache_control_on_tools": True,
        "supports_temperature": True,
        "allow_empty_signature": False,
        "supports_strict_tools": False,
        "supports_tool_references": _default_supports_tool_references(model),
        "force_adaptive_thinking": None,
    }


def _should_use_fine_grained_tool_streaming_beta(model: Any, context: Context) -> bool:
    """检查是否需要 fine-grained tool streaming beta。"""
    tools = context.tools
    return bool(tools) and not _get_anthropic_compat(model).get(
        "supports_eager_tool_input_streaming", True
    )


def _normalize_tool_call_id(
    id: str,
    _model: Any = None,
    _msg: Any = None,
) -> str:
    """归一化 tool call ID。

    兼容 ``transform_messages`` 的签名：``Callable[[str, Model, AssistantMessage], str]``。
    """
    import re

    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "_", id)
    return sanitized[:64]


def _format_anthropic_error(error: object) -> str:
    """格式化 Anthropic 错误。"""
    return format_provider_error(normalize_provider_error(error), "Anthropic API error")


def _map_stop_reason(
    reason: str,
    stop_details: Any | None = None,
) -> dict[str, Any]:
    """映射 stop reason。"""
    error_message: str | None = None
    if stop_details is not None and hasattr(stop_details, "explanation"):
        error_message = cast(str, stop_details.explanation)

    mapped: dict[str, dict[str, Any]] = {
        "end_turn": {"stop_reason": "stop"},
        "max_tokens": {"stop_reason": "length"},
        "tool_use": {"stop_reason": "tool_use"},
        "refusal": {
            "stop_reason": "error",
            "error_message": error_message
            or "The model refused to complete the request",
        },
        "pause_turn": {"stop_reason": "stop"},
        "stop_sequence": {"stop_reason": "stop"},
        "sensitive": {
            "stop_reason": "error",
            "error_message": "Provider stopped with: sensitive",
        },
    }
    result = mapped.get(reason)
    if result is not None:
        return result
    raise ValueError(f"Unhandled stop reason: {reason}")


def _map_thinking_level_to_effort(
    model: Any,
    level: str | None,
) -> AnthropicEffort | None:
    """映射 ThinkingLevel 到 Anthropic effort 级别。"""
    if not level:
        return None
    thinking_level_map = getattr(model, "thinking_level_map", None) or {}
    mapped = thinking_level_map.get(level)
    if isinstance(mapped, str) and mapped in (
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ):
        return cast(AnthropicEffort, mapped)
    effort_map: dict[str, AnthropicEffort] = {
        "minimal": "low",
        "low": "low",
        "medium": "medium",
        "high": "high",
    }
    return effort_map.get(level, "high")


def _convert_content_blocks(
    content: list[TextContent | ImageContent],
) -> str | list[dict[str, Any]]:
    """转换内容块为 Anthropic API 格式。

    如果只有文本块，返回拼接后的字符串。
    如果有图片，返回内容块数组。
    """
    has_images = any(c.type == "image" for c in content)
    if not has_images:
        return sanitize_surrogates(
            "\n".join(c.text for c in content if isinstance(c, TextContent))
        )

    blocks: list[dict[str, Any]] = []
    for block in content:
        if block.type == "text":
            blocks.append({"type": "text", "text": sanitize_surrogates(block.text)})
        else:
            blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": block.mime_type,
                        "data": block.data,
                    },
                }
            )

    # If only images (no text), add placeholder text block
    has_text = any(b.get("type") == "text" for b in blocks)
    if not has_text:
        blocks.insert(0, {"type": "text", "text": "(see attached image)"})

    return blocks


def _convert_tool_result(
    msg: Any,
    is_oauth_token: bool,
    deferred_tool_names: set[str],
    loaded_tool_names: set[str],
    normalize_tool_name: Callable[[str], str],
) -> dict[str, Any]:
    """转换工具结果消息。"""
    references: list[dict[str, str]] = []
    for name in getattr(msg, "added_tool_names", None) or []:
        normalized_name = normalize_tool_name(name)
        if (
            normalized_name not in deferred_tool_names
            or normalized_name in loaded_tool_names
        ):
            continue
        loaded_tool_names.add(normalized_name)
        references.append(
            {
                "type": "tool_reference",
                "tool_name": _to_claude_code_name(name) if is_oauth_token else name,
            }
        )

    converted_content = _convert_content_blocks(
        cast("list[TextContent | ImageContent]", msg.content)
    )
    return {
        "tool_result": {
            "type": "tool_result",
            "tool_use_id": msg.tool_call_id,
            "content": references if references else converted_content,
            "is_error": msg.is_error,
        },
        "sibling_content": (
            []
            if not references
            else (
                [{"type": "text", "text": converted_content}]
                if isinstance(converted_content, str)
                else converted_content
            )
        ),
    }


def _convert_messages(
    transformed_messages: list[Any],
    is_oauth_token: bool,
    cache_control: dict[str, Any] | None = None,
    allow_empty_signature: bool = False,
    deferred_tool_names: set[str] | None = None,
    normalize_tool_name: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """转换消息为 Anthropic API 格式。"""
    if deferred_tool_names is None:
        deferred_tool_names = set()
    if normalize_tool_name is None:
        normalize_tool_name = lambda name: name

    params: list[dict[str, Any]] = []
    loaded_tool_names: set[str] = set()

    i = 0
    while i < len(transformed_messages):
        msg = transformed_messages[i]

        if msg.role == "user":
            content = msg.content
            if isinstance(content, str):
                if content.strip():
                    params.append(
                        {"role": "user", "content": sanitize_surrogates(content)}
                    )
            else:
                user_blocks: list[dict[str, Any]] = []
                for item in content:
                    if item.type == "text":
                        user_blocks.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    else:
                        user_blocks.append(
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": cast(str, item.mime_type),
                                    "data": item.data,
                                },
                            }
                        )
                filtered_blocks = [
                    b
                    for b in user_blocks
                    if b.get("type") != "text" or b.get("text", "").strip()
                ]
                if not filtered_blocks:
                    i += 1
                    continue
                params.append({"role": "user", "content": filtered_blocks})

        elif msg.role == "assistant":
            asst_blocks: list[dict[str, Any]] = []
            for block in msg.content:
                if block.type == "text":
                    if block.text.strip():
                        asst_blocks.append(
                            {
                                "type": "text",
                                "text": sanitize_surrogates(block.text),
                            }
                        )
                elif block.type == "thinking":
                    thinking_block = cast(ThinkingBlock, block)
                    if getattr(thinking_block, "redacted", False):
                        asst_blocks.append(
                            {
                                "type": "redacted_thinking",
                                "data": thinking_block.signature or "",
                            }
                        )
                        continue
                    thinking_signature = thinking_block.signature
                    has_thinking_signature = bool(
                        thinking_signature and thinking_signature.strip()
                    )
                    if not thinking_block.text.strip() and not has_thinking_signature:
                        continue
                    if not has_thinking_signature:
                        if allow_empty_signature:
                            asst_blocks.append(
                                {
                                    "type": "thinking",
                                    "thinking": sanitize_surrogates(
                                        thinking_block.text
                                    ),
                                    "signature": "",
                                }
                            )
                        else:
                            asst_blocks.append(
                                {
                                    "type": "text",
                                    "text": sanitize_surrogates(thinking_block.text),
                                }
                            )
                    else:
                        asst_blocks.append(
                            {
                                "type": "thinking",
                                "thinking": sanitize_surrogates(thinking_block.text),
                                "signature": thinking_signature,
                            }
                        )
                elif block.type == "toolCall":
                    tool_call = cast(ToolCallContent, block)
                    asst_blocks.append(
                        {
                            "type": "tool_use",
                            "id": tool_call.tool_call_id,
                            "name": _to_claude_code_name(tool_call.name)
                            if is_oauth_token
                            else tool_call.name,
                            "input": tool_call.args or {},
                        }
                    )
            if not asst_blocks:
                i += 1
                continue
            params.append({"role": "assistant", "content": asst_blocks})

        elif msg.role == "toolResult":
            # Collect all consecutive toolResult messages
            tool_results: list[dict[str, Any]] = []
            sibling_content: list[dict[str, Any]] = []
            j = i
            while (
                j < len(transformed_messages)
                and transformed_messages[j].role == "toolResult"
            ):
                converted = _convert_tool_result(
                    transformed_messages[j],
                    is_oauth_token,
                    deferred_tool_names,
                    loaded_tool_names,
                    normalize_tool_name,
                )
                tool_results.append(converted["tool_result"])
                sibling_content.extend(converted["sibling_content"])
                j += 1

            # Skip the messages we've already processed
            i = j - 1

            params.append(
                {
                    "role": "user",
                    "content": [*tool_results, *sibling_content],
                }
            )

        i += 1

    # Add cache_control to the last user message
    if cache_control and params:
        last_message = params[-1]
        if last_message.get("role") == "user":
            if isinstance(last_message.get("content"), list):
                last_content = last_message["content"]
                if last_content:
                    last_block = last_content[-1]
                    if isinstance(last_block, dict) and last_block.get("type") in (
                        "text",
                        "image",
                        "tool_result",
                    ):
                        last_block["cache_control"] = cache_control
            elif isinstance(last_message.get("content"), str):
                last_message["content"] = [
                    {
                        "type": "text",
                        "text": last_message["content"],
                        "cache_control": cache_control,
                    }
                ]

    return params


def _convert_tools(
    tools: list[Any],
    is_oauth_token: bool,
    supports_eager_tool_input_streaming: bool,
    supports_strict_tools: bool,
    cache_control: dict[str, Any] | None = None,
    defer_loading: bool = False,
) -> list[dict[str, Any]]:
    """转换工具为 Anthropic API 格式。"""
    if not tools:
        return []

    result: list[dict[str, Any]] = []
    for index, tool in enumerate(tools):
        strict = resolve_json_schema_strict_sampling(tool, supports_strict_tools)
        schema = tool.parameters if isinstance(tool.parameters, dict) else {}
        legacy_input_schema: dict[str, Any] = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        input_schema = (
            {**schema, **legacy_input_schema} if strict is True else legacy_input_schema
        )

        tool_entry: dict[str, Any] = {
            "name": _to_claude_code_name(tool.name) if is_oauth_token else tool.name,
            "description": tool.description,
            "input_schema": input_schema,
        }
        if supports_eager_tool_input_streaming:
            tool_entry["eager_input_streaming"] = True
        if strict is True:
            tool_entry["strict"] = True
        if defer_loading:
            tool_entry["defer_loading"] = True
        if cache_control and index == len(tools) - 1:
            tool_entry["cache_control"] = cache_control

        result.append(tool_entry)

    return result


# ---------------------------------------------------------------------------
# create_client
# ---------------------------------------------------------------------------


def _create_client(
    model: Any,
    api_key: str | None,
    interleaved_thinking: bool,
    use_fine_grained_tool_streaming_beta: bool,
    options_headers: dict[str, str | None] | None = None,
    fetch: Any = None,
    dynamic_headers: dict[str, str] | None = None,
    session_id: str | None = None,
) -> dict[str, Any]:
    """创建 Anthropic 客户端。"""
    # Adaptive thinking models have interleaved thinking built in, so skip the beta header
    compat = _get_anthropic_compat(model)
    needs_interleaved_beta = interleaved_thinking and not compat.get(
        "force_adaptive_thinking", None
    )
    beta_features: list[str] = []
    if use_fine_grained_tool_streaming_beta:
        beta_features.append(FINE_GRAINED_TOOL_STREAMING_BETA)
    if needs_interleaved_beta:
        beta_features.append(INTERLEAVED_THINKING_BETA)

    provider = getattr(model, "provider", "")
    base_url = getattr(model, "base_url", None) or ""
    model_headers = getattr(model, "headers", None) or {}

    # Copilot: Bearer auth, selective betas
    if provider == "github-copilot":
        from anthropic import AsyncAnthropic  # type: ignore[import-not-found]

        client = AsyncAnthropic(
            api_key=None,
            auth_token=api_key,
            base_url=base_url,
            dangerously_allow_browser=True,
            http_client=fetch,
            default_headers=_merge_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    **(
                        {"anthropic-beta": ",".join(beta_features)}
                        if beta_features
                        else {}
                    ),
                },
                model_headers if isinstance(model_headers, dict) else None,
                cast("dict[str, str | None] | None", dynamic_headers),
                options_headers,
            ),
        )
        return {"client": client, "is_oauth_token": False}

    # OAuth: Bearer auth, Claude Code identity headers
    if api_key and _is_oauth_token(api_key):
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(
            api_key=None,
            auth_token=api_key,
            base_url=base_url,
            dangerously_allow_browser=True,
            http_client=fetch,
            default_headers=_merge_headers(
                {
                    "accept": "application/json",
                    "anthropic-dangerous-direct-browser-access": "true",
                    "anthropic-beta": ",".join(
                        ["claude-code-20250219", "oauth-2025-04-20", *beta_features]
                    ),
                    "user-agent": f"claude-cli/{CLAUDE_CODE_VERSION}",
                    "x-app": "cli",
                },
                model_headers if isinstance(model_headers, dict) else None,
                options_headers,
            ),
        )
        return {"client": client, "is_oauth_token": True}

    # API key or header-owned auth
    session_affinity_headers: dict[str, str | None] = {}
    if session_id and compat.get("send_session_affinity_headers"):
        session_affinity_headers = {"x-session-affinity": session_id}

    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(
        api_key=api_key,
        auth_token=None,
        base_url=base_url,
        dangerously_allow_browser=True,
        http_client=fetch,
        default_headers=_merge_headers(
            {
                "accept": "application/json",
                "anthropic-dangerous-direct-browser-access": "true",
                **(
                    {"anthropic-beta": ",".join(beta_features)} if beta_features else {}
                ),
            },
            session_affinity_headers,
            model_headers if isinstance(model_headers, dict) else None,
            options_headers,
        ),
    )
    return {"client": client, "is_oauth_token": False}


# ---------------------------------------------------------------------------
# build_params
# ---------------------------------------------------------------------------


def _build_params(
    model: Any,
    context: Context,
    is_oauth_token: bool,
    options: AnthropicOptions | None = None,
) -> dict[str, Any]:
    """构建请求参数。"""
    cache_control_result = _get_cache_control(
        model,
        options.cache_retention if options else None,
        options.env if options else None,
    )
    cache_control = cache_control_result.get("cache_control")
    compat = _get_anthropic_compat(model)

    # Transform messages
    transformed_messages = transform_messages(
        context.messages, model, _normalize_tool_call_id
    )

    normalize_tool_name = (
        _to_claude_code_name if is_oauth_token else (lambda name: name)
    )
    tool_placement = split_deferred_tools(
        context, compat.get("supports_tool_references", False), normalize_tool_name
    )
    immediate_tools = list(tool_placement[0])
    deferred_tools = list(tool_placement[1].values())
    if not immediate_tools and deferred_tools:
        immediate_tools = deferred_tools
        deferred_tools = []

    deferred_tool_names = {normalize_tool_name(t.name) for t in deferred_tools}

    params: dict[str, Any] = {
        "model": getattr(model, "model_id", "") or getattr(model, "id", ""),
        "messages": _convert_messages(
            transformed_messages,
            is_oauth_token,
            cache_control,
            compat.get("allow_empty_signature", False),
            deferred_tool_names,
            normalize_tool_name,
        ),
        "max_tokens": (
            options.max_tokens
            if options and options.max_tokens
            else getattr(model, "max_tokens", 1024)
        ),
        "stream": True,
    }

    # System prompt
    if is_oauth_token:
        system_blocks: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
            }
        ]
        if cache_control:
            system_blocks[0]["cache_control"] = cache_control
        if context.system_prompt:
            sys_block: dict[str, Any] = {
                "type": "text",
                "text": sanitize_surrogates(context.system_prompt),
            }
            if cache_control:
                sys_block["cache_control"] = cache_control
            system_blocks.append(sys_block)
        params["system"] = system_blocks
    elif context.system_prompt:
        sys_block2: dict[str, Any] = {
            "type": "text",
            "text": sanitize_surrogates(context.system_prompt),
        }
        if cache_control:
            sys_block2["cache_control"] = cache_control
        params["system"] = [sys_block2]

    # Temperature
    if (
        options
        and options.temperature is not None
        and not options.thinking_enabled
        and compat.get("supports_temperature", True)
    ):
        params["temperature"] = options.temperature

    # Tools
    if immediate_tools or deferred_tools:
        params["tools"] = [
            *_convert_tools(
                immediate_tools,
                is_oauth_token,
                compat.get("supports_eager_tool_input_streaming", True),
                compat.get("supports_strict_tools", False),
                cache_control
                if compat.get("supports_cache_control_on_tools", True)
                else None,
            ),
            *_convert_tools(
                deferred_tools,
                is_oauth_token,
                compat.get("supports_eager_tool_input_streaming", True),
                compat.get("supports_strict_tools", False),
                None,
                True,
            ),
        ]

    # Thinking mode
    if getattr(model, "reasoning", False):
        if options and options.thinking_enabled:
            display: AnthropicThinkingDisplay = options.thinking_display or "summarized"
            if compat.get("force_adaptive_thinking"):
                params["thinking"] = {"type": "adaptive", "display": display}
                if options.effort:
                    params["output_config"] = (
                        {"effort": options.effort}
                        if options.effort != "xhigh"
                        else cast(dict[str, Any], {"effort": options.effort})
                    )
            else:
                params["thinking"] = {
                    "type": "enabled",
                    "budget_tokens": options.thinking_budget_tokens or 1024,
                    "display": display,
                }
        elif options and options.thinking_enabled is False:
            thinking_level_map = getattr(model, "thinking_level_map", None) or {}
            if thinking_level_map.get("off") is not None:
                params["thinking"] = {"type": "disabled"}

    # Metadata
    if options and options.metadata:
        user_id = options.metadata.get("user_id")
        if isinstance(user_id, str):
            params["metadata"] = {"user_id": user_id}

    # Tool choice
    if options and options.tool_choice is not None:
        if isinstance(options.tool_choice, str):
            params["tool_choice"] = {"type": options.tool_choice}
        else:
            params["tool_choice"] = options.tool_choice

    return params


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: AnthropicOptions | None = None,
) -> AssistantMessageEventStream:
    """Anthropic Messages API 流式生成函数。"""
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
            client: Any
            is_oauth: bool

            if options and options.client:
                client = options.client
                is_oauth = False
            else:
                api_key = options.api_key if options else None
                _assert_request_auth(
                    provider, api_key, options.headers if options else None
                )

                copilot_dynamic_headers: dict[str, str] | None = None
                if provider == "github-copilot":
                    has_images = has_copilot_vision_input(context.messages)
                    copilot_dynamic_headers = build_copilot_dynamic_headers(
                        {
                            "messages": context.messages,
                            "has_images": has_images,
                        }
                    )

                cache_retention = _resolve_cache_retention(
                    options.cache_retention if options else None,
                    options.env if options else None,
                )
                cache_session_id = (
                    options.session_id
                    if options and cache_retention != "none"
                    else None
                )

                created = _create_client(
                    model,
                    api_key,
                    options.interleaved_thinking
                    if (options and options.interleaved_thinking is not None)
                    else True,
                    _should_use_fine_grained_tool_streaming_beta(model, context),
                    options.headers if options else None,
                    options.fetch if options else None,
                    copilot_dynamic_headers,
                    cache_session_id,
                )
                client = created["client"]
                is_oauth = created["is_oauth_token"]

            params = _build_params(model, context, is_oauth, options)

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
            if options and options.signal:
                request_options["signal"] = options.signal

            # 发起流式请求（带重试）
            stream_response = await retry_provider_request(
                lambda: client.messages.create(**params, **request_options),
                ProviderRetryOptions(
                    max_retries=(options.max_retries or 0) if options else 0,
                    max_retry_delay_ms=options.max_retry_delay_ms if options else None,
                    signal=options.signal if options else None,
                ),
            )

            # 获取响应头
            response_headers: dict[str, str] = {}
            response_status = 0
            raw_response = getattr(stream_response, "_response", None)
            if raw_response is not None:
                response_status = getattr(raw_response, "status_code", 0)
                response_headers = dict(getattr(raw_response, "headers", {}))

            # onResponse 回调
            if options and options.on_response:
                await options.on_response(
                    {
                        "status": response_status,
                        "headers": response_headers,
                    },
                    model,
                )

            event_stream.push(AssistantMessageSnapshot(message=output))

            # Track blocks by their Anthropic index
            content_blocks_by_index: dict[int, Any] = {}
            thinking_blocks_by_index: dict[int, Any] = {}

            # 处理流式事件
            stream_iter = cast(Any, stream_response)
            async for event in stream_iter:
                event_type = event.type if hasattr(event, "type") else event.get("type")

                if event_type == "message_start":
                    message = (
                        event.message
                        if hasattr(event, "message")
                        else event.get("message", {})
                    )
                    message_id = getattr(message, "id", None) or message.get("id")
                    if message_id:
                        object.__setattr__(output, "response_id", message_id)
                    msg_usage = getattr(message, "usage", None) or message.get(
                        "usage", {}
                    )
                    usage.input = getattr(
                        msg_usage, "input_tokens", 0
                    ) or msg_usage.get("input_tokens", 0)
                    usage.output = getattr(
                        msg_usage, "output_tokens", 0
                    ) or msg_usage.get("output_tokens", 0)
                    usage.cache_read = getattr(
                        msg_usage, "cache_read_input_tokens", 0
                    ) or msg_usage.get("cache_read_input_tokens", 0)
                    usage.cache_write = getattr(
                        msg_usage, "cache_creation_input_tokens", 0
                    ) or msg_usage.get("cache_creation_input_tokens", 0)
                    cache_creation = getattr(
                        msg_usage, "cache_creation", None
                    ) or msg_usage.get("cache_creation", {})
                    if cache_creation:
                        val = (
                            cache_creation.get("ephemeral_1h_input_tokens", 0)
                            if isinstance(cache_creation, dict)
                            else getattr(cache_creation, "ephemeral_1h_input_tokens", 0)
                        )
                        object.__setattr__(usage, "cache_write_1h", val)
                    usage.total_tokens = (
                        usage.input
                        + usage.output
                        + usage.cache_read
                        + usage.cache_write
                    )
                    calculate_cost(model, usage)

                elif event_type == "content_block_start":
                    content_block = (
                        event.content_block
                        if hasattr(event, "content_block")
                        else event.get("content_block", {})
                    )
                    block_index = (
                        event.index
                        if hasattr(event, "index")
                        else event.get("index", 0)
                    )
                    block_type = getattr(
                        content_block, "type", None
                    ) or content_block.get("type")

                    if block_type == "text":
                        text = getattr(
                            content_block, "text", None
                        ) or content_block.get("text", "")
                        text_block = TextContent(text=text or "")
                        output.content.append(text_block)
                        content_blocks_by_index[block_index] = {
                            "type": "text",
                            "block": text_block,
                            "content_index": len(output.content) - 1,
                        }
                        event_stream.push(AssistantMessageSnapshot(message=output))

                    elif block_type == "thinking":
                        thinking = getattr(
                            content_block, "thinking", None
                        ) or content_block.get("thinking", "")
                        signature = getattr(
                            content_block, "signature", None
                        ) or content_block.get("signature", "")
                        think_block = ThinkingBlock(
                            text=thinking or "", signature=signature or ""
                        )
                        output.thinking = (output.thinking or []) + [think_block]
                        thinking_blocks_by_index[block_index] = {
                            "type": "thinking",
                            "block": think_block,
                            "content_index": len(output.thinking) - 1,
                        }
                        event_stream.push(AssistantMessageSnapshot(message=output))

                    elif block_type == "redacted_thinking":
                        data = getattr(
                            content_block, "data", None
                        ) or content_block.get("data", "")
                        think_block = ThinkingBlock(
                            text="[Reasoning redacted]",
                            signature=data,
                        )
                        # Set redacted flag using object.__setattr__
                        object.__setattr__(think_block, "redacted", True)
                        output.thinking = (output.thinking or []) + [think_block]
                        thinking_blocks_by_index[block_index] = {
                            "type": "thinking",
                            "block": think_block,
                            "content_index": len(output.thinking) - 1,
                        }
                        event_stream.push(AssistantMessageSnapshot(message=output))

                    elif block_type == "tool_use":
                        tool_id = getattr(
                            content_block, "id", None
                        ) or content_block.get("id", "")
                        tool_name = getattr(
                            content_block, "name", None
                        ) or content_block.get("name", "")
                        tool_input = getattr(
                            content_block, "input", None
                        ) or content_block.get("input", {})
                        resolved_name = (
                            _from_claude_code_name(tool_name, context.tools)
                            if is_oauth
                            else tool_name
                        )
                        tc_block = _StreamingToolCall(
                            tool_call_id=tool_id,
                            name=resolved_name,
                            args=tool_input if isinstance(tool_input, dict) else {},
                            partial_json="",
                        )
                        output.content.append(
                            ToolCallContent(
                                tool_call_id=tc_block.tool_call_id,
                                name=tc_block.name,
                                args=tc_block.args,
                            )
                        )
                        content_blocks_by_index[block_index] = {
                            "type": "toolCall",
                            "block": tc_block,
                            "content_index": len(output.content) - 1,
                        }
                        event_stream.push(
                            AssistantToolCallStart(
                                tool_call_id=tool_id,
                                name=resolved_name,
                            )
                        )

                elif event_type == "content_block_delta":
                    delta = (
                        event.delta
                        if hasattr(event, "delta")
                        else event.get("delta", {})
                    )
                    block_index = (
                        event.index
                        if hasattr(event, "index")
                        else event.get("index", 0)
                    )
                    delta_type = getattr(delta, "type", None) or delta.get("type")

                    if delta_type == "text_delta":
                        delta_text = getattr(delta, "text", None) or delta.get(
                            "text", ""
                        )
                        block_info = content_blocks_by_index.get(block_index)
                        if block_info and block_info["type"] == "text":
                            block_info["block"].text += delta_text
                            event_stream.push(AssistantTextDelta(delta=delta_text))

                    elif delta_type == "thinking_delta":
                        delta_thinking = getattr(delta, "thinking", None) or delta.get(
                            "thinking", ""
                        )
                        block_info = thinking_blocks_by_index.get(block_index)
                        if block_info and block_info["type"] == "thinking":
                            block_info["block"].text += delta_thinking
                            event_stream.push(
                                AssistantThinkingDelta(delta=delta_thinking)
                            )

                    elif delta_type == "input_json_delta":
                        delta_json = getattr(delta, "partial_json", None) or delta.get(
                            "partial_json", ""
                        )
                        block_info = content_blocks_by_index.get(block_index)
                        if block_info and block_info["type"] == "toolCall":
                            tc_block = block_info["block"]
                            tc_block.partial_json += delta_json
                            tc_block.args = parse_streaming_json(tc_block.partial_json)
                            event_stream.push(
                                AssistantToolCallUpdate(
                                    tool_call_id=tc_block.tool_call_id,
                                    args=tc_block.args,
                                )
                            )

                    elif delta_type == "signature_delta":
                        delta_signature = getattr(
                            delta, "signature", None
                        ) or delta.get("signature", "")
                        block_info = thinking_blocks_by_index.get(block_index)
                        if block_info and block_info["type"] == "thinking":
                            think_block = block_info["block"]
                            if not think_block.signature:
                                think_block.signature = ""
                            think_block.signature += delta_signature

                elif event_type == "content_block_stop":
                    block_index = (
                        event.index
                        if hasattr(event, "index")
                        else event.get("index", 0)
                    )

                    # Check text/toolCall blocks
                    block_info = content_blocks_by_index.pop(block_index, None)
                    if block_info:
                        if block_info["type"] == "text":
                            event_stream.push(AssistantMessageSnapshot(message=output))
                        elif block_info["type"] == "toolCall":
                            tc_block = block_info["block"]
                            tc_block.args = parse_streaming_json(tc_block.partial_json)
                            tc_block.partial_json = None  # Clean up scratch buffer
                            event_stream.push(
                                AssistantToolCallEnd(
                                    tool_call_id=tc_block.tool_call_id,
                                    content=[
                                        ToolCallContent(
                                            type="toolCall",
                                            tool_call_id=tc_block.tool_call_id,
                                            name=tc_block.name,
                                            args=tc_block.args,
                                        )
                                    ],
                                )
                            )

                    # Check thinking blocks
                    thinking_info = thinking_blocks_by_index.pop(block_index, None)
                    if thinking_info:
                        event_stream.push(AssistantMessageSnapshot(message=output))

                elif event_type == "message_delta":
                    msg_delta = (
                        event.delta
                        if hasattr(event, "delta")
                        else event.get("delta", {})
                    )
                    msg_usage = (
                        event.usage if hasattr(event, "usage") else event.get("usage")
                    )

                    # Stop reason
                    stop_reason = getattr(
                        msg_delta, "stop_reason", None
                    ) or msg_delta.get("stop_reason")
                    if stop_reason:
                        object.__setattr__(output, "raw_stop_reason", stop_reason)
                        stop_details = getattr(
                            msg_delta, "stop_details", None
                        ) or msg_delta.get("stop_details")
                        mapped = _map_stop_reason(stop_reason, stop_details)
                        output.stop_reason = mapped.get("stop_reason", "stop")
                        if mapped.get("error_message"):
                            output.error_message = mapped["error_message"]

                    # Usage
                    if msg_usage:
                        input_tokens = getattr(
                            msg_usage, "input_tokens", None
                        ) or msg_usage.get("input_tokens")
                        if input_tokens is not None:
                            usage.input = input_tokens
                        output_tokens = getattr(
                            msg_usage, "output_tokens", None
                        ) or msg_usage.get("output_tokens")
                        if output_tokens is not None:
                            usage.output = output_tokens
                        cache_read = getattr(
                            msg_usage, "cache_read_input_tokens", None
                        ) or msg_usage.get("cache_read_input_tokens")
                        if cache_read is not None:
                            usage.cache_read = cache_read
                        cache_write = getattr(
                            msg_usage, "cache_creation_input_tokens", None
                        ) or msg_usage.get("cache_creation_input_tokens")
                        if cache_write is not None:
                            usage.cache_write = cache_write

                        # Reasoning tokens from output_tokens_details
                        output_tokens_details = None
                        if hasattr(msg_usage, "output_tokens_details"):
                            output_tokens_details = msg_usage.output_tokens_details
                        elif isinstance(msg_usage, dict):
                            output_tokens_details = msg_usage.get(
                                "output_tokens_details"
                            )
                        if output_tokens_details:
                            thinking_tokens = None
                            if hasattr(output_tokens_details, "thinking_tokens"):
                                thinking_tokens = output_tokens_details.thinking_tokens
                            elif isinstance(output_tokens_details, dict):
                                thinking_tokens = output_tokens_details.get(
                                    "thinking_tokens"
                                )
                            if thinking_tokens is not None:
                                object.__setattr__(usage, "reasoning", thinking_tokens)

                    usage.total_tokens = (
                        usage.input
                        + usage.output
                        + usage.cache_read
                        + usage.cache_write
                    )
                    calculate_cost(model, usage)

            # 检查中止
            if options and options.signal and getattr(options.signal, "aborted", False):
                raise RuntimeError("Request was aborted")

            if output.stop_reason == "pending":
                raise RuntimeError("Anthropic stream ended without a stop reason")
            if output.stop_reason in ("aborted", "error"):
                raise RuntimeError(output.error_message or "An unknown error occurred")

            event_stream.push(
                AssistantStreamEnd(reason=output.stop_reason, message=output)
            )
            event_stream.end()

        except Exception as error:
            # Clean up temporary fields
            for block in output.content:
                if hasattr(block, "partial_json"):
                    try:
                        delattr(block, "partial_json")
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
            output.error_message = _format_anthropic_error(error)
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
    """简化的流式接口。"""
    _assert_request_auth(
        getattr(model, "provider", ""),
        options.api_key if options else None,
        options.headers if options else None,
    )

    base = build_base_options(
        model, context, options, options.api_key if options else None
    )

    if not options or not options.reasoning:
        return stream(
            model,
            context,
            AnthropicOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                thinking_enabled=False,
            ),
        )

    # For models with adaptive thinking: use an effort level
    compat = _get_anthropic_compat(model)
    if compat.get("force_adaptive_thinking"):
        effort = _map_thinking_level_to_effort(model, options.reasoning)
        return stream(
            model,
            context,
            AnthropicOptions(
                **base.model_dump() if hasattr(base, "model_dump") else dict(base),
                thinking_enabled=True,
                effort=effort,
            ),
        )

    # Undefined means the caller did not request an output cap; let the helper use the model cap
    model_max_tokens = getattr(model, "max_tokens", 0)
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
        AnthropicOptions(
            **base.model_dump() if hasattr(base, "model_dump") else dict(base),
            max_tokens=max_tokens,
            thinking_enabled=True,
            thinking_budget_tokens=min(adjusted[1], max(0, max_tokens - 1024)),
        ),
    )


# ---------------------------------------------------------------------------
# 内部辅助类型
# ---------------------------------------------------------------------------


class _StreamingToolCall(BaseModel):
    """流式工具调用（内部使用，用于跟踪 partial_json）。"""

    type: Literal["toolCall"] = "toolCall"
    tool_call_id: str
    name: str
    args: dict[str, object]
    partial_json: str | None = None
