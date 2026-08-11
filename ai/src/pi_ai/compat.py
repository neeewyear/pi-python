"""兼容层（对应 ``compat.ts`` + ``compat/extension-oauth-types.ts``）。

保留旧版 pi-ai API 表面：API 分发 ``stream()``/``complete()`` 与 env API Key 注入、
API 注册表、已生成目录读取（``getModel``/``getModels``/``getProviders``）、
per-API 延迟流式包装器和图片生成。

新代码应使用 ``create_models()`` 和 provider 工厂。
"""

from __future__ import annotations

import random
from typing import Any, Protocol, cast

from .env_api_keys import get_env_api_key
from .providers.faux import _create_faux_core
from .providers.registry import builtin_models
from .types import (
    AssistantMessage,
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from .utils.event_stream import AssistantMessageEventStream

# 函数签名类型别名
StreamFunction = Any

# ---------------------------------------------------------------------------
# Extension OAuth 类型（对应 ``compat/extension-oauth-types.ts``）
# ---------------------------------------------------------------------------


class OAuthPrompt:
    """遗留扩展 OAuth 提示。"""

    message: str = ""
    placeholder: str | None = None
    allow_empty: bool = False


class OAuthAuthInfo:
    """遗留扩展 OAuth 授权链接。"""

    url: str = ""
    instructions: str | None = None


class OAuthDeviceCodeInfo:
    """遗留扩展 OAuth 设备码通知。"""

    user_code: str = ""
    verification_uri: str = ""
    interval_seconds: int | None = None
    expires_in_seconds: int | None = None


class OAuthSelectOption:
    """OAuth 选择选项。"""

    id: str = ""
    label: str = ""


class OAuthSelectPrompt:
    """OAuth 选择提示。"""

    message: str = ""
    options: list[OAuthSelectOption] = []


class OAuthLoginCallbacks(Protocol):
    """遗留扩展 OAuth 登录回调表面。"""

    def on_auth(self, info: OAuthAuthInfo) -> None: ...
    def on_device_code(self, info: OAuthDeviceCodeInfo) -> None: ...
    async def on_prompt(self, prompt: OAuthPrompt) -> str: ...
    def on_progress(self, message: str) -> None: ...
    async def on_manual_code_input(self) -> str: ...
    async def on_select(self, prompt: OAuthSelectPrompt) -> str | None: ...

    signal: Any  # AbortSignal


# ---------------------------------------------------------------------------
# API Provider 注册表
# ---------------------------------------------------------------------------


ApiStreamFunction = Any
ApiStreamSimpleFunction = Any


class _ApiProviderInternal:
    """内部 API provider。"""

    api: str
    stream: ApiStreamFunction
    stream_simple: ApiStreamSimpleFunction

    def __init__(self, api: str, stream: Any, stream_simple: Any) -> None:
        self.api = api
        self.stream = stream
        self.stream_simple = stream_simple


class _RegisteredApiProvider:
    """已注册的 API provider。"""

    provider: _ApiProviderInternal
    source_id: str | None = None

    def __init__(
        self, provider: _ApiProviderInternal, source_id: str | None = None
    ) -> None:
        self.provider = provider
        self.source_id = source_id


_api_provider_registry: dict[str, _RegisteredApiProvider] = {}


def _wrap_stream(api: str, stream: StreamFunction) -> Any:
    """包装 stream 函数以验证 API 匹配。"""

    def _wrapped(
        model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return cast(
            AssistantMessageEventStream, stream(cast(Any, model), context, options)
        )

    return _wrapped


def _wrap_stream_simple(api: str, stream_simple: StreamFunction) -> Any:
    """包装 stream_simple 函数以验证 API 匹配。"""

    def _wrapped(
        model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return cast(
            AssistantMessageEventStream,
            stream_simple(cast(Any, model), context, options),
        )

    return _wrapped


def register_api_provider(
    api: str,
    stream: StreamFunction,
    stream_simple: StreamFunction,
    source_id: str | None = None,
) -> None:
    """注册 API provider。"""
    _api_provider_registry[api] = _RegisteredApiProvider(
        provider=_ApiProviderInternal(
            api,
            _wrap_stream(api, stream),
            _wrap_stream_simple(api, stream_simple),
        ),
        source_id=source_id,
    )


def get_api_provider(api: str) -> _ApiProviderInternal | None:
    """获取已注册的 API provider。"""
    entry = _api_provider_registry.get(api)
    return entry.provider if entry else None


def get_api_providers() -> list[_ApiProviderInternal]:
    """获取所有已注册的 API provider。"""
    return [entry.provider for entry in _api_provider_registry.values()]


def unregister_api_providers(source_id: str) -> None:
    """取消注册指定 source 的所有 API provider。"""
    to_delete = [
        api
        for api, entry in _api_provider_registry.items()
        if entry.source_id == source_id
    ]
    for api in to_delete:
        _api_provider_registry.pop(api, None)


def _clear_api_providers() -> None:
    """清除所有 API provider。"""
    _api_provider_registry.clear()


def register_faux_provider(options: dict[str, Any] | None = None) -> Any:
    """注册假 provider（用于测试）。"""
    core = _create_faux_core(**(options or {}))
    source_id = f"faux-provider-{random.randint(0, 2**31):08x}"
    register_api_provider(core["api"], core["stream"], core["stream_simple"], source_id)

    class _FauxHandle:
        """Faux provider 句柄。"""

        api = core["api"]
        models = core["models"]
        get_model = core["get_model"]
        state = core["state"]
        set_responses = core["set_responses"]
        append_responses = core["append_responses"]
        get_pending_response_count = core["get_pending_response_count"]

        def unregister(self) -> None:
            unregister_api_providers(source_id)

    return _FauxHandle()


# ---------------------------------------------------------------------------
# 内置 API 注册
# ---------------------------------------------------------------------------

from .api.anthropic_messages_lazy import (
    anthropic_messages_api as _anthropic_messages_api,
)
from .api.azure_openai_responses_lazy import (
    azure_openai_responses_api as _azure_openai_responses_api,
)
from .api.bedrock_converse_stream_lazy import (
    bedrock_converse_stream_api as _bedrock_converse_stream_api,
)
from .api.google_generative_ai_lazy import (
    google_generative_ai_api as _google_generative_ai_api,
)
from .api.google_vertex_lazy import google_vertex_api as _google_vertex_api
from .api.mistral_conversations_lazy import (
    mistral_conversations_api as _mistral_conversations_api,
)
from .api.openai_codex_responses_lazy import (
    openai_codex_responses_api as _openai_codex_responses_api,
)
from .api.openai_completions_lazy import (
    openai_completions_api as _openai_completions_api,
)
from .api.openai_responses_lazy import openai_responses_api as _openai_responses_api
from .api.pi_messages_lazy import pi_messages_api as _pi_messages_api

BUILTIN_APIS: list[tuple[str, Any]] = [
    ("anthropic-messages", _anthropic_messages_api()),
    ("openai-completions", _openai_completions_api()),
    ("openai-responses", _openai_responses_api()),
    ("openai-codex-responses", _openai_codex_responses_api()),
    ("azure-openai-responses", _azure_openai_responses_api()),
    ("google-generative-ai", _google_generative_ai_api()),
    ("google-vertex", _google_vertex_api()),
    ("mistral-conversations", _mistral_conversations_api()),
    ("bedrock-converse-stream", _bedrock_converse_stream_api()),
    ("pi-messages", _pi_messages_api()),
]

_builtin_api_provider_instances: dict[str, _ApiProviderInternal | None] = {}


def register_builtin_api_providers() -> None:
    """注册内置 API 实现到注册表，不覆盖已有条目。"""
    for api, streams in BUILTIN_APIS:
        if not get_api_provider(api):
            register_api_provider(api, streams.stream, streams.stream_simple)
        _builtin_api_provider_instances[api] = get_api_provider(api)


def reset_api_providers() -> None:
    """重置所有 API provider。"""
    _clear_api_providers()
    _builtin_api_provider_instances.clear()
    register_builtin_api_providers()


register_builtin_api_providers()

# ---------------------------------------------------------------------------
# 兼容 API 分发
# ---------------------------------------------------------------------------

_compat_models = builtin_models()
_AMBIENT_AUTH_MARKER = "<authenticated>"


def _has_explicit_api_key(api_key: str | None) -> bool:
    """检查是否存在显式 API Key。"""
    return isinstance(api_key, str) and api_key.strip() != ""


def _with_env_api_key(model: Model, options: Any) -> Any:
    """注入环境 API Key 到选项。"""
    if options and _has_explicit_api_key(getattr(options, "api_key", None)):
        return options
    api_key = get_env_api_key(
        model.provider, getattr(options, "env", None) if options else None
    )
    if not api_key or api_key == _AMBIENT_AUTH_MARKER:
        return options
    merged = dict(getattr(options, "__dict__", options) if options else {})
    merged["api_key"] = api_key
    return merged


def _has_resolved_cloudflare_auth(options: Any) -> bool:
    """检查是否已解析 Cloudflare 认证。"""
    return _has_explicit_api_key(
        getattr(options, "api_key", None) if options else None
    ) or (
        bool(getattr(options, "headers", None) if options else None)
        and "cf-aig-authorization" in options.headers
    )


def _get_builtin_provider_for_model(model: Model) -> Any:
    """获取模型的内置 provider（如果已注册且匹配）。"""
    if get_api_provider(model.api) is not _builtin_api_provider_instances.get(
        model.api
    ):
        return None
    provider = _compat_models.get_provider(model.provider)
    if provider and any(c.api == model.api for c in provider.get_models()):
        return provider
    return None


def _resolve_api_provider(api: str) -> _ApiProviderInternal:
    """解析 API provider。"""
    provider = get_api_provider(api)
    if not provider:
        raise ValueError(f"No API provider registered for api: {api}")
    return provider


def stream(
    model: Model,
    context: Context,
    options: Any = None,
) -> AssistantMessageEventStream:
    """兼容 stream 函数。"""
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider:
        if model.provider.startswith(
            "cloudflare-"
        ) and not _has_resolved_cloudflare_auth(options):
            return _compat_models.stream(model, context, options)
        return cast(
            AssistantMessageEventStream,
            builtin_provider.stream(model, context, _with_env_api_key(model, options)),
        )
    provider = _resolve_api_provider(model.api)
    return cast(
        AssistantMessageEventStream,
        provider.stream(model, context, _with_env_api_key(model, options)),
    )


async def complete(
    model: Model,
    context: Context,
    options: Any = None,
) -> AssistantMessage:
    """兼容 complete 函数。"""
    s = stream(model, context, options)
    return await s.result()


def stream_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessageEventStream:
    """兼容 stream_simple 函数。"""
    builtin_provider = _get_builtin_provider_for_model(model)
    if builtin_provider:
        if model.provider.startswith(
            "cloudflare-"
        ) and not _has_resolved_cloudflare_auth(options):
            return _compat_models.stream_simple(model, context, options)
        return cast(
            AssistantMessageEventStream,
            builtin_provider.stream_simple(
                model, context, _with_env_api_key(model, options)
            ),
        )
    provider = _resolve_api_provider(model.api)
    return cast(
        AssistantMessageEventStream,
        provider.stream_simple(model, context, _with_env_api_key(model, options)),
    )


async def complete_simple(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
) -> AssistantMessage:
    """兼容 complete_simple 函数。"""
    s = stream_simple(model, context, options)
    return await s.result()


# ---------------------------------------------------------------------------
# 已废弃的别名
# ---------------------------------------------------------------------------


def get_model(provider: str, id: str) -> Any:
    """已废弃。使用 ``Models.get_model()``。"""
    return _compat_models.get_model(provider, id)


def get_models() -> list[Any]:
    """已废弃。使用 ``Models.get_models()``。"""
    return _compat_models.get_models()


def get_providers() -> list[Any]:
    """已废弃。使用 ``Models.get_providers()``。"""
    return _compat_models.get_providers()


__all__ = [
    # OAuth types
    "OAuthPrompt",
    "OAuthAuthInfo",
    "OAuthDeviceCodeInfo",
    "OAuthSelectOption",
    "OAuthSelectPrompt",
    "OAuthLoginCallbacks",
    # API registry
    "register_api_provider",
    "get_api_provider",
    "get_api_providers",
    "unregister_api_providers",
    "register_faux_provider",
    "register_builtin_api_providers",
    "reset_api_providers",
    # Compat dispatch
    "stream",
    "complete",
    "stream_simple",
    "complete_simple",
    # Deprecated aliases
    "get_model",
    "get_models",
    "get_providers",
]
