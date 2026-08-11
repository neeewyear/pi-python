"""Provider 组合器（对应 TS ``provider-composer.ts``）。

将内置 provider、``models.json`` 配置和扩展 provider 三层组合为一个
统一的 ``Provider`` 对象，处理认证、模型列表和流式请求。
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, cast

from pi_ai.auth import (
    AuthContext,
    AuthResult,
    ModelAuth,
    OAuthCredentials,
)
from pi_ai.compat import (
    get_api_provider,
)
from pi_ai.models import (
    Provider,
    RefreshModelsContext,
    lazy_stream,
)
from pi_ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream

from .resolve_config_value import (
    get_config_value_env_var_names,
    is_command_config_value,
    is_config_value_configured,
    resolve_config_value_or_throw,
    resolve_headers_or_throw,
)

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class ExtensionOAuthConfig:
    """扩展 OAuth 配置。"""

    def __init__(
        self,
        name: str,
        login: Callable[..., Any],
        refresh_token: Callable[..., Any],
        get_api_key: Callable[..., str],
        modify_models: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.login = login
        self.refresh_token = refresh_token
        self.get_api_key = get_api_key
        self.modify_models = modify_models


class ProviderConfigInput:
    """扩展注册 provider 的输入类型。"""

    def __init__(
        self,
        name: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        api: str | None = None,
        stream_simple: Callable[..., Any] | None = None,
        headers: dict[str, str] | None = None,
        auth_header: bool | None = None,
        oauth: ExtensionOAuthConfig | None = None,
        models: list[dict[str, Any]] | None = None,
        refresh_models: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self.base_url = base_url
        self.api_key = api_key
        self.api = api
        self.stream_simple = stream_simple
        self.headers = headers
        self.auth_header = auth_header
        self.oauth = oauth
        self.models = models
        self.refresh_models = refresh_models


class AuthStatus:
    """认证状态。"""

    def __init__(
        self,
        configured: bool,
        source: str | None = None,
        label: str | None = None,
    ) -> None:
        self.configured = configured
        self.source = source
        self.label = label


class CompatibilityRequestConfig:
    """兼容性请求配置。"""

    def __init__(
        self,
        headers: dict[str, str] | None = None,
        auth_header: bool = False,
    ) -> None:
        self.headers = headers
        self.auth_header = auth_header


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _merge_compat(
    base: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """合并 compat 配置。"""
    if not override:
        return base
    merged = {**(base or {}), **override}
    # 深度合并嵌套对象
    for key in (
        "openRouterRouting",
        "vercelGatewayRouting",
        "chatTemplateKwargs",
        "chatTemplateArgs",
    ):
        base_val = (base or {}).get(key)
        override_val = override.get(key)
        if isinstance(base_val, dict) or isinstance(override_val, dict):
            merged[key] = {**(base_val or {}), **(override_val or {})}
    return merged


def _apply_model_override(
    model: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    """应用模型覆盖。"""
    result = dict(model)
    if "name" in override and override["name"] is not None:
        result["name"] = override["name"]
    if "reasoning" in override and override["reasoning"] is not None:
        result["reasoning"] = override["reasoning"]
    if override.get("thinkingLevelMap"):
        result["thinkingLevelMap"] = {
            **(model.get("thinkingLevelMap") or {}),
            **override["thinkingLevelMap"],
        }
    if "input" in override and override["input"] is not None:
        result["input"] = override["input"]
    if override.get("cost"):
        cost = dict(model.get("cost") or {})
        for key in ("input", "output", "cacheRead", "cacheWrite"):
            if key in override["cost"]:
                cost[key] = override["cost"][key]
        if "tiers" in override["cost"]:
            cost["tiers"] = override["cost"]["tiers"]
        result["cost"] = cost
    if "contextWindow" in override and override["contextWindow"] is not None:
        result["contextWindow"] = override["contextWindow"]
    if "maxTokens" in override and override["maxTokens"] is not None:
        result["maxTokens"] = override["maxTokens"]
    if override.get("samplingParams"):
        result["samplingParams"] = {
            **(model.get("samplingParams") or {}),
            **override["samplingParams"],
        }
    compat = _merge_compat(model.get("compat"), override.get("compat"))
    if compat is not None:
        result["compat"] = compat
    return result


def _model_from_json(
    provider_id: str,
    definition: dict[str, Any],
    provider_config: dict[str, Any],
    defaults: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """从 models.json 定义创建模型。"""
    api = (
        definition.get("api")
        or provider_config.get("api")
        or (defaults or {}).get("api")
    )
    if not api:
        raise ValueError(
            f"Provider {provider_id}, model {definition['id']}: "
            'no "api" specified. Set at provider or model level.'
        )
    base_url = (
        definition.get("baseUrl")
        or provider_config.get("baseUrl")
        or (defaults or {}).get("baseUrl")
    )
    if not base_url:
        raise ValueError(
            f'Provider {provider_id}: "baseUrl" is required when defining custom models.'
        )
    context_window = definition.get("contextWindow")
    if context_window is not None and context_window <= 0:
        raise ValueError(
            f"Provider {provider_id}, model {definition['id']}: invalid contextWindow"
        )
    max_tokens = definition.get("maxTokens")
    if max_tokens is not None and max_tokens <= 0:
        raise ValueError(
            f"Provider {provider_id}, model {definition['id']}: invalid maxTokens"
        )

    return {
        "id": definition["id"],
        "name": definition.get("name") or definition["id"],
        "api": api,
        "provider": provider_id,
        "baseUrl": base_url,
        "reasoning": definition.get("reasoning", False),
        "thinkingLevelMap": definition.get("thinkingLevelMap"),
        "input": definition.get("input", ["text"]),
        "cost": definition.get(
            "cost", {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
        ),
        "contextWindow": context_window or 128000,
        "maxTokens": max_tokens or 16384,
        "samplingParams": definition.get("samplingParams"),
        "headers": None,
        "compat": _merge_compat(
            provider_config.get("compat"), definition.get("compat")
        ),
    }


def _apply_models_json(
    provider_id: str,
    base_models: list[dict[str, Any]],
    config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """应用 models.json 配置到模型列表。"""
    if not config:
        return list(base_models)
    if config.get("oauth") and not config.get("baseUrl"):
        raise ValueError(
            f'Provider {provider_id}: "baseUrl" is required when "oauth" is set.'
        )

    has_overrides = config.get("modelOverrides") and len(config["modelOverrides"]) > 0
    if (
        not config.get("models")
        and not config.get("baseUrl")
        and not config.get("headers")
        and not config.get("compat")
        and not has_overrides
        and not config.get("apiKey")
        and not config.get("oauth")
        and config.get("authHeader") is None
    ):
        raise ValueError(
            f'Provider {provider_id}: must specify "baseUrl", "headers", "compat", '
            '"modelOverrides", or "models".'
        )

    models = [
        {
            **model,
            "baseUrl": config.get("baseUrl", model.get("baseUrl")),
            "compat": _merge_compat(model.get("compat"), config.get("compat")),
        }
        for model in base_models
    ]

    for definition in config.get("models") or []:
        existing_index = next(
            (i for i, m in enumerate(models) if m["id"] == definition["id"]),
            -1,
        )
        defaults = (
            models[existing_index]
            if existing_index >= 0
            else (models[0] if models else None)
        )
        model = _model_from_json(provider_id, definition, config, defaults)
        if existing_index >= 0:
            models[existing_index] = model
        else:
            models.append(model)

    return models


def _apply_extension(
    provider_id: str,
    models: list[dict[str, Any]],
    config: ProviderConfigInput | None,
) -> list[dict[str, Any]]:
    """应用扩展 provider 配置到模型列表。"""
    if not config:
        return list(models)
    if not config.models:
        if config.base_url:
            return [{**m, "baseUrl": config.base_url} for m in models]
        return list(models)

    result: list[dict[str, Any]] = []
    for definition in config.models:
        defaults = next(
            (m for m in models if m["id"] == definition["id"]),
            models[0] if models else None,
        )
        api = definition.get("api") or config.api or (defaults or {}).get("api")
        if not api:
            raise ValueError(
                f"Provider {provider_id}, model {definition.get('id', '?')}: "
                'no "api" specified.'
            )
        base_url = (
            definition.get("baseUrl")
            or config.base_url
            or (defaults or {}).get("baseUrl")
        )
        if not base_url:
            raise ValueError(
                f'Provider {provider_id}: "baseUrl" is required when defining custom models.'
            )
        result.append(
            {
                **definition,
                "api": api,
                "provider": provider_id,
                "baseUrl": base_url,
                "headers": None,
            }
        )
    return result


def _adapt_oauth(config: ExtensionOAuthConfig) -> dict[str, Any]:
    """适配扩展 OAuth 配置为标准 OAuthAuth。"""

    async def login(callbacks: dict[str, Any]) -> OAuthCredentials:
        credential = await config.login(
            {
                "onAuth": lambda info: callbacks.get("notify", lambda _: None)(
                    {"type": "auth_url", **info}
                ),
                "onDeviceCode": lambda info: callbacks.get("notify", lambda _: None)(
                    {"type": "device_code", **info}
                ),
                "onPrompt": lambda prompt: callbacks.get("prompt", lambda _: None)(
                    {"type": "text", **prompt}
                ),
                "onProgress": lambda msg: callbacks.get("notify", lambda _: None)(
                    {"type": "progress", "message": msg}
                ),
                "onManualCodeInput": lambda: callbacks.get("prompt", lambda _: None)(
                    {"type": "manual_code", "message": "Paste the authorization code"}
                ),
                "onSelect": lambda prompt: callbacks.get("prompt", lambda _: None)(
                    {"type": "select", **prompt}
                ),
                "signal": callbacks.get("signal"),
            }
        )
        return cast(OAuthCredentials, {**credential, "type": "oauth"})

    async def refresh(credential: OAuthCredentials, signal: Any) -> OAuthCredentials:
        result = await config.refresh_token(credential, signal)
        return cast(OAuthCredentials, {**result, "type": "oauth"})

    async def to_auth(credential: OAuthCredentials) -> ModelAuth:
        return {"api_key": config.get_api_key(credential)}

    return {"login": login, "refresh": refresh, "toAuth": to_auth, "name": config.name}


def _with_configured_auth(
    auth: ModelAuth,
    headers: dict[str, str | None] | None,
    auth_header: bool,
) -> ModelAuth:
    """添加配置的认证头信息。"""
    merged_headers: dict[str, str | None] | None = None
    auth_headers = auth.get("headers")
    if auth_headers or headers:
        merged_headers = {**(auth_headers or {}), **(headers or {})}
    if auth_header:
        api_key = auth.get("api_key")
        if not api_key:
            raise ValueError("authHeader requires a resolved API key")
        merged_headers = {
            **(merged_headers or {}),
            "Authorization": f"Bearer {api_key}",
        }
    return cast(ModelAuth, {**auth, "headers": merged_headers})


def _configured_api_key(
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> str | None:
    """获取配置的 API key。"""
    if extension and extension.api_key:
        return extension.api_key
    if config and config.get("apiKey"):
        return str(config["apiKey"])
    return None


def _configured_headers(
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> dict[str, str] | None:
    """获取配置的头信息。"""
    config_headers: dict[str, str] = cast(
        dict[str, str], (config or {}).get("headers") or {}
    )
    ext_headers: dict[str, str] | None = extension.headers if extension else None
    if not config_headers and not ext_headers:
        return None
    merged = dict(config_headers)
    if ext_headers:
        merged.update(ext_headers)
    return merged


async def _config_context_env(
    values: list[str],
    ctx: AuthContext,
    explicit: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """解析配置值中的环境变量。"""
    env: dict[str, str] = dict(explicit or {})
    for name in set(name for v in values for name in get_config_value_env_var_names(v)):
        if name in env:
            continue
        value = await ctx.env(name)
        if value is not None:
            env[name] = value
    return env if env else None


def _compose_api_key_auth(
    provider_id: str,
    base: Provider | None,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> dict[str, Any] | None:
    """组合 API key 认证。"""
    inherited = getattr(base, "auth", None) if base else None
    raw_key = _configured_api_key(config, extension)
    oauth = extension.oauth if extension else None
    if not oauth:
        base_auth = getattr(base, "auth", {}) if base else {}
        oauth = base_auth.get("oauth") if isinstance(base_auth, dict) else None
    if not inherited and raw_key is None and oauth:
        return None

    raw_headers = _configured_headers(config, extension)
    auth_header = extension.auth_header if extension else None
    if auth_header is None:
        auth_header = (config or {}).get("authHeader", False) if config else False

    def _check(input: Any) -> Any:
        if input.get("credential"):
            return {"type": "api_key", "source": "stored credential"}
        if raw_key is not None:
            if is_command_config_value(raw_key):
                return {"type": "api_key", "source": "configured API key"}
            env_names = get_config_value_env_var_names(raw_key)
            return {"type": "api_key", "source": "configured API key"}
        return None

    async def _resolve(input: Any) -> AuthResult | None:
        if input.get("credential"):
            if input["credential"].get("key"):
                return {
                    "auth": {"api_key": input["credential"]["key"]},
                    "env": input["credential"].get("env"),
                    "source": "stored credential",
                }
            return None
        if raw_key is not None:
            env = await _config_context_env([raw_key], input.get("ctx", {}))
            key = resolve_config_value_or_throw(
                raw_key, f'API key for provider "{provider_id}"', env
            )
            header_env = await _config_context_env(
                list((raw_headers or {}).values()),
                input.get("ctx", {}),
                env,
            )
            headers = resolve_headers_or_throw(
                raw_headers, f'provider "{provider_id}"', header_env
            )
            return {
                "auth": _with_configured_auth(
                    {"api_key": key},
                    cast("dict[str, str | None]", headers),
                    auth_header,
                ),
                "source": "configured API key",
            }
        return None

    return {
        "check": _check,
        "resolve": _resolve,
        "name": "API key",
    }


def _compose_oauth_auth(
    provider_id: str,
    base: Provider | None,
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> dict[str, Any] | None:
    """组合 OAuth 认证。"""
    oauth = None
    if extension and extension.oauth:
        oauth = _adapt_oauth(extension.oauth)
    elif base and getattr(base, "auth", None):
        base_auth = getattr(base, "auth", {})
        oauth = base_auth.get("oauth") if isinstance(base_auth, dict) else None

    if not oauth:
        return None

    raw_headers = _configured_headers(config, extension)
    auth_header = extension.auth_header if extension else None
    if auth_header is None:
        auth_header = (config or {}).get("authHeader", False) if config else False

    async def to_auth(credential: OAuthCredentials) -> ModelAuth:
        auth = await oauth["toAuth"](credential)
        env = credential.get("env")
        headers = resolve_headers_or_throw(
            raw_headers,
            f'provider "{provider_id}"',
            env if isinstance(env, dict) else None,
        )
        return _with_configured_auth(
            auth,
            cast("dict[str, str | None]", headers),
            auth_header,
        )

    return {
        "login": oauth["login"],
        "refresh": oauth["refresh"],
        "toAuth": to_auth,
        "name": oauth.get("name", provider_id),
    }


def _raw_model_headers(
    model: dict[str, Any],
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> dict[str, str] | None:
    """获取模型的原始头信息。"""
    headers: dict[str, str] = {}
    if config:
        model_overrides = config.get("modelOverrides") or {}
        if model["id"] in model_overrides:
            override_headers = model_overrides[model["id"]].get("headers") or {}
            headers.update(override_headers)
        for entry in config.get("models") or []:
            if entry["id"] == model["id"]:
                headers.update(entry.get("headers") or {})
    if extension:
        for entry in extension.models or []:
            if entry.get("id") == model["id"]:
                headers.update(entry.get("headers") or {})
    return headers if headers else None


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def validate_extension_provider(
    provider_id: str,
    base: Provider | None,
    models_config: dict[str, Any] | None,
    extension: ProviderConfigInput,
) -> None:
    """验证扩展 provider 配置。"""
    if extension.stream_simple and not extension.api:
        raise ValueError(
            f'Provider {provider_id}: "api" is required when registering streamSimple.'
        )
    _apply_extension(
        provider_id,
        _apply_models_json(
            provider_id,
            cast(list[dict[str, Any]], list(base.get_models())) if base else [],
            models_config,
        ),
        extension,
    )


def compose_model_provider(
    provider_id: str,
    base: Provider | None,
    model_config: Any,  # ModelConfig
    extension: ProviderConfigInput | None,
) -> dict[str, Any]:
    """组合内置、models.json 和扩展 provider 层。"""
    config = model_config.get_provider(provider_id) if model_config else None
    extension_oauth_credential: OAuthCredentials | None = None
    refreshed_extension_models: list[dict[str, Any]] | None = None

    def current_extension() -> ProviderConfigInput | None:
        nonlocal refreshed_extension_models
        if extension and refreshed_extension_models is not None:
            ext = copy.copy(extension)
            ext.models = refreshed_extension_models
            return ext
        return extension

    def get_models() -> list[dict[str, Any]]:
        nonlocal extension_oauth_credential
        models = _apply_extension(
            provider_id,
            _apply_models_json(
                provider_id,
                cast(list[dict[str, Any]], list(base.get_models())) if base else [],
                config,
            ),
            current_extension(),
        )
        if (
            extension_oauth_credential
            and extension
            and extension.oauth
            and extension.oauth.modify_models
        ):
            models = extension.oauth.modify_models(models, extension_oauth_credential)
        # 应用模型覆盖
        model_overrides = (config or {}).get("modelOverrides") or {}
        return [
            _apply_model_override(m, model_overrides[m["id"]])
            if m["id"] in model_overrides
            else m
            for m in models
        ]

    # 立即验证
    get_models()

    api_key = _compose_api_key_auth(provider_id, base, config, extension)
    oauth = _compose_oauth_auth(provider_id, base, config, extension)
    if not api_key and not oauth:
        raise ValueError(
            f"Provider {provider_id}: no authentication method configured."
        )

    auth: dict[str, Any] = {}
    if api_key:
        auth["api_key"] = api_key
    if oauth:
        auth["oauth"] = oauth

    def supports_base_api(model: dict[str, Any]) -> bool:
        return any(
            getattr(m, "api", "") == model.get("api", "")
            for m in (base.get_models() if base else [])
        )

    def stream_with(
        model: dict[str, Any],
        context: Context,
        options: dict[str, Any] | None,
        simple: bool,
    ) -> AssistantMessageEventStream:
        return lazy_stream(
            cast(Model, model), lambda: _do_stream(model, context, options, simple)
        )

    def _do_stream(
        model: dict[str, Any],
        context: Context,
        options: dict[str, Any] | None,
        simple: bool,
    ) -> AssistantMessageEventStream:
        if extension and extension.stream_simple and model.get("api") == extension.api:
            return cast(
                AssistantMessageEventStream,
                extension.stream_simple(model, context, options),
            )
        if base and supports_base_api(model):
            if simple:
                return base.stream_simple(
                    cast(Model, model), context, cast("SimpleStreamOptions", options)
                )
            return base.stream(
                cast(Model, model), context, cast("StreamOptions", options)
            )
        api_provider = get_api_provider(model["api"])
        if not api_provider:
            raise ValueError(f"No API provider registered for api: {model['api']}")
        if simple:
            return cast(
                AssistantMessageEventStream,
                api_provider.stream_simple(model, context, options),
            )
        return cast(
            AssistantMessageEventStream, api_provider.stream(model, context, options)
        )

    provider_name = (
        (extension.name if extension else None)
        or (config.get("name") if config else None)
        or (base.name if base else None)
        or (extension.oauth.name if extension and extension.oauth else None)
        or provider_id
    )

    provider: dict[str, Any] = {
        "id": provider_id,
        "name": provider_name,
        "baseUrl": (extension.base_url if extension else None)
        or (config.get("baseUrl") if config else None)
        or (base.base_url if base else None),
        "headers": base.headers if base else None,
        "auth": auth,
        "getModels": get_models,
        "stream": lambda model, context, options=None: stream_with(
            model, context, options, False
        ),
        "streamSimple": lambda model, context, options=None: stream_with(
            model, context, options, True
        ),
    }

    # refreshModels
    if (base and getattr(base, "refresh_models", None) is not None) or (
        extension
        and (
            extension.refresh_models
            or (extension.oauth and extension.oauth.modify_models)
        )
    ):

        async def refresh_models(context: RefreshModelsContext) -> None:
            nonlocal refreshed_extension_models, extension_oauth_credential
            if base and getattr(base, "refresh_models", None) is not None:
                await base.refresh_models(context)
            if extension and extension.refresh_models:
                refreshed = await extension.refresh_models(context)
                # 验证
                _apply_extension(
                    provider_id,
                    _apply_models_json(
                        provider_id,
                        cast(list[dict[str, Any]], list(base.get_models()))
                        if base
                        else [],
                        config,
                    ),
                    ProviderConfigInput(models=refreshed),
                )
                refreshed_extension_models = refreshed
            if (
                getattr(context, "credential", None)
                and getattr(context.credential, "get", None)
                and context.credential.get("type") == "oauth"
            ):
                extension_oauth_credential = context.credential

        provider["refreshModels"] = refresh_models

    # filterModels
    if base and getattr(base, "filter_models", None) is not None:
        provider["filterModels"] = lambda models, credential: base.filter_models(
            models, credential
        )

    return provider


def resolve_configured_model_headers(
    model: dict[str, Any],
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """解析模型的已配置头信息。"""
    model_provider = model.get("provider", "?")
    model_id = model.get("id", "?")
    return resolve_headers_or_throw(
        _raw_model_headers(model, config, extension),
        f'model "{model_provider}/{model_id}"',
        env,
    )


def resolve_compatibility_request_config(
    model: dict[str, Any],
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> CompatibilityRequestConfig:
    """解析兼容性请求配置。"""
    model_provider = model.get("provider", "?")
    model_id = model.get("id", "?")
    config_headers: dict[str, str] = {}
    cfg_headers = (config or {}).get("headers") or {}
    config_headers.update(cfg_headers)
    raw_headers = _raw_model_headers(model, config, extension) or {}
    config_headers.update(raw_headers)
    configured = resolve_headers_or_throw(
        config_headers if config_headers else None,
        f'model "{model_provider}/{model_id}"',
    )
    model_headers = model.get("headers") or {}
    merged_headers = (
        {**model_headers, **(configured or {})}
        if configured
        else (model_headers or None)
    )
    auth_header = extension.auth_header if extension else None
    if auth_header is None:
        auth_header = (config or {}).get("authHeader", False) if config else False
    return CompatibilityRequestConfig(
        headers=merged_headers if merged_headers else None,
        auth_header=auth_header,
    )


def configured_request_auth_status(
    config: dict[str, Any] | None,
    extension: ProviderConfigInput | None,
) -> AuthStatus | None:
    """获取配置的认证状态。"""
    value = _configured_api_key(config, extension)
    if value is None:
        return None
    if is_command_config_value(value):
        return AuthStatus(configured=True, source="models_json_command")
    names = get_config_value_env_var_names(value)
    if names:
        configured = is_config_value_configured(value)
        return AuthStatus(
            configured=configured,
            source="environment" if configured else None,
            label=", ".join(names) if names else None,
        )
    return AuthStatus(
        configured=True,
        source="fallback" if (extension and extension.api_key) else "models_json_key",
    )
