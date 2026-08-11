"""模型注册表（对应 TS ``model-registry.ts``）。

``ModelRegistry`` 是编码代理扩展公开的同步兼容门面。
coding-agent 内部直接使用 ``ModelRuntime``。
"""

from __future__ import annotations

from typing import Any, cast

from pi_ai.auth import AuthResult
from pi_ai.models import (
    ModelsRefreshOptions,
    ModelsRefreshResult,
    Provider,
)
from pi_ai.types import AssistantMessage, Context, Model

from .model_runtime import ModelRuntime
from .provider_composer import (
    AuthStatus,
    ProviderConfigInput,
)


class ResolvedRequestAuth:
    """解析后的请求认证信息。"""


def _resolved_ok(
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    base_url: str | None = None,
    env: dict[str, str] | None = None,
) -> dict[str, Any]:
    return {
        "ok": True,
        "api_key": api_key,
        "headers": headers,
        "base_url": base_url,
        "env": env,
    }


def _resolved_error(error: str) -> dict[str, Any]:
    return {"ok": False, "error": error}


class ModelRegistry:
    """同步兼容门面，暴露给扩展使用。

    coding-agent 内部直接使用 ``ModelRuntime``。
    """

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime

    async def refresh(
        self, options: ModelsRefreshOptions | None = None
    ) -> ModelsRefreshResult:
        """重新加载 models.json。在进行同步注册表读取前等待此方法完成。"""
        return await self._runtime.refresh(options)

    def get_error(self) -> str | None:
        return self._runtime.get_error()

    def get_all(self) -> list[Model]:
        return list(self._runtime.get_models())

    def get_available(self) -> list[Model]:
        return list(self._runtime.get_available_snapshot())

    def find(self, provider: str, model_id: str) -> Model | None:
        return self._runtime.get_model(provider, model_id)

    def has_configured_auth(self, model: Model) -> bool:
        return self._runtime.has_configured_auth(model.provider)

    async def get_api_key_and_headers(self, model: Model) -> dict[str, Any]:
        try:
            resolution = await self._runtime.get_auth(model)
            if not resolution:
                compatibility = self._runtime.get_compatibility_request_config(model)
                if compatibility.auth_header:
                    return _resolved_error(f'No API key found for "{model.provider}"')
                return _resolved_ok(headers=compatibility.headers)
            auth = resolution.get("auth", {})
            return _resolved_ok(
                api_key=auth.get("api_key"),
                headers=cast("dict[str, str] | None", auth.get("headers")),
                base_url=auth.get("base_url"),
                env=resolution.get("env"),
            )
        except Exception as error:
            cause = error.__cause__ if isinstance(error, Exception) else None
            message = str(cause) if cause else str(error)
            if message == "authHeader requires a resolved API key":
                return _resolved_error(f'No API key found for "{model.provider}"')
            return _resolved_error(message)

    def get_provider_auth_status(self, provider: str) -> AuthStatus:
        return self._runtime.get_provider_auth_status(provider)

    def get_provider(self, provider: str) -> Provider | None:
        return self._runtime.get_provider(provider)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: Any = None,
    ) -> AssistantMessage:
        return await self._runtime.complete(model, context, options)

    def get_provider_display_name(self, provider: str) -> str:
        p = self._runtime.get_provider(provider)
        return p.name if p else provider

    async def get_provider_auth(self, provider: str) -> AuthResult | None:
        return await self._runtime.get_auth(provider)

    async def get_api_key_for_provider(self, provider: str) -> str | None:
        try:
            result = await self._runtime.get_auth(provider)
            if result:
                auth = result.get("auth", {})
                return auth.get("api_key")
        except Exception:
            pass
        return None

    def is_using_oauth(self, model: Model) -> bool:
        return self._runtime.is_using_oauth(model.provider)

    def register_provider(
        self,
        provider_or_name: Any,
        config: ProviderConfigInput | None = None,
    ) -> None:
        if isinstance(provider_or_name, str):
            if not config:
                raise ValueError("Provider config is required when registering by name")
            self._runtime.register_provider(provider_or_name, config)
            return
        self._runtime.register_native_provider(provider_or_name)

    def unregister_provider(self, provider_name: str) -> None:
        self._runtime.unregister_provider(provider_name)

    def get_registered_provider_config(
        self, provider_name: str
    ) -> ProviderConfigInput | None:
        return self._runtime.get_registered_provider_config(provider_name)

    def get_registered_native_provider(self, provider_name: str) -> Provider | None:
        return self._runtime.get_registered_native_provider(provider_name)

    def get_registered_provider_ids(self) -> list[str]:
        return self._runtime.get_registered_provider_ids()
