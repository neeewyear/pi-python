"""图片生成 API 兼容层。"""

from __future__ import annotations

import asyncio
from typing import Any, Protocol, cast

from .auth.context import default_provider_auth_context as _default_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import ModelsError, ProviderAuthInfo, resolve_provider_auth
from .auth.types import (
    AuthContext,
    AuthResult,
    CredentialStore,
    ProviderAuth,
)
from .models import CreateModelsOptions
from .providers.images.register_builtins import register_builtin_images_providers
from .types import (
    AssistantImages,
    ImagesContext,
    ImagesModel,
    ImagesOptions,
    Usage,
)

register_builtin_images_providers()

# ---------------------------------------------------------------------------
# Images API 注册表
# ---------------------------------------------------------------------------


ImagesApiFunction = Any  # (model, context, options) -> AssistantImages


class _ImagesApiProviderInternal:
    """内部图片 API provider。"""

    api: str
    generate_images: ImagesApiFunction

    def __init__(self, api: str, generate_images: ImagesApiFunction) -> None:
        self.api = api
        self.generate_images = generate_images


class _RegisteredImagesApiProvider:
    """已注册的图片 API provider。"""

    provider: _ImagesApiProviderInternal
    source_id: str | None = None

    def __init__(
        self, provider: _ImagesApiProviderInternal, source_id: str | None = None
    ) -> None:
        self.provider = provider
        self.source_id = source_id


_images_api_provider_registry: dict[str, _RegisteredImagesApiProvider] = {}


def _wrap_generate_images(api: str, fn: Any) -> Any:
    """包装 generate_images 函数以验证 API 匹配。"""

    async def _wrapped(
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages:
        if model.api != api:
            raise ValueError(f"Mismatched api: {model.api} expected {api}")
        return cast(AssistantImages, await fn(cast(Any, model), context, options))

    return _wrapped


def register_images_api_provider(
    api: str,
    generate_images: Any,
    source_id: str | None = None,
) -> None:
    """注册图片 API provider。"""
    _images_api_provider_registry[api] = _RegisteredImagesApiProvider(
        provider=_ImagesApiProviderInternal(
            api, _wrap_generate_images(api, generate_images)
        ),
        source_id=source_id,
    )


def get_images_api_provider(api: str) -> _ImagesApiProviderInternal | None:
    """获取已注册的图片 API provider。"""
    entry = _images_api_provider_registry.get(api)
    return entry.provider if entry else None


# ---------------------------------------------------------------------------
# ImagesProvider / ImagesModels
# ---------------------------------------------------------------------------


class ImagesProvider(Protocol):
    """图片生成 provider。"""

    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def auth(self) -> ProviderAuth: ...

    def get_models(self) -> list[Any]: ...
    async def refresh_models(self) -> None: ...
    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: Any = None,
    ) -> AssistantImages: ...


class ImagesModels(Protocol):
    """图片生成模型集合。"""

    def get_providers(self) -> list[Any]: ...
    def get_provider(self, id: str) -> Any | None: ...
    def get_models(self, provider: str | None = None) -> list[Any]: ...
    def get_model(self, provider: str, id: str) -> Any | None: ...
    async def refresh(self, provider: str | None = None) -> None: ...
    async def get_auth(
        self, provider_or_model: str | Any, overrides: Any = None
    ) -> AuthResult | None: ...
    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: Any = None,
    ) -> AssistantImages: ...


class MutableImagesModels(Protocol):
    """可变图片生成模型集合。"""

    def set_provider(self, provider: Any) -> None: ...
    def delete_provider(self, id: str) -> None: ...
    def clear_providers(self) -> None: ...
    def get_providers(self) -> list[Any]: ...
    def get_provider(self, id: str) -> Any | None: ...
    def get_models(self, provider: str | None = None) -> list[Any]: ...
    def get_model(self, provider: str, id: str) -> Any | None: ...
    async def refresh(self, provider: str | None = None) -> None: ...
    async def get_auth(
        self, provider_or_model: str | Any, overrides: Any = None
    ) -> AuthResult | None: ...
    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: Any = None,
    ) -> AssistantImages: ...


class _ImagesModelsImpl:
    """ImagesModels 实现。"""

    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        self._providers: dict[str, Any] = {}
        self._credentials: CredentialStore = (
            options.credentials
            if options and options.credentials
            else InMemoryCredentialStore()
        )
        self._auth_context: AuthContext = (
            options.auth_context
            if options and options.auth_context
            else _default_auth_context()
        )

    def set_provider(self, provider: Any) -> None:
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        self._providers.clear()

    def get_providers(self) -> list[Any]:
        return list(self._providers.values())

    def get_provider(self, id: str) -> Any | None:
        return self._providers.get(id)

    def get_models(self, provider: str | None = None) -> list[Any]:
        if provider is not None:
            entry = self._providers.get(provider)
            if not entry:
                return []
            try:
                return cast("list[Any]", entry.get_models())
            except Exception:
                return []

        models: list[Any] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:
                pass
        return models

    def get_model(self, provider: str, id: str) -> Any | None:
        models = self.get_models(provider)
        for m in models:
            if m.model_id == id:
                return m
        return None

    async def refresh(self, provider: str | None = None) -> None:
        if provider is not None:
            entry = self._providers.get(provider)
            if (
                entry
                and hasattr(entry, "refresh_models")
                and callable(entry.refresh_models)
            ):
                try:
                    await entry.refresh_models()
                except ModelsError:
                    raise
                except Exception as error:
                    raise ModelsError(
                        "model_source", f"Model refresh failed for {provider}"
                    ) from error
            return

        tasks: list[Any] = []
        for entry in self._providers.values():
            if hasattr(entry, "refresh_models") and callable(entry.refresh_models):
                tasks.append(entry.refresh_models())
        await asyncio.gather(*tasks, return_exceptions=True)

    async def get_auth(
        self, provider_or_model: str | Any, overrides: Any = None
    ) -> AuthResult | None:
        provider_id = (
            provider_or_model
            if isinstance(provider_or_model, str)
            else provider_or_model.provider
        )
        entry = self._providers.get(provider_id)
        if not entry:
            return None
        # 转换为 ProviderAuthInfo 以匹配 resolve_provider_auth 签名
        auth_info: ProviderAuthInfo = {"id": entry.id, "auth": entry.auth}
        return await resolve_provider_auth(
            auth_info, self._credentials, self._auth_context, cast(Any, overrides)
        )

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: Any = None,
    ) -> AssistantImages:
        try:
            entry = self._providers.get(model.provider)
            if not entry:
                raise ModelsError("provider", f"Unknown provider: {model.provider}")

            resolution = await self.get_auth(
                model,
                {
                    "api_key": getattr(options, "api_key", None) if options else None,
                    "env": getattr(options, "env", None) if options else None,
                    "signal": getattr(options, "signal", None) if options else None,
                },
            )
            auth = resolution.get("auth") if resolution else None

            if not auth:
                return cast(
                    AssistantImages,
                    await entry.generate_images(model, context, options),
                )

            # 合并认证信息到请求选项
            merged_options = dict(
                getattr(options, "__dict__", options) if options else {}
            )
            auth_api_key = auth.get("api_key") if isinstance(auth, dict) else None
            merged_options["api_key"] = (
                getattr(options, "api_key", None) if options else None or auth_api_key
            )

            auth_headers = auth.get("headers") if isinstance(auth, dict) else None
            opt_headers = getattr(options, "headers", None) if options else None
            if auth_headers or opt_headers:
                merged_headers = dict(auth_headers or {})
                if opt_headers:
                    merged_headers.update(opt_headers)
                merged_options["headers"] = merged_headers

            resolution_env = resolution.get("env", {}) if resolution else {}
            opt_env = getattr(options, "env", {}) if options else {}
            if resolution_env or opt_env:
                merged_env = dict(resolution_env or {})
                merged_env.update(opt_env or {})
                merged_options["env"] = merged_env

            return cast(
                AssistantImages,
                await entry.generate_images(model, context, merged_options),
            )
        except Exception:
            return AssistantImages(
                data=[],
                usage=Usage(
                    input=0, output=0, cache_read=0, cache_write=0, total_tokens=0
                ),
            )


def create_images_models(options: CreateModelsOptions | None = None) -> Any:
    """创建图片生成模型集合。"""
    return _ImagesModelsImpl(options)


# ---------------------------------------------------------------------------
# CreateImagesProvider
# ---------------------------------------------------------------------------


class CreateImagesProviderOptions:
    """创建图片 provider 的选项。"""

    id: str
    name: str | None = None
    auth: ProviderAuth
    models: list[Any]
    refresh_models: Any = None
    api: Any  # ProviderImages


def create_images_provider(input: CreateImagesProviderOptions) -> Any:
    """创建图片生成 provider。"""
    _models = input.models
    _refresh_fn = input.refresh_models

    class _Provider:
        id = input.id
        name = input.name or input.id
        auth = input.auth
        _models = _models
        _inflight_refresh: Any = None

        def get_models(self) -> list[Any]:
            return self._models

        async def refresh_models(self) -> None:
            nonlocal _models
            if _refresh_fn:
                _models = await _refresh_fn()

        async def generate_images(
            self, model: Any, context: Any, options: Any = None
        ) -> AssistantImages:
            return cast(
                AssistantImages,
                await input.api.generate_images(model, context, options),
            )

    return _Provider()


# ---------------------------------------------------------------------------
# 顶层 generate_images
# ---------------------------------------------------------------------------


async def generate_images(
    model: ImagesModel,
    context: ImagesContext,
    options: Any = None,
) -> AssistantImages:
    """生成图片。"""
    provider = get_images_api_provider(model.api)
    if not provider:
        raise ValueError(f"No API provider registered for api: {model.api}")
    return cast(
        AssistantImages, await provider.generate_images(model, context, options)
    )


__all__ = [
    "ImagesModels",
    "ImagesProvider",
    "MutableImagesModels",
    "create_images_models",
    "create_images_provider",
    "generate_images",
    "get_images_api_provider",
    "register_images_api_provider",
]
