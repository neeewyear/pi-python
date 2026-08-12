"""模型运行时（对应 TS ``model-runtime.ts``）。

``ModelRuntime`` 是 coding-agent 的核心运行时，实现 ``Models`` 协议，
管理 provider 注册、模型加载、认证和 API 调用。
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, cast

from pi_ai.auth import (
    AuthCheck,
    AuthInteraction,
    AuthOperationOptions,
    AuthResult,
    Credential,
    CredentialInfo,
    CredentialStore,
)
from pi_ai.models import (
    AssistantMessageEventStream,
    CreateModelsOptions,
    Model,
    ModelsError,
    ModelsRefreshOptions,
    ModelsRefreshResult,
    ModelsStore,
    MutableModels,
    Provider,
    create_models,
    lazy_stream,
)
from pi_ai.types import (
    AssistantMessage,
    Context,
    DeferredFetchOptions,
    DeferredHandle,
    ProviderHeaders,
    SimpleStreamOptions,
    StreamOptions,
)

from pi_coding_agent.config import get_agent_dir

from .auth_storage import AuthStorage
from .model_config import ModelConfig
from .models_store import FileModelsStore, InMemoryCodingAgentModelsStore
from .provider_composer import (
    AuthStatus,
    CompatibilityRequestConfig,
    ProviderConfigInput,
    compose_model_provider,
    configured_request_auth_status,
    resolve_compatibility_request_config,
    resolve_configured_model_headers,
    validate_extension_provider,
)
from .remote_catalog_provider import with_remote_catalog
from .runtime_credentials import RuntimeCredentials

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

CredentialSynchronizationOperation = (
    str  # "login" | "logout" | "setRuntimeApiKey" | "removeRuntimeApiKey"
)


class ModelRuntimeSnapshot:
    """模型运行时快照。"""

    def __init__(
        self,
        all_models: list[Model] | None = None,
        available: list[Model] | None = None,
        configured_providers: set[str] | None = None,
        stored_providers: set[str] | None = None,
        auth: dict[str, AuthCheck | None] | None = None,
    ) -> None:
        self.all: list[Model] = all_models or []
        self.available: list[Model] = available or []
        self.configured_providers: set[str] = configured_providers or set()
        self.stored_providers: set[str] = stored_providers or set()
        self.auth: dict[str, AuthCheck | None] = auth or {}


class CreateModelRuntimeOptions:
    """创建 ModelRuntime 的选项。"""

    def __init__(
        self,
        credentials: CredentialStore | None = None,
        auth_path: str | None = None,
        models_path: str | None = None,
        models_store: ModelsStore | None = None,
        models_store_path: str | None = None,
        allow_model_network: bool = False,
        model_refresh_timeout_ms: int | None = None,
        catalog_base_url: str | None = None,
        signal: Any = None,
    ) -> None:
        self.credentials = credentials
        self.auth_path = auth_path
        self.models_path = models_path
        self.models_store = models_store
        self.models_store_path = models_store_path
        self.allow_model_network = allow_model_network
        self.model_refresh_timeout_ms = model_refresh_timeout_ms
        self.catalog_base_url = catalog_base_url
        self.signal = signal


class ModelRuntimeAuthOverrides:
    """模型运行时认证覆盖选项。"""

    def __init__(
        self,
        api_key: str | None = None,
        env: dict[str, str] | None = None,
        min_oauth_validity_ms: int | None = None,
        signal: Any = None,
    ) -> None:
        self.api_key = api_key
        self.env = env
        self.min_oauth_validity_ms = min_oauth_validity_ms
        self.signal = signal


class CredentialSynchronizationError(Exception):
    """凭据变更成功但本地快照同步失败。"""

    def __init__(
        self,
        provider_id: str,
        operation: str,
        credential: Credential | None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(
            f"Credential {operation} committed for {provider_id}, "
            f"but local synchronization failed"
        )
        self.provider_id = provider_id
        self.operation = operation
        self.credential = credential
        if cause:
            self.__cause__ = cause


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _merge_headers(
    base: ProviderHeaders | None,
    override: ProviderHeaders | None,
) -> ProviderHeaders | None:
    """合并头信息，覆盖同名 header（不区分大小写）。"""
    if not base and not override:
        return None
    merged = dict(base or {})
    for name, value in (override or {}).items():
        lower_name = name.lower()
        for existing_name in list(merged.keys()):
            if existing_name.lower() == lower_name:
                del merged[existing_name]
        merged[name] = value
    return merged


def _opt_get(options: Any, name: str, default: Any = None) -> Any:
    """从 dict 或 Pydantic 模型读取选项字段。"""
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


def _operation_signal(signal: Any) -> Any:
    """获取操作信号。"""
    import signal as _signal

    if signal is not None:
        return signal
    return _signal.Signals.SIGINT


async def _race_with_abort_signal(
    promise: asyncio.Future[Any],
    signal: Any,
) -> Any:
    """与中止信号竞速。"""
    if signal is None:
        return await promise
    try:
        return await asyncio.wait_for(promise, timeout=None)
    except asyncio.CancelledError:
        raise


def _model_to_dict(model: Model) -> dict[str, Any]:
    """将 Model 协议对象转换为字典（用于调用期望 dict 的 API）。"""
    if hasattr(model, "model_dump"):
        return dict(model.model_dump(by_alias=True))
    result: dict[str, Any] = {
        "id": model.model_id,
        "provider": model.provider,
        "api": model.api,
    }
    for attr in ("name", "base_url", "reasoning", "cost", "thinking_level_map"):
        if hasattr(model, attr):
            result[attr] = getattr(model, attr)
    return result


# ---------------------------------------------------------------------------
# ModelRuntime
# ---------------------------------------------------------------------------


class ModelRuntime:
    """配置好的 pi-ai Models 集合，供 coding-agent 和 SDK 消费者使用。"""

    def __init__(
        self,
        credentials: RuntimeCredentials,
        config: ModelConfig,
        models_path: str | None,
        models_store: ModelsStore,
        providers: list[Provider],
        model_network_enabled: bool,
    ) -> None:
        self._credentials = credentials
        self._config = config
        self._models_path = models_path
        self._model_network_enabled = model_network_enabled
        self._models = cast(
            MutableModels,
            create_models(
                CreateModelsOptions(
                    credentials=credentials,
                    models_store=models_store,
                )
            ),
        )
        self._default_builtins: dict[str, Provider] = {p.id: p for p in providers}
        self._builtins: dict[str, Provider] = dict(self._default_builtins)
        self._native_extension_providers: dict[str, Provider] = {}
        self._extension_providers: dict[str, ProviderConfigInput] = {}
        self._composition_errors: dict[str, str] = {}
        self._snapshot = ModelRuntimeSnapshot()
        self._availability_refresh_seq = 0
        self._availability_error_seq = 0
        self._provider_availability_seq: dict[str, int] = {}
        self._availability_error: str | None = None
        self._credential_operations: dict[str, asyncio.Task[Any]] = {}
        self._rebuild_providers()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    async def create(
        cls,
        options: CreateModelRuntimeOptions | None = None,
    ) -> ModelRuntime:
        """创建 ModelRuntime 实例。"""
        opts = options or CreateModelRuntimeOptions()
        credentials = RuntimeCredentials(
            opts.credentials
            or AuthStorage.create(Path(opts.auth_path) if opts.auth_path else None)
        )
        models_path: str | None = opts.models_path
        if models_path is None:
            models_path = str(get_agent_dir() / "models.json")

        config = await ModelConfig.load(models_path)
        models_store = opts.models_store
        if models_store is None:
            if models_path:
                from pathlib import Path as _Path

                models_dir = _Path(models_path).parent
                models_store = FileModelsStore(str(models_dir / "models-store.json"))
            else:
                models_store = InMemoryCodingAgentModelsStore()

        # 导入内置 provider 目录
        from pi_ai.providers.registry import builtin_providers

        builtin_model_data_generated_at: Any = None
        providers = [
            with_remote_catalog(
                provider,
                opts.catalog_base_url or "https://pi.dev",
                builtin_model_data_generated_at,
            )
            if provider.id != "radius"
            else provider
            for provider in builtin_providers()
        ]

        model_network_enabled = opts.allow_model_network
        runtime = cls(
            credentials,
            config,
            models_path,
            models_store,
            providers,
            model_network_enabled,
        )
        runtime._configure_radius_providers()
        runtime._rebuild_providers()

        refresh_from_network = model_network_enabled and opts.allow_model_network

        await runtime.refresh(
            ModelsRefreshOptions(
                allow_network=refresh_from_network,
                signal=opts.signal,
            )
        )

        return runtime

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _configure_radius_providers(self) -> None:
        """配置 Radius provider。"""
        self._builtins = dict(self._default_builtins)
        for provider_id in self._config.get_provider_ids():
            cfg = self._config.get_provider(provider_id)
            if not cfg or cfg.get("oauth") != "radius" or not cfg.get("baseUrl"):
                continue
            from pi_ai.providers import radius_provider

            self._builtins[provider_id] = radius_provider(
                provider_id=provider_id,
                name=cfg.get("name", provider_id) or provider_id,
                gateway=cfg["baseUrl"].rstrip("/v1").rstrip("/"),
            )

    def _provider_ids(self) -> set[str]:
        """获取所有 provider ID。"""
        ids: set[str] = set()
        ids.update(self._builtins.keys())
        ids.update(self._native_extension_providers.keys())
        ids.update(self._config.get_provider_ids())
        ids.update(self._extension_providers.keys())
        return ids

    def _recompose_provider(self, provider_id: str) -> None:
        """重新组合指定 provider。"""
        base = self._native_extension_providers.get(provider_id) or self._builtins.get(
            provider_id
        )
        extension = self._extension_providers.get(provider_id)
        config_provider = self._config.get_provider(provider_id)

        if not base and not config_provider and not extension:
            self._models.delete_provider(provider_id)
            self._composition_errors.pop(provider_id, None)
            return

        if base and not config_provider and not extension:
            self._models.set_provider(base)
            self._composition_errors.pop(provider_id, None)
            return

        try:
            composed = compose_model_provider(
                provider_id, base, self._config, extension
            )
            # 转换为 Provider 协议对象
            provider_obj = _dict_to_provider(composed)
            self._models.set_provider(provider_obj)
            self._composition_errors.pop(provider_id, None)
        except Exception as error:
            self._composition_errors[provider_id] = str(error)
            if base:
                self._models.set_provider(base)
            else:
                self._models.delete_provider(provider_id)

    def _rebuild_providers(self) -> None:
        """重建所有 provider。"""
        self._models.clear_providers()
        self._composition_errors.clear()
        for provider_id in self._provider_ids():
            self._recompose_provider(provider_id)
        self._update_model_snapshot()

    def _update_model_snapshot(self) -> None:
        """更新模型快照。"""
        all_models = list(self._models.get_models())
        self._snapshot.all = all_models
        self._snapshot.available = [
            m for m in all_models if m.provider in self._snapshot.configured_providers
        ]

    async def _run_availability_refresh(
        self, seq: int, error_seq: int, signal: Any
    ) -> None:
        """运行可用性刷新。"""
        providers = self._models.get_providers()

        available_task = self._models.get_available(None, {"signal": signal})
        checks_tasks = [
            self._models.check_auth(p.id, {"signal": signal}) for p in providers
        ]
        credentials_task = self._credentials.list({"signal": signal})

        available = await available_task
        checks = await asyncio.gather(*checks_tasks, return_exceptions=True)
        credentials = await credentials_task

        if seq != self._availability_refresh_seq:
            return

        auth: dict[str, AuthCheck | None] = {}
        configured_providers: set[str] = set()
        for i, provider in enumerate(providers):
            check = checks[i]
            if isinstance(check, Exception):
                auth[provider.id] = None
            else:
                auth[provider.id] = check  # type: ignore[assignment]
                if check is not None:
                    configured_providers.add(provider.id)

        self._snapshot = ModelRuntimeSnapshot(
            all_models=list(self._models.get_models()),
            available=list(available),
            configured_providers=configured_providers,
            stored_providers={c["provider_id"] for c in credentials},
            auth=auth,
        )
        if error_seq == self._availability_error_seq:
            self._availability_error = None

    async def _queue_availability_refresh(self, signal: Any = None) -> None:
        """排队可用性刷新。"""
        seq = self._availability_refresh_seq + 1
        self._availability_refresh_seq = seq
        for provider_id in self._provider_availability_seq:
            self._provider_availability_seq[provider_id] += 1
        error_seq = self._availability_error_seq + 1
        self._availability_error_seq = error_seq

        try:
            await self._run_availability_refresh(seq, error_seq, signal)
        except Exception as error:
            if error_seq == self._availability_error_seq and not (
                hasattr(signal, "aborted") and signal.aborted
            ):
                self._availability_error = str(error)
            raise

    async def _refresh_provider_availability(
        self, provider_id: str, signal: Any
    ) -> None:
        """刷新指定 provider 的可用性。"""
        self._availability_refresh_seq += 1
        provider_seq = self._provider_availability_seq.get(provider_id, 0) + 1
        self._provider_availability_seq[provider_id] = provider_seq
        error_seq = self._availability_error_seq + 1
        self._availability_error_seq = error_seq

        try:
            available = await self._models.get_available(
                provider_id, {"signal": signal}
            )
            auth = await self._models.check_auth(provider_id, {"signal": signal})
            credential = await self._credentials.read(provider_id, {"signal": signal})

            if self._provider_availability_seq.get(provider_id) != provider_seq:
                return

            configured_providers = set(self._snapshot.configured_providers)
            stored_providers = set(self._snapshot.stored_providers)
            auth_by_provider = dict(self._snapshot.auth)

            if auth:
                configured_providers.add(provider_id)
                auth_by_provider[provider_id] = auth
            else:
                configured_providers.discard(provider_id)
                auth_by_provider.pop(provider_id, None)

            if credential:
                stored_providers.add(provider_id)
            else:
                stored_providers.discard(provider_id)

            all_models = list(self._models.get_models())
            available_by_id = {
                f"{m.provider}\0{m.model_id}": m
                for m in list(self._snapshot.available)
                if m.provider != provider_id
            }
            for m in available:
                available_by_id[f"{m.provider}\0{m.model_id}"] = m

            self._snapshot = ModelRuntimeSnapshot(
                all_models=all_models,
                available=[
                    m
                    for m in all_models
                    if f"{m.provider}\0{m.model_id}" in available_by_id
                ],
                configured_providers=configured_providers,
                stored_providers=stored_providers,
                auth=auth_by_provider,
            )
            if error_seq == self._availability_error_seq:
                self._availability_error = None
        except Exception as error:
            if (
                self._provider_availability_seq.get(provider_id) == provider_seq
                and error_seq == self._availability_error_seq
                and not (hasattr(signal, "aborted") and signal.aborted)
            ):
                self._availability_error = str(error)
            raise

    # ------------------------------------------------------------------
    # Models 协议接口
    # ------------------------------------------------------------------

    def get_providers(self) -> list[Provider]:
        return self._models.get_providers()

    def get_provider(self, provider_id: str) -> Provider | None:
        return self._models.get_provider(provider_id)

    def get_models(self, provider_id: str | None = None) -> list[Model]:
        return self._models.get_models(provider_id)

    def get_model(self, provider_id: str, model_id: str) -> Model | None:
        return self._models.get_model(provider_id, model_id)

    async def check_auth(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> AuthCheck | None:
        return await self._models.check_auth(provider_id, options)

    async def get_available(
        self,
        provider_id: str | None = None,
        options: AuthOperationOptions | None = None,
    ) -> list[Model]:
        if provider_id:
            error_seq = self._availability_error_seq + 1
            self._availability_error_seq = error_seq
            try:
                available = await self._models.get_available(provider_id, options)
                if error_seq == self._availability_error_seq:
                    self._availability_error = None
                return list(available)
            except Exception as error:
                if error_seq == self._availability_error_seq and not getattr(
                    (options or {}).get("signal"), "aborted", False
                ):
                    self._availability_error = str(error)
                raise
        await self._queue_availability_refresh((options or {}).get("signal"))
        return list(self._snapshot.available)

    def get_available_snapshot(self) -> list[Model]:
        return list(self._snapshot.available)

    def get_error(self) -> str | None:
        errors: list[str] = []
        config_error = self._config.get_error()
        if config_error:
            errors.append(config_error)
        for provider_id, error in self._composition_errors.items():
            errors.append(f'Provider "{provider_id}": {error}')
        if self._availability_error:
            errors.append(f"Availability refresh: {self._availability_error}")
        return "\n\n".join(errors) if errors else None

    # ------------------------------------------------------------------
    # Provider 注册管理
    # ------------------------------------------------------------------

    def get_registered_provider_config(
        self, provider_id: str
    ) -> ProviderConfigInput | None:
        return self._extension_providers.get(provider_id)

    def get_registered_provider_ids(self) -> list[str]:
        return list(
            set(self._extension_providers.keys())
            | set(self._native_extension_providers.keys())
        )

    def get_registered_native_provider(self, provider_id: str) -> Provider | None:
        return self._native_extension_providers.get(provider_id)

    def get_compatibility_request_config(
        self, model: Model
    ) -> CompatibilityRequestConfig:
        return resolve_compatibility_request_config(
            _model_to_dict(model),
            self._config.get_provider(model.provider),
            self._extension_providers.get(model.provider),
        )

    def is_using_oauth(self, provider_id: str) -> bool:
        auth = self._snapshot.auth.get(provider_id)
        return auth is not None and auth.get("type") == "oauth"

    def has_configured_auth(self, provider_id: str) -> bool:
        return provider_id in self._snapshot.configured_providers

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: ModelRuntimeAuthOverrides | None = None,
    ) -> AuthResult | None:
        ov = overrides or ModelRuntimeAuthOverrides()
        if isinstance(provider_or_model, str):
            return await self._models.get_auth(provider_or_model, ov)

        model = provider_or_model
        resolution = await self._models.get_auth(model, ov)
        if not resolution:
            return None
        configured_headers = resolve_configured_model_headers(
            _model_to_dict(model),
            self._config.get_provider(model.provider),
            self._extension_providers.get(model.provider),
            {**(resolution.get("env") or {}), **(ov.env or {})},
        )
        auth = dict(resolution.get("auth", {}))
        auth["headers"] = _merge_headers(
            cast("ProviderHeaders | None", auth.get("headers")),
            cast("ProviderHeaders | None", configured_headers),
        )
        return cast("AuthResult | None", {**resolution, "auth": auth})

    def _enqueue_credential_operation(
        self, provider_id: str, signal: Any, task: Any
    ) -> Any:
        """排队凭据操作。"""
        prev: asyncio.Future[Any] | None = self._credential_operations.get(provider_id)
        if prev is None:
            prev = asyncio.get_event_loop().create_future()
            prev.set_result(None)

        async def _run() -> Any:
            await prev
            return await task()

        operation = asyncio.ensure_future(_run())
        tail = asyncio.ensure_future(
            self._cleanup_credential_operation(provider_id, operation)
        )
        self._credential_operations[provider_id] = operation
        return operation

    async def _cleanup_credential_operation(
        self, provider_id: str, operation: Any
    ) -> None:
        try:
            await operation
        except Exception:
            pass
        finally:
            if self._credential_operations.get(provider_id) is operation:
                del self._credential_operations[provider_id]

    async def _synchronize_credential_state(
        self,
        provider_id: str,
        operation: str,
        credential: Credential | None,
        signal: Any,
    ) -> None:
        try:
            self._recompose_provider(provider_id)
            composition_error = self._composition_errors.get(provider_id)
            if composition_error:
                raise ValueError(composition_error)
            result = await self._models.refresh(
                ModelsRefreshOptions(
                    allow_network=False,
                    providers=[provider_id],
                    signal=signal,
                )
            )
            self._update_model_snapshot()
            await self._refresh_provider_availability(provider_id, signal)
        except Exception as cause:
            raise CredentialSynchronizationError(
                provider_id, operation, credential, cause
            ) from cause

    async def set_runtime_api_key(
        self,
        provider_id: str,
        api_key: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        signal = (options or {}).get("signal")

        async def _task() -> None:
            self._credentials.set_runtime_api_key(provider_id, api_key)
            await self._synchronize_credential_state(
                provider_id,
                "setRuntimeApiKey",
                {"type": "api_key", "key": api_key},
                signal,
            )

        await self._enqueue_credential_operation(provider_id, signal, _task())

    async def remove_runtime_api_key(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        signal = (options or {}).get("signal")

        async def _task() -> None:
            self._credentials.remove_runtime_api_key(provider_id)
            await self._synchronize_credential_state(
                provider_id, "removeRuntimeApiKey", None, signal
            )

        await self._enqueue_credential_operation(provider_id, signal, _task())

    async def list_credentials(
        self, options: AuthOperationOptions | None = None
    ) -> list[CredentialInfo]:
        return await self._credentials.list(options)

    def get_provider_auth_status(self, provider_id: str) -> AuthStatus:
        if self._credentials.has_runtime_api_key(provider_id):
            return AuthStatus(configured=True, source="runtime")
        if provider_id in self._snapshot.stored_providers:
            return AuthStatus(configured=True, source="stored")
        configured = configured_request_auth_status(
            self._config.get_provider(provider_id),
            self._extension_providers.get(provider_id),
        )
        if configured:
            return configured
        check = self._snapshot.auth.get(provider_id)
        if check:
            return AuthStatus(
                configured=True,
                source="environment",
                label=check.get("source"),
            )
        return AuthStatus(configured=False)

    # ------------------------------------------------------------------
    # Provider 请求
    # ------------------------------------------------------------------

    async def _prepare_request(
        self,
        model: Model,
        options: Any = None,
    ) -> dict[str, Any]:
        """准备请求所需的 provider、model 和 options。

        保持 model 与 options 的原始类型（Pydantic 模型对象），
        仅把认证解析出的 api_key / headers / env 合并进去，
        供下游 API 实现通过属性（getattr）访问。
        """
        provider = self._models.get_provider(model.provider)
        if not provider:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")

        resolution = await self.get_auth(
            model, cast("ModelRuntimeAuthOverrides | None", options)
        )
        if not resolution:
            raise ModelsError("auth", f"Provider is not configured: {model.provider}")

        auth = resolution.get("auth", {})
        headers = _merge_headers(
            auth.get("headers"),
            _opt_get(options, "headers"),
        )
        env = None
        env_value = _opt_get(options, "env")
        if resolution.get("env") or env_value:
            env = {
                **(resolution.get("env") or {}),
                **(dict(env_value or {})),
            }

        req_model: Model = model
        if auth.get("base_url") and hasattr(model, "model_copy"):
            req_model = model.model_copy(update={"base_url": auth["base_url"]})

        api_key = _opt_get(options, "api_key") or auth.get("api_key")
        if options is not None and hasattr(options, "model_copy"):
            merged_options: Any = options.model_copy(
                update={"api_key": api_key, "headers": headers, "env": env}
            )
        else:
            merged_options = {
                **dict(options or {}),
                "api_key": api_key,
                "headers": headers,
                "env": env,
            }

        return {
            "provider": provider,
            "model": req_model,
            "options": merged_options,
        }

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return lazy_stream(model, lambda: self._do_stream(model, context, options))

    async def _do_stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        prepared = await self._prepare_request(model, options)
        return cast(
            AssistantMessageEventStream,
            prepared["provider"].stream(
                prepared["model"],
                context,
                prepared["options"],
            ),
        )

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        stream = self.stream(model, context, options)
        return await stream.result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return lazy_stream(
            model, lambda: self._do_stream_simple(model, context, options)
        )

    async def _do_stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        prepared = await self._prepare_request(model, options)
        return cast(
            AssistantMessageEventStream,
            prepared["provider"].stream_simple(
                prepared["model"],
                context,
                prepared["options"],
            ),
        )

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        stream = self.stream_simple(model, context, options)
        return await stream.result()

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessage:
        stream = lazy_stream(
            model, lambda: self._do_fetch_deferred(model, handle, options)
        )
        return await stream.result()

    async def _do_fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        prepared = await self._prepare_request(model, options)
        provider = prepared["provider"]
        if not hasattr(provider, "fetch_deferred") or provider.fetch_deferred is None:
            raise ModelsError(
                "provider",
                f"Provider {model.provider} does not support deferred responses",
            )
        return cast(
            AssistantMessageEventStream,
            provider.fetch_deferred(
                prepared["model"],
                handle,
                prepared["options"],
            ),
        )

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: Any = None,
    ) -> None:
        prepared = await self._prepare_request(model, options)
        provider = prepared["provider"]
        if not hasattr(provider, "cancel_deferred") or provider.cancel_deferred is None:
            raise ModelsError(
                "provider",
                f"Provider {model.provider} does not support deferred responses",
            )
        await provider.cancel_deferred(
            prepared["model"],
            handle,
            prepared["options"],
        )

    # ------------------------------------------------------------------
    # 认证生命周期
    # ------------------------------------------------------------------

    async def login(
        self,
        provider_id: str,
        auth_type: str,
        interaction: AuthInteraction,
    ) -> Credential:
        signal = getattr(interaction, "signal", None)

        async def _task() -> Credential:
            credential = await self._models.login(
                provider_id, auth_type, cast("Any", interaction)
            )
            await self._synchronize_credential_state(
                provider_id, "login", credential, signal
            )
            return cast(Credential, credential)

        return cast(
            Credential,
            await self._enqueue_credential_operation(provider_id, signal, _task()),
        )

    async def logout(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        signal = (options or {}).get("signal")

        async def _task() -> None:
            await self._models.logout(provider_id, {"signal": signal})
            await self._synchronize_credential_state(
                provider_id, "logout", None, signal
            )

        await self._enqueue_credential_operation(provider_id, signal, _task())

    async def refresh(
        self, options: ModelsRefreshOptions | None = None
    ) -> ModelsRefreshResult:
        opts = options or ModelsRefreshOptions()
        self._config = await ModelConfig.load(self._models_path)
        self._configure_radius_providers()

        if opts.providers:
            for provider_id in set(opts.providers):
                self._recompose_provider(provider_id)
            self._update_model_snapshot()
        else:
            self._rebuild_providers()

        refresh_options = ModelsRefreshOptions(
            allow_network=opts.allow_network
            if opts.allow_network is not None
            else self._model_network_enabled,
            providers=opts.providers,
            force=opts.force,
            signal=opts.signal,
        )
        result = await self._models.refresh(refresh_options)
        self._update_model_snapshot()

        errors: dict[str, Exception] = {}
        if opts.providers:
            for provider_id in set(opts.providers):
                try:
                    await self._refresh_provider_availability(provider_id, opts.signal)
                except Exception as error:
                    if not (
                        opts.signal is not None
                        and hasattr(opts.signal, "aborted")
                        and opts.signal.aborted
                    ):
                        errors[provider_id] = (
                            error
                            if isinstance(error, Exception)
                            else Exception(str(error))
                        )
        else:
            try:
                await self._queue_availability_refresh(opts.signal)
            except Exception:
                pass

        aborted = result.aborted or (
            opts.signal is not None
            and hasattr(opts.signal, "aborted")
            and opts.signal.aborted
        )
        return ModelsRefreshResult(
            aborted=aborted,
            errors=errors,
        )

    def register_native_provider(self, provider: Provider) -> None:
        if not provider.id.strip():
            raise ValueError("Provider id must not be empty.")
        self._extension_providers.pop(provider.id, None)
        self._native_extension_providers[provider.id] = provider
        self._recompose_provider(provider.id)
        self._update_model_snapshot()
        asyncio.ensure_future(self.refresh(ModelsRefreshOptions(allow_network=False)))

    def register_provider(self, provider_id: str, config: ProviderConfigInput) -> None:
        validate_extension_provider(
            provider_id,
            self._builtins.get(provider_id),
            self._config.get_provider(provider_id),
            config,
        )
        self._native_extension_providers.pop(provider_id, None)
        prev = self._extension_providers.get(provider_id)
        effective = ProviderConfigInput()
        if prev:
            for k, v in vars(prev).items():
                setattr(effective, k, v)
        for k, v in vars(config).items():
            if v is not None:
                setattr(effective, k, v)

        self._extension_providers[provider_id] = effective
        self._recompose_provider(provider_id)
        self._update_model_snapshot()

        auth_status = configured_request_auth_status(
            self._config.get_provider(provider_id), effective
        )
        if provider_id in self._snapshot.stored_providers or (
            auth_status is not None and auth_status.configured
        ):
            configured_providers = set(self._snapshot.configured_providers)
            configured_providers.add(provider_id)
            auth = dict(self._snapshot.auth)
            if provider_id not in auth:
                auth[provider_id] = {
                    "type": "oauth"
                    if effective.oauth and not effective.api_key
                    else "api_key",
                    "source": "configured provider",
                }
            self._snapshot.configured_providers = configured_providers
            self._snapshot.auth = auth
            self._snapshot.available = [
                m for m in self._snapshot.all if m.provider in configured_providers
            ]

        asyncio.ensure_future(self.refresh(ModelsRefreshOptions(allow_network=False)))

    def unregister_provider(self, provider_id: str) -> None:
        self._extension_providers.pop(provider_id, None)
        self._native_extension_providers.pop(provider_id, None)
        self._recompose_provider(provider_id)
        self._update_model_snapshot()
        asyncio.ensure_future(self.refresh(ModelsRefreshOptions(allow_network=False)))


# ---------------------------------------------------------------------------
# 辅助：dict 转 Provider 协议对象
# ---------------------------------------------------------------------------


def _dict_to_provider(data: dict[str, Any]) -> Provider:
    """将 dict 转换为 Provider 协议对象。"""

    class _DictProvider:
        id: str = data["id"]
        name: str = data["name"]
        base_url: str | None = data.get("baseUrl")
        headers: ProviderHeaders | None = data.get("headers")
        auth: Any = data.get("auth", {})

        def get_models(self) -> list[Model]:
            return cast("list[Model]", data["getModels"]())

        def refresh_models(self, context: Any) -> Any:
            fn = data.get("refreshModels")
            if fn:
                return fn(context)
            return None

        def filter_models(self, models: list[Model], credential: Any) -> list[Model]:
            fn = data.get("filterModels")
            if fn:
                return cast("list[Model]", fn(models, credential))
            return models

        def stream(
            self,
            model: Model,
            context: Context,
            options: StreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            return cast(
                AssistantMessageEventStream,
                data["stream"](model, context, options),
            )

        def stream_simple(
            self,
            model: Model,
            context: Context,
            options: SimpleStreamOptions | None = None,
        ) -> AssistantMessageEventStream:
            return cast(
                AssistantMessageEventStream,
                data["streamSimple"](model, context, options),
            )

        def fetch_deferred(
            self,
            model: Model,
            handle: DeferredHandle,
            options: Any = None,
        ) -> AssistantMessageEventStream:
            fn = data.get("fetchDeferred")
            if fn:
                return cast(
                    AssistantMessageEventStream,
                    fn(model, handle, options),
                )
            raise ModelsError(
                "provider",
                f"Provider {data.get('id')} does not support deferred responses",
            )

        async def cancel_deferred(
            self,
            model: Model,
            handle: DeferredHandle,
            options: Any = None,
        ) -> None:
            fn = data.get("cancelDeferred")
            if fn:
                await fn(model, handle, options)
            else:
                raise ModelsError(
                    "provider",
                    f"Provider {data.get('id')} does not support deferred responses",
                )

    return _DictProvider()
