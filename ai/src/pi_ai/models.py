"""模型注册表与运行时。

提供 ``Models`` 集合、``Provider`` 协议、``ModelsImpl`` 实现类，
以及 ``calculateCost``、``getSupportedThinkingLevels`` 等工具函数。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterable, Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast, runtime_checkable

from pydantic import BaseModel, Field

from .auth.context import default_provider_auth_context as _default_auth_context
from .auth.credential_store import InMemoryCredentialStore
from .auth.resolve import ProviderAuthInfo, resolve_provider_auth
from .auth.types import (
    AuthContext,
    AuthResult,
    CredentialStore,
    ModelAuth,
)
from .env_api_keys import get_env_api_key
from .models_store import (
    InMemoryModelsStore,
    ModelsStore,
    ModelsStoreEntry,
)
from .types import (
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    ModelThinkingLevel,
    ProviderHeaders,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingLevelMap,
    Usage,
)
from .utils.abort import CancellationToken, race_with_abort_signal
from .utils.event_stream import AssistantMessageEventStream

# ---------------------------------------------------------------------------
# 错误类型
# ---------------------------------------------------------------------------

ModelsErrorCode: type = str  # "provider" | "auth" | "model_source" | "stream"


class ModelsError(Exception):
    """模型系统错误。"""

    def __init__(
        self, code: str, message: str, *, cause: BaseException | None = None
    ) -> None:
        self.code = code
        self.cause = cause
        super().__init__(message)


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class ModelCostRates(BaseModel):
    """模型成本费率。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = Field(default=0.0, alias="cacheRead")
    cache_write: float = Field(default=0.0, alias="cacheWrite")


class ModelCostTier(ModelCostRates):
    """模型成本费率层级。"""

    input_tokens_above: int = Field(default=0, alias="inputTokensAbove")


class ModelCost(ModelCostRates):
    """模型成本。"""

    tiers: list[ModelCostTier] | None = None


class ModelRecord(BaseModel):
    """模型记录——``Model`` 协议的具体实现。"""

    model_config = {"extra": "allow", "populate_by_name": True}

    model_id: str = Field(alias="id")
    """模型 ID。"""
    name: str = ""
    """模型显示名称。"""
    api: str = ""
    """API 类型。"""
    provider: str = ""
    """Provider ID。"""
    base_url: str = Field(default="", alias="baseUrl")
    """基础 URL。"""
    reasoning: bool = False
    """是否支持推理。"""
    thinking_level_map: ThinkingLevelMap | None = Field(
        default=None, alias="thinkingLevelMap"
    )
    """思考级别映射。"""
    input_types: list[str] = Field(default_factory=lambda: ["text"], alias="input")
    """输入类型（text / image）。"""
    cost: ModelCost = Field(default_factory=ModelCost)
    """成本费率。"""
    context_window: int = Field(default=0, alias="contextWindow")
    """上下文窗口大小。"""
    max_tokens: int = Field(default=0, alias="maxTokens")
    """最大输出 token 数。"""
    sampling_params: dict[str, Any] | None = None
    """默认采样参数。"""
    headers: dict[str, str] | None = None
    """模型级请求头覆盖。"""


@dataclass
class ModelsPublication:
    """Provider 发布的模型目录更新。"""

    persist: ModelsStoreEntry | None = None
    """Provider 选择的持久化目录。省略表示不修改存储；None 表示删除。"""
    update: Callable[[], None] | None = None
    """可选的同步内存状态更新。"""


@dataclass
class RefreshModelsContext:
    """模型刷新上下文。"""

    credential: Any = None  # Credential | None
    """有效的已配置凭证。"""
    stored: ModelsStoreEntry | None = None
    """刷新前捕捉的 Provider 目录快照。"""
    publish: Callable[[ModelsPublication], Any] = lambda _: True
    """发布更新。"""
    allow_network: bool = True
    """是否允许网络访问。"""
    force: bool | None = None
    """是否强制刷新。"""
    signal: CancellationToken | None = None
    """取消信号。"""


@dataclass
class ModelsRefreshOptions:
    """模型刷新选项。"""

    allow_network: bool = True
    """是否允许网络访问。"""
    providers: list[str] | None = None
    """限制刷新的 provider ID 列表。"""
    force: bool | None = None
    """是否强制刷新。"""
    signal: CancellationToken | None = None
    """取消信号。"""


@dataclass
class ModelsRefreshResult:
    """模型刷新结果。"""

    aborted: bool = False
    """是否被中止。"""
    errors: dict[str, Exception] = field(default_factory=dict)
    """Provider 错误映射。"""


@dataclass
class ModelsRequestTransforms:
    """模型请求转换。"""

    transform_headers: Callable[[ProviderHeaders], Any] | None = None
    """请求头发送前的转换函数。"""


ModelsApiStreamOptions: type = StreamOptions
"""API Stream 选项类型别名。"""

ModelsSimpleStreamOptions: type = SimpleStreamOptions
"""Simple Stream 选项类型别名。"""

ModelsDeferredFetchOptions: type = DeferredFetchOptions
"""Deferred 请求选项类型别名。"""

ModelsDeferredCancelOptions: type = DeferredCancelOptions
"""Deferred 取消选项类型别名。"""


@runtime_checkable
class Provider(Protocol):
    """Provider 运行时单元。

    拥有 id/name/base 元数据、认证方法、模型列表和流式行为。
    """

    id: str
    name: str
    base_url: str | None
    headers: ProviderHeaders | None

    def get_models(self) -> list[Model]: ...

    def refresh_models(self, context: RefreshModelsContext) -> Any: ...

    def filter_models(self, models: list[Model], credential: Any) -> list[Model]: ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None: ...


class Models(Protocol):
    """模型集合运行时。"""  

    def get_providers(self) -> list[Provider]: ...

    def get_provider(self, id: str) -> Provider | None: ...

    def get_models(self, provider: str | None = None) -> list[Model]: ...

    def get_model(self, provider: str, id: str) -> Model | None: ...

    async def refresh(
        self, options: ModelsRefreshOptions | None = None
    ) -> ModelsRefreshResult: ...

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream: ...

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage: ...

    async def check_auth(
        self, provider_id: str, options: Any | None = None
    ) -> Any | None: ...

    async def get_available(
        self, provider_id: str | None = None, options: Any | None = None
    ) -> list[Model]: ...

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: Any | None = None,
    ) -> Any | None: ...

    async def login(
        self, provider_id: str, auth_type: str, interaction: Any
    ) -> Any: ...

    async def logout(self, provider_id: str, options: Any | None = None) -> None: ...


class MutableModels(Models, Protocol):
    """可变的模型集合。"""  

    def set_provider(self, provider: Provider) -> None: ...

    def delete_provider(self, id: str) -> None: ...

    def clear_providers(self) -> None: ...


@dataclass
class CreateModelsOptions:
    """创建 Models 实例的选项。"""

    credentials: Any = None  # CredentialStore
    """凭证存储（可选，默认 InMemoryCredentialStore）。"""
    models_store: ModelsStore | None = None
    """模型存储（可选，默认 InMemoryModelsStore）。"""
    auth_context: Any = None  # AuthContext
    """认证上下文（可选）。"""


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _merge_headers(
    base: ProviderHeaders | None,
    override: ProviderHeaders | None,
) -> ProviderHeaders | None:
    """合并请求头（大小写不敏感）。"""
    if not base and not override:
        return None
    merged = dict(base or {})
    if override:
        for name, value in override.items():
            lower_name = name.lower()
            for existing_name in list(merged.keys()):
                if existing_name.lower() == lower_name:
                    del merged[existing_name]
            if value is not None:
                merged[name] = value
    return merged or None


def _create_setup_error_message(model: Model, error: BaseException) -> AssistantMessage:
    """创建设置错误消息。"""    
    # model 可能是 Model 对象，也可能是 provider 组合流程传入的 dict
    if isinstance(model, dict):
        api = model.get("api", "")
        provider = model.get("provider", "")
        model_id = model.get("model_id", model.get("id", ""))
    else:
        api = getattr(model, "api", "")
        provider = getattr(model, "provider", "")
        model_id = getattr(model, "model_id", "")
    return AssistantMessage(
        content=[],
        api=str(api),
        provider=str(provider),
        model=str(model_id),
        stop_reason="error",
        error_message=str(error),
        timestamp=int(time.time() * 1000),
    )


def lazy_stream(
    model: Model,
    setup: Callable[[], Any],
) -> AssistantMessageEventStream:
    """延迟流创建。 

    同步返回流，异步执行 setup（认证解析、延迟模块加载）。
    setup 失败时以错误事件终止流。
    """
    outer = AssistantMessageEventStream()

    async def _run() -> None:
        try:
            inner = await setup()
            async for event in inner:
                outer.push(event)
            # 尝试获取 result（如果 inner 有 result 方法）
            inner_result = getattr(inner, "result", None)
            if inner_result is not None:
                result = await inner_result()
                outer.end(result)
            else:
                outer.end()
        except Exception as error:
            message = _create_setup_error_message(model, error)
            outer.push(AssistantErrorEvent(error=message))
            outer.end(message)
        finally:
            if not outer._done:
                outer.end()

    asyncio.ensure_future(_run())
    return outer


# ---------------------------------------------------------------------------
# ModelsImpl
# ---------------------------------------------------------------------------


class ModelsImpl:
    """Models 实现类。  

    管理 Provider 集合、认证、模型刷新和流式请求。
    """

    def __init__(self, options: CreateModelsOptions | None = None) -> None:
        opts = options or CreateModelsOptions()
        self._providers: dict[str, Provider] = {}
        self._models_store: ModelsStore = opts.models_store or InMemoryModelsStore()
        self._refresh_generations: dict[str, int] = {}
        self._refresh_controllers: dict[str, Any] = {}
        self._publication_chains: dict[str, Any] = {}
        self._credentials: CredentialStore = (
            opts.credentials if opts and opts.credentials else InMemoryCredentialStore()
        )
        self._auth_context: AuthContext = (
            opts.auth_context if opts and opts.auth_context else _default_auth_context()
        )

    # -- Provider 管理 ---------------------------------------------------

    def set_provider(self, provider: Provider) -> None:
        """设置/替换 provider。"""
        self._supersede_provider_refresh(provider.id)
        self._providers[provider.id] = provider

    def delete_provider(self, id: str) -> None:
        """删除 provider。"""
        self._supersede_provider_refresh(id)
        self._providers.pop(id, None)

    def clear_providers(self) -> None:
        """清除所有 provider。"""
        for pid in list(self._providers.keys()):
            self._supersede_provider_refresh(pid)
        self._providers.clear()

    def get_providers(self) -> list[Provider]:
        """获取所有 provider。"""
        return list(self._providers.values())

    def get_provider(self, id: str) -> Provider | None:
        """按 ID 获取 provider。"""
        return self._providers.get(id)

    # -- 模型访问 -------------------------------------------------------

    def get_models(self, provider: str | None = None) -> list[Model]:
        """获取模型列表。"""
        if provider is not None:
            entry = self._providers.get(provider)
            if not entry:
                return []
            try:
                return entry.get_models()
            except Exception:
                return []
        models: list[Model] = []
        for entry in self._providers.values():
            try:
                models.extend(entry.get_models())
            except Exception:
                pass
        return models

    def get_model(self, provider: str, id: str) -> Model | None:
        """按 provider 和 ID 查找模型。"""
        for m in self.get_models(provider):
            if m.model_id == id:
                return m
        return None

    # -- 认证支持 -------------------------------------------------------

    async def check_auth(
        self, provider_id: str, options: Any | None = None
    ) -> Any | None:
        """检查 provider 认证状态。

        检查 provider 是否有可用的认证配置（API key 或 OAuth）。
        如果没有显式 auth 配置，回退到环境变量检查。
        """
        entry = self._providers.get(provider_id)
        if not entry:
            return None
        auth = getattr(entry, "auth", None)
        if auth:
            if auth.get("api_key") or auth.get("oauth"):
                return {"source": "configured", "type": "api_key"}
            return None
        # 无显式 auth 配置时，回退到环境变量
        override_env = cast(
            "dict[str, str] | None", getattr(options, "env", None) if options else None
        )
        api_key = get_env_api_key(provider_id, override_env)
        if api_key:
            return {"source": "environment", "type": "api_key"}
        return None

    async def get_auth(
        self,
        provider_or_model: str | Model,
        overrides: Any | None = None,
    ) -> Any | None:
        """获取 provider 认证信息。

        通过 ``resolve_provider_auth`` 解析认证：
        1. 已存储的凭据（OAuth 或 API key）
        2. 环境变量中的 API key
        3. 配置的 API key（models.json / extension）

        如果 provider 没有显式 auth 配置，则回退到环境变量检查。
        """
        provider_id = (
            provider_or_model
            if isinstance(provider_or_model, str)
            else provider_or_model.provider
        )
        entry = self._providers.get(provider_id)
        if not entry:
            return None
        auth = getattr(entry, "auth", None)
        if auth:
            auth_info: ProviderAuthInfo = {"id": entry.id, "auth": auth}
            return await resolve_provider_auth(
                auth_info, self._credentials, self._auth_context, cast("Any", overrides)
            )
        # 无显式 auth 配置时，回退到环境变量
        override_env = cast(
            "dict[str, str] | None",
            getattr(overrides, "env", None) if overrides else None,
        )
        api_key = get_env_api_key(provider_id, override_env)
        if api_key:
            return AuthResult(
                auth=ModelAuth(api_key=api_key),
                source="environment",
                env={provider_id: api_key} if override_env else None,
            )
        return None

    async def get_available(
        self, provider_id: str | None = None, options: Any | None = None
    ) -> list[Model]:
        """获取可用模型列表。

        如果指定 provider_id，只返回该 provider 的模型；
        否则返回所有 provider 的模型。
        """
        return self.get_models(provider_id)

    async def login(self, provider_id: str, auth_type: str, interaction: Any) -> Any:
        """登录 provider（默认抛出未实现错误）。"""
        raise NotImplementedError(f"Login not supported for provider: {provider_id}")

    async def logout(self, provider_id: str, options: Any | None = None) -> None:
        """登出 provider（默认无操作）。"""
        return

    # -- 刷新机制 -------------------------------------------------------

    def _supersede_provider_refresh(self, provider_id: str) -> int:
        """取消正在进行的 provider 刷新。"""
        generation = self._refresh_generations.get(provider_id, 0) + 1
        self._refresh_generations[provider_id] = generation
        previous = self._refresh_controllers.pop(provider_id, None)
        if previous is not None:
            previous.cancel()
        return generation

    def _begin_provider_refresh(self, provider_id: str) -> tuple[int, Any]:
        """开始 provider 刷新。"""
        generation = self._supersede_provider_refresh(provider_id)
        controller = CancellationToken()
        self._refresh_controllers[provider_id] = controller
        return generation, controller

    async def _publish_provider_models(
        self,
        provider_id: str,
        generation: int,
        signal: CancellationToken,
        publication: ModelsPublication,
    ) -> bool:
        """发布 provider 模型更新。"""
        previous = self._publication_chains.get(provider_id)
        if previous is not None:
            try:
                await previous
            except Exception:
                pass

        async def _queued() -> bool:
            if (
                signal.aborted
                or self._refresh_generations.get(provider_id) != generation
            ):
                return False
            if publication.persist is None:
                await self._models_store.delete(provider_id)
            elif publication.persist is not None:
                await self._models_store.write(provider_id, publication.persist)
            if (
                signal.aborted
                or self._refresh_generations.get(provider_id) != generation
            ):
                return False
            if publication.update:
                publication.update()
            return True

        queued_coro = _queued()
        self._publication_chains[provider_id] = queued_coro
        try:
            result = await race_with_abort_signal(queued_coro, signal)
            return bool(result) if result is not None else False
        except Exception:
            return False

    async def _run_provider_refresh_phase(
        self,
        provider: Provider,
        credential: Any,
        allow_network: bool,
        force: bool | None,
        generation: int,
        signal: CancellationToken,
    ) -> None:
        """运行 provider 刷新阶段。"""
        stored = await self._models_store.read(provider.id)
        context = RefreshModelsContext(
            credential=credential,
            stored=stored,
            publish=lambda pub: self._publish_provider_models(
                provider.id, generation, signal, pub
            ),
            allow_network=allow_network,
            force=force if allow_network else None,
            signal=signal,
        )
        await provider.refresh_models(context)

    async def refresh(
        self, options: ModelsRefreshOptions | None = None
    ) -> ModelsRefreshResult:
        """刷新模型目录。"""
        opts = options or ModelsRefreshOptions()
        signal = opts.signal or CancellationToken()
        errors: dict[str, Exception] = {}
        if signal.aborted:
            return ModelsRefreshResult(aborted=True, errors=errors)

        selected = set(opts.providers) if opts.providers else None
        refreshable = [
            p
            for p in self._providers.values()
            if hasattr(p, "refresh_models") and (selected is None or p.id in selected)
        ]

        async def _refresh_provider(provider: Provider) -> None:
            generation, controller = self._begin_provider_refresh(provider.id)
            # 组合信号
            combined_signal = CancellationToken()
            if signal.aborted or controller.aborted:
                combined_signal.cancel()
            else:
                signal.add_callback(
                    lambda: (
                        combined_signal.cancel()
                        if not combined_signal.aborted
                        else None
                    )
                )
                controller.add_callback(
                    lambda: (
                        combined_signal.cancel()
                        if not combined_signal.aborted
                        else None
                    )
                )

            try:
                # 第一阶段：离线恢复
                await self._run_provider_refresh_phase(
                    provider, None, False, None, generation, combined_signal
                )
                if not opts.allow_network or combined_signal.aborted:
                    return
                # 第二阶段：网络刷新
                await self._run_provider_refresh_phase(
                    provider, None, True, opts.force, generation, combined_signal
                )
            except Exception as error:
                if not signal.aborted:
                    errors[provider.id] = error
            finally:
                if self._refresh_controllers.get(provider.id) is controller:
                    self._refresh_controllers.pop(provider.id, None)

        tasks = [_refresh_provider(p) for p in refreshable]
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

        return ModelsRefreshResult(aborted=signal.aborted, errors=errors)

    # -- 流式请求 -------------------------------------------------------

    def _require_provider(self, model: Model) -> Provider:
        """获取 model 所属的 provider。"""
        provider = self._providers.get(model.provider)
        if not provider:
            raise ModelsError("provider", f"Unknown provider: {model.provider}")
        return provider

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """流式请求。"""

        async def _setup() -> AsyncIterable[AssistantMessageEvent]:
            provider = self._require_provider(model)
            return provider.stream(model, context, options)

        return lazy_stream(model, _setup)

    async def complete(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessage:
        """完整请求（收集流并返回最终消息）。"""
        return await self.stream(model, context, options).result()

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        """简化流式请求。"""

        async def _setup() -> AsyncIterable[AssistantMessageEvent]:
            provider = self._require_provider(model)
            return provider.stream_simple(model, context, options)

        return lazy_stream(model, _setup)

    async def complete_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessage:
        """简化完整请求。"""
        return await self.stream_simple(model, context, options).result()

    async def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessage:
        """获取 deferred 结果。"""
        provider = self._require_provider(model)
        if not hasattr(provider, "fetch_deferred"):
            raise ModelsError(
                "provider",
                f"Provider {model.provider} does not support deferred responses",
            )
        stream = lazy_stream(
            model, lambda: provider.fetch_deferred(model, handle, options)
        )
        return await stream.result()

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None:
        """取消 deferred 请求。"""
        provider = self._require_provider(model)
        cancel_deferred = getattr(provider, "cancel_deferred", None)
        if cancel_deferred is None:
            raise ModelsError(
                "provider",
                f"Provider {model.provider} does not support deferred responses",
            )
        await cancel_deferred(model, handle, options)


# ---------------------------------------------------------------------------
# ProviderImpl 具体实现
# ---------------------------------------------------------------------------


@dataclass
class _ProviderImpl:
    """``Provider`` 协议的具体实现（内部使用）。"""

    id: str
    name: str
    base_url: str | None = None
    headers: ProviderHeaders | None = None
    _get_models: Callable[[], list[Model]] = list
    _refresh_models: Any = None
    _filter_models: Any = None
    _stream: Callable[..., AssistantMessageEventStream] = lambda *a, **kw: (
        AssistantMessageEventStream()
    )
    _stream_simple: Callable[..., AssistantMessageEventStream] = lambda *a, **kw: (
        AssistantMessageEventStream()
    )

    def get_models(self) -> list[Model]:
        return self._get_models()

    def refresh_models(self, context: RefreshModelsContext) -> Any:
        if self._refresh_models is not None:
            return self._refresh_models(context)
        return None

    def filter_models(self, models: list[Model], credential: Any) -> list[Model]:
        if self._filter_models is not None:
            return cast("list[Model]", self._filter_models(models, credential))
        return models

    def stream(
        self, model: Model, context: Context, options: StreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if self._stream is not None:
            return self._stream(model, context, options)
        return AssistantMessageEventStream()

    def stream_simple(
        self, model: Model, context: Context, options: SimpleStreamOptions | None = None
    ) -> AssistantMessageEventStream:
        if self._stream_simple is not None:
            return self._stream_simple(model, context, options)
        return AssistantMessageEventStream()

    def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        raise ModelsError(
            "provider", f"Provider {self.id} does not support deferred responses"
        )

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None:
        raise ModelsError(
            "provider", f"Provider {self.id} does not support deferred responses"
        )


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def create_models(options: CreateModelsOptions | None = None) -> ModelsImpl:
    """创建 Models 实例。"""
    return ModelsImpl(options)


@dataclass
class CreateProviderOptions:
    """创建 Provider 的选项。"""

    id: str
    """Provider ID。"""
    name: str | None = None
    """显示名称（默认：id）。"""
    base_url: str | None = None
    """基础 URL。"""
    headers: ProviderHeaders | None = None
    """请求头。"""
    models: list[Model] = field(default_factory=list)
    """静态基线模型列表。"""
    fetch_models: Callable[[RefreshModelsContext], Any] | None = None
    """动态模型获取函数。"""
    filter_models: Callable[[list[Model], Any], list[Model]] | None = None
    """模型过滤函数。"""
    api: Any = None
    """ProviderStreams 实现或 API 映射。"""

def create_provider(input: CreateProviderOptions) -> Provider:
    """从组件构建 Provider。"""
    baseline_models = list(input.models)
    dynamic_models: list[Model] = []

    def _current_models() -> list[Model]:
        merged = list(baseline_models)
        existing_ids = {m.model_id for m in merged}
        for dm in dynamic_models:
            if dm.model_id in existing_ids:
                for i, m in enumerate(merged):
                    if m.model_id == dm.model_id:
                        merged[i] = dm
                        break
            else:
                merged.append(dm)
        return merged

    # 判断是单 API 实现还是 API 映射
    single = (
        input.api
        if hasattr(input.api, "stream") and callable(input.api.stream)
        else None
    )
    by_api = None if single else input.api

    def _api_for(model: Model) -> Any:
        if single:
            return single
        if by_api and isinstance(by_api, dict):
            return by_api.get(model.api)
        return None

    async def _raise_no_api_error(model: Model, provider_id: str) -> None:
        raise ModelsError(
            "stream",
            f'Provider {provider_id} has no API implementation for "{model.api}"',
        )

    def _dispatch(
        model: Model,
        run: Callable[[Any], AssistantMessageEventStream],
    ) -> AssistantMessageEventStream:
        streams = _api_for(model)
        if not streams:
            return lazy_stream(model, lambda: _raise_no_api_error(model, input.id))
        return run(streams)

    async def _refresh_impl(context: RefreshModelsContext) -> None:
        nonlocal dynamic_models
        if context.stored:
            restored = [
                m
                for m in context.stored.models
                if isinstance(m, dict) and m.get("provider") == input.id
            ]
            # 将 restored 转为 ModelRecord
            restored_models = [
                ModelRecord(**m) for m in restored if isinstance(m, dict)
            ]
            published = await context.publish(
                ModelsPublication(
                    update=lambda: setattr(_refresh_impl, "_restored", restored_models)
                )
            )
            if published:
                dynamic_models = list(restored_models)
        if not context.allow_network or (context.signal and context.signal.aborted):
            return
        if input.fetch_models:
            refreshed = await input.fetch_models(context)
            if context.signal and context.signal.aborted:
                return
            refreshed_list = (
                list(refreshed) if isinstance(refreshed, (list, tuple)) else []
            )
            await context.publish(
                ModelsPublication(
                    persist=ModelsStoreEntry(
                        models=[
                            m.model_dump() if hasattr(m, "model_dump") else dict(m)
                            for m in refreshed_list
                        ],
                        checked_at=int(time.time() * 1000),
                    ),
                    update=lambda: setattr(_refresh_impl, "_refreshed", refreshed_list),
                )
            )
            dynamic_models = refreshed_list

    # 使用 dataclass 创建 Provider 实例
    provider_obj = _ProviderImpl(
        id=input.id,
        name=input.name or input.id,
        base_url=input.base_url,
        headers=input.headers,
        _get_models=_current_models,
        _refresh_models=_refresh_impl if input.fetch_models else None,
        _filter_models=input.filter_models,
        _stream=lambda model, context, options=None: _dispatch(
            model, lambda s: s.stream(model, context, options)
        ),
        _stream_simple=lambda model, context, options=None: _dispatch(
            model, lambda s: s.stream_simple(model, context, options)
        ),
    )
    return provider_obj


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def has_api(model: Model, api: str) -> bool:
    """检查模型是否使用指定 API。"""
    return model.api == api


def calculate_cost(model: Model, usage: Usage) -> Usage:
    """计算 token 成本。

    基于输入 token 分档计算，包含 Anthropic 缓存写入 2x 计费逻辑。
    """
    input_tokens = usage.input + usage.cache_read + usage.cache_write

    # 获取模型成本费率 —— 从 model 对象中提取
    cost_rates = getattr(model, "cost", None)
    if cost_rates is None:
        return usage

    # 提取费率
    base_input = getattr(cost_rates, "input", 0.0)
    base_output = getattr(cost_rates, "output", 0.0)
    base_cache_read = getattr(cost_rates, "cache_read", 0.0)
    base_cache_write = getattr(cost_rates, "cache_write", 0.0)

    rates = {
        "input": base_input,
        "output": base_output,
        "cache_read": base_cache_read,
        "cache_write": base_cache_write,
    }
    matched_threshold = -1

    # 分档匹配
    tiers = getattr(cost_rates, "tiers", None) or []
    for tier in tiers:
        threshold = getattr(tier, "input_tokens_above", 0)
        if input_tokens > threshold > matched_threshold:
            rates["input"] = getattr(tier, "input", base_input)
            rates["output"] = getattr(tier, "output", base_output)
            rates["cache_read"] = getattr(tier, "cache_read", base_cache_read)
            rates["cache_write"] = getattr(tier, "cache_write", base_cache_write)
            matched_threshold = threshold

    # Anthropic 1h 缓存写入 2x 计费
    cache_write_1h = getattr(usage, "cache_write_1h", 0) or 0
    short_write = usage.cache_write - cache_write_1h

    if usage.cost is not None:
        usage.cost.input = (rates["input"] / 1_000_000) * usage.input
        usage.cost.output = (rates["output"] / 1_000_000) * usage.output
        usage.cost.cache_read = (rates["cache_read"] / 1_000_000) * usage.cache_read
        usage.cost.cache_write = (
            rates["cache_write"] * short_write + base_input * 2 * cache_write_1h
        ) / 1_000_000
        usage.cost.total = (
            usage.cost.input
            + usage.cost.output
            + usage.cost.cache_read
            + usage.cost.cache_write
        )

    return usage


EXTENDED_THINKING_LEVELS: list[ModelThinkingLevel] = [
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
]


def get_supported_thinking_levels(model: Model) -> list[ModelThinkingLevel]:
    """获取模型支持的思考级别。"""
    reasoning = getattr(model, "reasoning", False)
    if not reasoning:
        return ["off"]

    thinking_level_map = getattr(model, "thinking_level_map", None) or {}

    def _is_supported(level: ModelThinkingLevel) -> bool:
        mapped = thinking_level_map.get(level)
        if mapped is None:
            return False
        if level in ("xhigh", "max"):
            return mapped is not None
        return True

    return [level for level in EXTENDED_THINKING_LEVELS if _is_supported(level)]


def clamp_thinking_level(model: Model, level: ModelThinkingLevel) -> ModelThinkingLevel:
    """将思考级别夹紧到模型支持的范围内。"""
    available = get_supported_thinking_levels(model)
    if level in available:
        return level

    if level in EXTENDED_THINKING_LEVELS:
        requested_index = EXTENDED_THINKING_LEVELS.index(level)
    else:
        return available[0] if available else "off"

    # 向上搜索
    for i in range(requested_index, len(EXTENDED_THINKING_LEVELS)):
        if EXTENDED_THINKING_LEVELS[i] in available:
            return EXTENDED_THINKING_LEVELS[i]
    # 向下搜索
    for i in range(requested_index - 1, -1, -1):
        if EXTENDED_THINKING_LEVELS[i] in available:
            return EXTENDED_THINKING_LEVELS[i]

    return available[0] if available else "off"


def models_are_equal(a: Model | None, b: Model | None) -> bool:
    """检查两个模型是否相等。"""
    if not a or not b:
        return False
    return a.model_id == b.model_id and a.provider == b.provider


__all__ = [
    "EXTENDED_THINKING_LEVELS",
    "AssistantMessageEventStream",
    "CreateModelsOptions",
    "Model",
    "ModelCost",
    "ModelCostRates",
    "ModelCostTier",
    "ModelRecord",
    "Models",
    "ModelsApiStreamOptions",
    "ModelsDeferredCancelOptions",
    "ModelsDeferredFetchOptions",
    "ModelsError",
    "ModelsErrorCode",
    "ModelsImpl",
    "ModelsPublication",
    "ModelsRefreshOptions",
    "ModelsRefreshResult",
    "ModelsRequestTransforms",
    "ModelsSimpleStreamOptions",
    "ModelsStore",
    "MutableModels",
    "Provider",
    "RefreshModelsContext",
    "calculate_cost",
    "clamp_thinking_level",
    "create_models",
    "create_provider",
    "get_supported_thinking_levels",
    "has_api",
    "lazy_stream",
]
