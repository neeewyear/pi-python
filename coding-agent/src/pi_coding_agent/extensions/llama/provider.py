"""Provider 实现（对应 TS ``extensions/llama/provider.ts``）。

简化版实现，创建 llama.cpp Provider 控制器。
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from pi_ai.auth.types import ApiKeyCredential, AuthContext
from pi_ai.compat import stream, stream_simple
from pi_ai.models import (
    ModelRecord,
    ModelsPublication,
    Provider,
    RefreshModelsContext,
)
from pi_ai.models_store import ModelsStoreEntry
from pi_ai.types import (
    Context,
    DeferredCancelOptions,
    DeferredFetchOptions,
    DeferredHandle,
    Model,
    SimpleStreamOptions,
    StreamOptions,
)
from pi_ai.utils.abort import CancellationToken
from pi_ai.utils.event_stream import AssistantMessageEventStream

from .client import (
    LlamaClient,
    LlamaModelInfo,
    llama_inference_url,
    normalize_llama_server_url,
)

# ============================================================================
# Constants
# ============================================================================

LLAMA_PROVIDER_ID = "llama.cpp"
"""Provider ID（对应 TS ``LLAMA_PROVIDER_ID``）。"""

DEFAULT_LLAMA_SERVER_URL = "http://127.0.0.1:8080"
"""默认服务器 URL（对应 TS ``DEFAULT_LLAMA_SERVER_URL``）。"""


# ============================================================================
# Helper Functions
# ============================================================================


def _credential_server_url(
    credential: ApiKeyCredential | None,
) -> str | None:
    """从 credential 中提取服务器 URL。"""
    if credential is None:
        return None
    env = credential.get("env") or {}
    value = env.get("LLAMA_BASE_URL")
    if isinstance(value, str) and value.strip():
        return normalize_llama_server_url(value)
    return None


async def _resolve_server_url(
    ctx: AuthContext, credential: ApiKeyCredential | None
) -> str | None:
    """解析服务器 URL。"""
    configured = _credential_server_url(credential)
    if configured:
        return configured
    env_url = await ctx.env("LLAMA_BASE_URL")
    if env_url and env_url.strip():
        return normalize_llama_server_url(env_url.strip())
    return None


def _to_pi_model(model: LlamaModelInfo, server_url: str) -> ModelRecord:
    """将 LlamaModelInfo 转换为 ModelRecord。"""
    meta = model.meta
    reported_context_window = meta.n_ctx if meta else None
    if reported_context_window is None and meta:
        reported_context_window = meta.n_ctx_train
    context_window = (
        reported_context_window
        if reported_context_window and reported_context_window > 0
        else 128000
    )
    arch = model.architecture
    input_modalities: list[str] = ["text"]
    if arch and arch.input_modalities and "image" in arch.input_modalities:
        input_modalities = ["text", "image"]

    return ModelRecord(
        id=model.id,
        name=model.id,
        api="openai-completions",
        provider=LLAMA_PROVIDER_ID,
        base_url=llama_inference_url(server_url),
        reasoning=False,
        input=input_modalities,
        context_window=context_window,
        max_tokens=context_window,
    )


def _cancellation_to_event(
    token: CancellationToken | None,
) -> asyncio.Event:
    """将 CancellationToken 转换为 asyncio.Event。"""
    event = asyncio.Event()
    if token:
        token.add_callback(lambda: event.set())
    return event


# ============================================================================
# Provider Controller
# ============================================================================


class _LlamaProvider:
    """Llama.cpp Provider 实现（实现 Provider 协议）。"""

    def __init__(self, controller: LlamaProviderController) -> None:
        self._controller = controller
        self.id: str = LLAMA_PROVIDER_ID
        self.name: str = "llama.cpp"
        self.base_url: str | None = llama_inference_url(DEFAULT_LLAMA_SERVER_URL)
        self.headers: Any = None

    def get_models(self) -> list[Model]:
        return list(self._controller._models)

    def refresh_models(self, context: RefreshModelsContext) -> Any:
        return self._controller._refresh_models(context)

    def filter_models(self, models: list[Model], credential: Any) -> list[Model]:
        return models

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return stream(model, context, options)

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AssistantMessageEventStream:
        return stream_simple(model, context, options)

    def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AssistantMessageEventStream:
        raise NotImplementedError("Deferred fetch is not supported for llama.cpp")

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None:
        raise NotImplementedError("Deferred cancel is not supported for llama.cpp")


class LlamaProviderController:
    """Llama Provider 控制器（对应 TS ``LlamaProviderController``）。

    管理 provider 实例和模型目录。
    """

    def __init__(self) -> None:
        self._models: list[ModelRecord] = []
        self._provider: _LlamaProvider | None = None

    @property
    def provider(self) -> Provider:
        """获取 Provider 实例。"""
        if self._provider is None:
            self._provider = _LlamaProvider(self)
        return self._provider

    def set_catalog(self, catalog: Sequence[LlamaModelInfo], server_url: str) -> None:
        """设置模型目录。"""
        self._models = [
            _to_pi_model(m, server_url) for m in catalog if m.status.value == "loaded"
        ]

    async def _refresh_models(self, context: RefreshModelsContext) -> None:
        """刷新模型列表。"""
        if context.stored:
            restored = [
                m
                for m in (context.stored.models or [])
                if isinstance(m, dict)
                and m.get("provider") == LLAMA_PROVIDER_ID
                and m.get("api") == "openai-completions"
            ]

            def _update_stored() -> None:
                self._models.clear()
                self._models.extend(restored)  # type: ignore[arg-type]

            published = context.publish(ModelsPublication(update=_update_stored))
            if not published:
                return

        if not context.allow_network:
            return
        if context.signal and context.signal.aborted:
            return
        credential = context.credential
        if (
            not credential
            or not isinstance(credential, dict)
            or credential.get("type") != "api_key"
        ):
            return
        server_url = _credential_server_url(credential)  # type: ignore[arg-type]
        if not server_url:
            return
        signal = _cancellation_to_event(context.signal)
        catalog = await LlamaClient(
            server_url,
            credential.get("key"),  # type: ignore[arg-type]
        ).list(signal=signal)
        if context.signal and context.signal.aborted:
            return
        refreshed = [
            _to_pi_model(m, server_url) for m in catalog if m.status.value == "loaded"
        ]

        def _update_refreshed() -> None:
            self._models.clear()
            self._models.extend(refreshed)

        context.publish(
            ModelsPublication(
                persist=ModelsStoreEntry(
                    models=[m.model_dump() for m in refreshed],
                    checked_at=0,
                ),
                update=_update_refreshed,
            )
        )


def create_llama_provider() -> LlamaProviderController:
    """创建 Llama Provider 控制器（对应 TS ``createLlamaProvider``）。"""
    return LlamaProviderController()


__all__ = [
    "DEFAULT_LLAMA_SERVER_URL",
    "LLAMA_PROVIDER_ID",
    "LlamaProviderController",
    "create_llama_provider",
]
