"""运行时凭据覆盖层。

``RuntimeCredentials`` 包装一个 ``CredentialStore``，提供非持久化的运行时
API key 覆盖机制。运行时设置的 key 优先级高于持久化存储中的 key。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pi_ai.auth import (
    AuthOperationOptions,
    Credential,
    CredentialInfo,
    CredentialStore,
)


class RuntimeCredentials(CredentialStore):
    """异步凭据存储覆盖层，用于非持久化的运行时 API key。

    当一个 provider 的 API key 通过命令行参数（``--api-key``）设置时，
    优先使用运行时 key 而非持久化存储中的 key。
    """

    def __init__(self, store: CredentialStore) -> None:
        self._store = store
        self._overrides: dict[str, str] = {}

    def set_runtime_api_key(self, provider_id: str, api_key: str) -> None:
        """设置运行时 API key。"""
        self._overrides[provider_id] = api_key

    def remove_runtime_api_key(self, provider_id: str) -> None:
        """移除运行时 API key。"""
        self._overrides.pop(provider_id, None)

    def has_runtime_api_key(self, provider_id: str) -> bool:
        """检查是否存在运行时 API key。"""
        return provider_id in self._overrides

    async def read(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> Credential | None:
        if options and options.get("signal"):
            options["signal"].throw_if_aborted()
        override = self._overrides.get(provider_id)
        if override:
            return {"type": "api_key", "key": override}
        return await self._store.read(provider_id, options)

    async def list(
        self, options: AuthOperationOptions | None = None
    ) -> list[CredentialInfo]:
        entries = {
            entry["provider_id"]: entry for entry in await self._store.list(options)
        }
        if options and options.get("signal"):
            options["signal"].throw_if_aborted()
        for provider_id in self._overrides:
            if provider_id not in entries:
                entries[provider_id] = CredentialInfo(
                    provider_id=provider_id, type="api_key"
                )
        return list(entries.values())

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        return await self._store.modify(provider_id, fn, options)

    async def delete(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> None:
        if options and options.get("signal"):
            options["signal"].throw_if_aborted()
        await self._store.delete(provider_id, options)
        self._overrides.pop(provider_id, None)
