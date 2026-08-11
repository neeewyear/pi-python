"""模型存储（对应 TS ``models-store.ts``）。

提供两种 ``ModelsStore`` 实现：
- ``InMemoryCodingAgentModelsStore``：内存存储，用于测试或无持久化需求
- ``FileModelsStore``：JSON 文件持久化存储
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, cast

from pi_ai.models_store import (
    ModelsStore,
    ModelsStoreEntry,
    ModelsStoreOperationOptions,
)

from .auth_storage import FileAuthStorageBackend, LockResult

StoredModels = dict[str, ModelsStoreEntry]


class InMemoryCodingAgentModelsStore(ModelsStore):
    """内存中的模型存储。"""

    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        entry = self._entries.get(provider_id)
        return copy.deepcopy(entry) if entry else None

    async def write(
        self,
        provider_id: str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions | None = None,
    ) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        self._entries[provider_id] = copy.deepcopy(entry)

    async def delete(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        if options and options.signal:
            options.signal.throw_if_aborted()
        self._entries.pop(provider_id, None)


class FileModelsStore(ModelsStore):
    """基于 JSON 文件的持久化模型存储（带锁定）。"""

    def __init__(self, path: str | None = None) -> None:
        from pi_coding_agent.config import get_agent_dir

        if path is None:
            path = str(get_agent_dir() / "models-store.json")
        self._storage = FileAuthStorageBackend(Path(path))

    def _parse(self, content: str | None) -> StoredModels:
        return json.loads(content) if content else {}

    async def read(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> ModelsStoreEntry | None:
        async def _read(content: str | None) -> LockResult:
            return LockResult(
                result=copy.deepcopy(self._parse(content).get(provider_id))
            )

        result = await self._storage.with_lock_async(_read, cast("Any", options))
        return result  # type: ignore[return-value]

    async def write(
        self,
        provider_id: str,
        entry: ModelsStoreEntry,
        options: ModelsStoreOperationOptions | None = None,
    ) -> None:
        async def _write(content: str | None) -> LockResult:
            current = self._parse(content)
            current[provider_id] = copy.deepcopy(entry)
            return LockResult(None, json.dumps(current, indent=2, ensure_ascii=False))

        await self._storage.with_lock_async(_write, cast("Any", options))

    async def delete(
        self, provider_id: str, options: ModelsStoreOperationOptions | None = None
    ) -> None:
        async def _delete(content: str | None) -> LockResult:
            current = self._parse(content)
            current.pop(provider_id, None)
            return LockResult(None, json.dumps(current, indent=2, ensure_ascii=False))

        await self._storage.with_lock_async(_delete, cast("Any", options))
