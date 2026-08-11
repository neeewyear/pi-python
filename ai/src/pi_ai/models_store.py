"""模型存储（对应 ``models-store.ts``）。

提供 ``ModelsStoreEntry``、``ModelsStore`` 协议、``InMemoryModelsStore``。
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class ModelsStoreEntry:
    """模型存储条目（对应 TS ``ModelsStoreEntry``）。"""

    models: list[dict[str, Any]]
    """Provider 模型列表（序列化形式）。"""
    last_modified: int | None = None
    """远程目录的 Last-Modified Unix 时间戳。"""
    checked_at: int | None = None
    """上次远程检查完成的 Unix 时间戳。"""
    etag: str | None = None
    """远程目录的 ETag 值。"""


@dataclass
class ModelsStoreOperationOptions:
    """模型存储操作选项（对应 TS ``ModelsStoreOperationOptions``）。"""

    signal: Any = None  # CancellationToken


class ModelsStore(Protocol):
    """持久化模型目录存储（对应 TS ``ModelsStore``）。"""

    async def read(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> ModelsStoreEntry | None: ...

    async def write(self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None) -> None: ...

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None: ...


class InMemoryModelsStore:
    """内存模型存储（对应 TS ``InMemoryModelsStore``）。"""

    def __init__(self) -> None:
        self._entries: dict[str, ModelsStoreEntry] = {}

    async def read(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> ModelsStoreEntry | None:
        """读取 provider 的模型目录。"""
        entry = self._entries.get(provider_id)
        if entry is None:
            return None
        return deepcopy(entry)

    async def write(self, provider_id: str, entry: ModelsStoreEntry, options: ModelsStoreOperationOptions | None = None) -> None:
        """写入 provider 的模型目录。"""
        self._entries[provider_id] = deepcopy(entry)

    async def delete(self, provider_id: str, options: ModelsStoreOperationOptions | None = None) -> None:
        """删除 provider 的模型目录。"""
        self._entries.pop(provider_id, None)