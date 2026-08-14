"""凭据存储。

提供 ``AuthStorageBackend`` 协议、文件/内存后端实现，以及 ``AuthStorage``
类实现 ``CredentialStore`` 接口。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import orjson
from pi_ai.auth import (
    AuthOperationOptions,
    Credential,
    CredentialInfo,
    CredentialStore,
)

from ..config import get_auth_path

# ---------------------------------------------------------------------------
# LockResult
# ---------------------------------------------------------------------------


class LockResult(NamedTuple):
    """锁定操作结果。"""

    result: object
    next: str | None = None


# ---------------------------------------------------------------------------
# AuthStorageBackend 协议
# ---------------------------------------------------------------------------


@runtime_checkable
class AuthStorageBackend(Protocol):
    """认证存储后端协议。"""

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> object: ...

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> object: ...


# ---------------------------------------------------------------------------
# FileAuthStorageBackend
# ---------------------------------------------------------------------------


class FileAuthStorageBackend:
    """文件存储后端。

    使用 ``aiofiles`` 进行异步文件操作，``orjson`` 进行序列化。
    """

    def __init__(self, auth_path: Path | None = None) -> None:
        self._auth_path = auth_path or get_auth_path()

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> object:
        """同步锁定并执行操作。"""
        self._ensure_parent_dir()
        self._ensure_file_exists()

        current = self._read_file()
        lock_result = fn(current)
        if lock_result.next is not None:
            self._write_file(lock_result.next)
        return lock_result.result

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> object:
        """异步锁定并执行操作。"""

        self._ensure_parent_dir()
        self._ensure_file_exists()

        current = await self._read_file_async()
        lock_result = await fn(current)
        if lock_result.next is not None:
            await self._write_file_async(lock_result.next)
        return lock_result.result

    def _ensure_parent_dir(self) -> None:
        self._auth_path.parent.mkdir(parents=True, exist_ok=True)

    def _ensure_file_exists(self) -> None:
        if not self._auth_path.exists():
            self._auth_path.write_bytes(b"{}")
            self._auth_path.chmod(0o600)

    def _read_file(self) -> str | None:
        try:
            return self._auth_path.read_text("utf-8")
        except FileNotFoundError:
            return None

    def _write_file(self, content: str) -> None:
        self._auth_path.write_text(content, "utf-8")
        self._auth_path.chmod(0o600)

    async def _read_file_async(self) -> str | None:
        import aiofiles

        try:
            async with aiofiles.open(self._auth_path, mode="r", encoding="utf-8") as f:
                return await f.read()
        except FileNotFoundError:
            return None

    async def _write_file_async(self, content: str) -> None:
        import aiofiles

        async with aiofiles.open(self._auth_path, mode="w", encoding="utf-8") as f:
            await f.write(content)
        self._auth_path.chmod(0o600)


# ---------------------------------------------------------------------------
# InMemoryAuthStorageBackend
# ---------------------------------------------------------------------------


class InMemoryAuthStorageBackend:
    """内存存储后端。"""

    def __init__(self) -> None:
        self._value: str | None = None

    def with_lock(self, fn: Callable[[str | None], LockResult]) -> object:
        lock_result = fn(self._value)
        if lock_result.next is not None:
            self._value = lock_result.next
        return lock_result.result

    async def with_lock_async(
        self,
        fn: Callable[[str | None], Awaitable[LockResult]],
        options: AuthOperationOptions | None = None,
    ) -> object:
        lock_result = await fn(self._value)
        if lock_result.next is not None:
            self._value = lock_result.next
        return lock_result.result


# ---------------------------------------------------------------------------
# AuthStorage
# ---------------------------------------------------------------------------


class AuthStorage(CredentialStore):
    """凭据存储。

    实现 ``CredentialStore`` 接口，使用 ``AuthStorageBackend`` 进行实际存储。
    """

    def __init__(
        self,
        storage: AuthStorageBackend,
        auth_path: Path | None = None,
    ) -> None:
        self._storage = storage
        self._auth_path = auth_path
        self._data: dict[str, Credential] = {}

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, auth_path: Path | None = None) -> AuthStorage:
        """从文件创建。"""
        resolved = auth_path or get_auth_path()
        return cls(FileAuthStorageBackend(resolved), resolved)

    @classmethod
    def from_storage(cls, storage: AuthStorageBackend) -> AuthStorage:
        """从任意后端创建。"""
        return cls(storage)

    @classmethod
    def in_memory(cls, data: dict[str, Credential] | None = None) -> AuthStorage:
        """创建纯内存实例。"""
        storage = InMemoryAuthStorageBackend()
        if data:
            raw = orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
            storage.with_lock(lambda _: LockResult(None, raw))
        return cls.from_storage(storage)

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_data(content: str | None) -> dict[str, Credential]:
        if not content:
            return {}
        try:
            data = orjson.loads(content)
            if isinstance(data, dict):
                return data
        except orjson.JSONDecodeError:
            pass
        return {}

    def _update_data(self, data: dict[str, Credential]) -> None:
        self._data = data

    # ------------------------------------------------------------------
    # CredentialStore 接口
    # ------------------------------------------------------------------

    def reload(self) -> None:
        """从存储重新加载凭据。"""
        try:
            captured: list[str | None] = [None]

            def _read(current: str | None) -> LockResult:
                captured[0] = current
                return LockResult(None)

            self._storage.with_lock(_read)
            self._update_data(self._parse_data(captured[0]))
        except (OSError, orjson.JSONDecodeError):
            pass

    async def read(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        """读取 provider 的凭据。"""
        captured: list[dict[str, Credential]] = [{}]

        async def _read(current: str | None) -> LockResult:
            captured[0] = self._parse_data(current)
            return LockResult(None)

        await self._storage.with_lock_async(_read, options)
        self._update_data(captured[0])
        return self._data.get(provider_id)

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        """修改 provider 的凭据。"""
        captured_result: list[Credential | None] = [None]
        captured_data: list[dict[str, Credential]] = [{}]

        async def _modify(current: str | None) -> LockResult:
            data = self._parse_data(current)
            captured_data[0] = data
            next_val = await fn(data.get(provider_id))
            captured_result[0] = next_val
            if next_val is None:
                return LockResult(data.get(provider_id))
            merged: dict[str, Credential] = {**data, provider_id: next_val}
            captured_data[0] = merged
            return LockResult(
                next_val,
                orjson.dumps(merged, option=orjson.OPT_INDENT_2).decode("utf-8"),
            )

        await self._storage.with_lock_async(_modify, options)
        self._update_data(captured_data[0])
        return captured_result[0]

    async def delete(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        """删除 provider 的凭据。"""
        captured_data: list[dict[str, Credential]] = [{}]

        async def _delete(current: str | None) -> LockResult:
            data = self._parse_data(current)
            data.pop(provider_id, None)
            captured_data[0] = data
            return LockResult(
                None, orjson.dumps(data, option=orjson.OPT_INDENT_2).decode("utf-8")
            )

        await self._storage.with_lock_async(_delete, options)
        self._update_data(captured_data[0])

    async def list(
        self,
        options: AuthOperationOptions | None = None,
    ) -> list[CredentialInfo]:
        """列出所有凭据元信息。"""
        captured_data: list[dict[str, Credential]] = [{}]

        async def _list(current: str | None) -> LockResult:
            captured_data[0] = self._parse_data(current)
            return LockResult(None)

        await self._storage.with_lock_async(_list, options)
        self._update_data(captured_data[0])
        return [
            CredentialInfo(provider_id=pid, type=cred["type"])
            for pid, cred in self._data.items()
        ]


# ---------------------------------------------------------------------------
# read_stored_credential
# ---------------------------------------------------------------------------


def read_stored_credential(
    provider_id: str,
    auth_path: Path | None = None,
) -> Credential | None:
    """一次性同步读取存储的凭据。"""
    path = auth_path or get_auth_path()
    try:
        raw = path.read_bytes()
        data = orjson.loads(raw)
        if isinstance(data, dict):
            return data.get(provider_id)
    except (FileNotFoundError, orjson.JSONDecodeError):
        pass
    return None


__all__ = [
    "AuthStorage",
    "AuthStorageBackend",
    "FileAuthStorageBackend",
    "InMemoryAuthStorageBackend",
    "LockResult",
    "read_stored_credential",
]
