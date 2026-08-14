"""凭据存储实现"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import TypeVar

from ..utils.abort import operation_signal, race_with_abort_signal
from .types import (
    AuthOperationOptions,
    Credential,
    CredentialInfo,
    CredentialStore,
)

T = TypeVar("T")


class InMemoryCredentialStore(CredentialStore):
    """内存凭据存储（实现 ``CredentialStore`` 接口）。

    使用 ``_chains`` 字典维护每个 provider 的任务链，实现串行化写入。
    """

    def __init__(self) -> None:
        self._credentials: dict[str, Credential] = {}
        self._chains: dict[str, asyncio.Task[None]] = {}

    async def _enqueue(
        self,
        provider_id: str,
        task: Callable[[], Awaitable[T]],
        options: AuthOperationOptions | None = None,
    ) -> T:
        """串行化任务队列。

        等待前一个任务完成后执行当前任务，使用 ``CancellationToken`` 支持取消。
        """
        signal = operation_signal(options.get("signal") if options else None)
        previous = self._chains.get(provider_id)

        async def _run() -> T:
            if previous is not None:
                try:
                    await previous
                except Exception:
                    pass
            signal.throw_if_cancelled()
            return await task()

        queued = asyncio.ensure_future(_run())

        async def _tail() -> None:
            try:
                await queued
            except Exception:
                pass

        tail = asyncio.ensure_future(_tail())
        self._chains[provider_id] = tail
        tail.add_done_callback(lambda _: self._chains.pop(provider_id, None))

        # 使用 race_with_abort_signal 确保初始取消检查
        result = await race_with_abort_signal(queued, signal)
        return result  # type: ignore[return-value]

    async def read(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        signal = operation_signal(options.get("signal") if options else None)
        signal.throw_if_cancelled()
        return self._credentials.get(provider_id)

    async def list(
        self,
        options: AuthOperationOptions | None = None,
    ) -> list[CredentialInfo]:
        signal = operation_signal(options.get("signal") if options else None)
        signal.throw_if_cancelled()
        return [
            CredentialInfo(provider_id=pid, type=cred["type"])
            for pid, cred in self._credentials.items()
        ]

    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None:
        async def _inner() -> Credential | None:
            current = self._credentials.get(provider_id)
            next_val = await fn(current)
            sig = operation_signal(options.get("signal") if options else None)
            sig.throw_if_cancelled()
            if next_val is not None:
                self._credentials[provider_id] = next_val
            return next_val if next_val is not None else current

        return await self._enqueue(provider_id, _inner, options)

    async def delete(
        self,
        provider_id: str,
        options: AuthOperationOptions | None = None,
    ) -> None:
        async def _inner() -> None:
            self._credentials.pop(provider_id, None)

        await self._enqueue(provider_id, _inner, options)
