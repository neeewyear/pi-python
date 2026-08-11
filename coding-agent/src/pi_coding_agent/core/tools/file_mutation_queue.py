"""File mutation queue for serializing file operations.

Serializes file mutation operations targeting the same file path.
Operations for different files still run in parallel.
"""

from __future__ import annotations

import asyncio
import os.path
from collections.abc import Awaitable, Callable
from typing import TypeVar

T = TypeVar("T")


def _resolved_future() -> asyncio.Future[None]:
    f: asyncio.Future[None] = asyncio.Future()
    f.set_result(None)
    return f


_file_mutation_queues: dict[str, asyncio.Future[None]] = {}
_registration_queue: asyncio.Future[None] = _resolved_future()


async def _get_mutation_queue_key(file_path: str) -> str:
    """Get the canonical path for queue keying."""
    resolved_path = os.path.normpath(os.path.expanduser(file_path))
    try:
        real_path = os.path.realpath(resolved_path)
        return real_path
    except OSError:
        return resolved_path


async def with_file_mutation_queue(file_path: str, fn: Callable[[], Awaitable[T]]) -> T:
    """Serialize file mutation operations targeting the same file.

    Operations for different files still run in parallel.
    """
    global _registration_queue

    async def _register() -> dict[str, object]:
        key = await _get_mutation_queue_key(file_path)
        current_queue = _file_mutation_queues.get(key)
        current_future = (
            current_queue if current_queue is not None else _resolved_future()
        )

        next_queue: asyncio.Future[None] = asyncio.Future()
        chained = _chain_future(current_future, next_queue)
        _file_mutation_queues[key] = chained
        return {
            "key": key,
            "current_queue": current_future,
            "chained_queue": chained,
            "release_next": next_queue,
        }

    # Serialize registration
    prev_registration = _registration_queue
    registration_future: asyncio.Future[dict[str, object]] = asyncio.ensure_future(
        _chain_and_register(prev_registration, _register)
    )
    _registration_queue = _swallow_future(registration_future)  # type: ignore[arg-type]

    reg = await registration_future
    current_future: asyncio.Future[None] = reg["current_queue"]  # type: ignore[assignment]
    key: str = reg["key"]  # type: ignore[assignment]
    chained: asyncio.Future[None] = reg["chained_queue"]  # type: ignore[assignment]
    release_next: asyncio.Future[None] = reg["release_next"]  # type: ignore[assignment]

    await current_future
    try:
        return await fn()
    finally:
        release_next.set_result(None)
        if _file_mutation_queues.get(key) is chained:
            del _file_mutation_queues[key]


async def _chain_and_register(
    prev: asyncio.Future[None],
    register_fn: Callable[[], Awaitable[dict[str, object]]],
) -> dict[str, object]:
    await prev
    return await register_fn()


def _chain_future(
    current: asyncio.Future[None],
    next_future: asyncio.Future[None],
) -> asyncio.Future[None]:
    f: asyncio.Future[None] = asyncio.ensure_future(_chain_impl(current, next_future))
    return f


async def _chain_impl(
    current: asyncio.Future[None], next_future: asyncio.Future[None]
) -> None:
    await current
    await next_future


def _swallow_future(f: asyncio.Future[object]) -> asyncio.Future[None]:
    sf: asyncio.Future[None] = asyncio.Future()

    def _done(_f: asyncio.Future[object]) -> None:
        if not sf.done():
            sf.set_result(None)

    f.add_done_callback(_done)
    return sf
