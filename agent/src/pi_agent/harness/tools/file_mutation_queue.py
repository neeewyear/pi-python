"""文件变更队列。

按 canonical path 串行化对同一执行环境的文件变更，避免并发写冲突。
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Awaitable
from typing import TypeVar

from ...result import get_or_throw
from ..types import ExecutionEnv

T = TypeVar("T")


class _MutationQueueState:
    """文件变更队列状态。"""

    __slots__ = ("queues", "registration")

    def __init__(self) -> None:
        self.queues: dict[str, asyncio.Future[None]] = {}
        self.registration: asyncio.Future[None] = _resolved_future()


def _resolved_future() -> asyncio.Future[None]:
    f: asyncio.Future[None] = asyncio.Future()
    f.set_result(None)
    return f


# 使用 id(env) 作为 WeakMap 的替代
_states: dict[int, _MutationQueueState] = {}


def _get_state(env: ExecutionEnv) -> _MutationQueueState:
    eid = id(env)
    state = _states.get(eid)
    if state is None:
        state = _MutationQueueState()
        _states[eid] = state
    return state


async def _get_mutation_queue_key(env: ExecutionEnv, path: str) -> str:
    absolute_path = get_or_throw(await env.absolute_path(path))
    canonical_result = await env.canonical_path(absolute_path)
    if canonical_result.is_ok():
        return canonical_result.value
    if canonical_result.error.code in ("not_found", "not_supported"):
        return absolute_path
    raise canonical_result.error


async def with_file_mutation_queue(
    env: ExecutionEnv,
    path: str,
    fn: Callable[[], Awaitable[T]],
) -> T:
    """按 canonical path 串行化文件变更。"""
    state = _get_state(env)

    async def _register() -> dict[str, object]:
        key = await _get_mutation_queue_key(env, path)
        current_queue = state.queues.get(key)
        current_future = current_queue if current_queue is not None else _resolved_future()

        next_queue: asyncio.Future[None] = asyncio.Future()
        chained = _chain_future(current_future, next_queue)
        state.queues[key] = chained
        return {
            "key": key,
            "current_queue": current_future,
            "chained_queue": chained,
            "release_next": next_queue,
        }

    # 串行化 registration
    prev_registration = state.registration
    registration_future: asyncio.Future[dict[str, object]] = asyncio.ensure_future(
        _chain_and_register(prev_registration, _register)
    )
    state.registration = _swallow_future(registration_future)  # type: ignore[arg-type]

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
        if state.queues.get(key) is chained:
            del state.queues[key]


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
    """创建链式 Future：current 完成后 next_future 完成。"""
    f: asyncio.Future[None] = asyncio.ensure_future(_chain_impl(current, next_future))
    return f


async def _chain_impl(current: asyncio.Future[None], next_future: asyncio.Future[None]) -> None:
    await current
    await next_future


def _swallow_future(f: asyncio.Future[object]) -> asyncio.Future[None]:
    """创建忽略结果/异常的 Future。"""
    sf: asyncio.Future[None] = asyncio.Future()

    def _done(_f: asyncio.Future[object]) -> None:
        if not sf.done():
            sf.set_result(None)

    f.add_done_callback(_done)
    return sf