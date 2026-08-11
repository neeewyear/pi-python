from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable

RAW_STDOUT_RETRY_DELAY_MS = 0.01  # 10ms in seconds

_stdout_takeover_state: dict[str, object] | None = None
_raw_stdout_write_tail: asyncio.Future[None] = asyncio.Future()
_raw_stdout_write_tail.set_result(None)


def _get_raw_stdout_write() -> Callable[..., object]:
    global _stdout_takeover_state
    if _stdout_takeover_state:
        return _stdout_takeover_state["raw_stdout_write"]  # type: ignore
    return sys.stdout.write


async def _write_raw_stdout_chunk(text: str) -> None:
    while True:
        try:
            loop = asyncio.get_event_loop()
            future = loop.create_future()

            def _write() -> None:
                try:
                    _get_raw_stdout_write()(text)
                    future.set_result(None)
                except Exception as e:
                    future.set_exception(e)

            loop.call_soon(_write)
            await future
            return
        except OSError as e:
            if getattr(e, "errno", None) not in (None,):
                raise
            await asyncio.sleep(RAW_STDOUT_RETRY_DELAY_MS)


def take_over_stdout() -> None:
    global _stdout_takeover_state
    if _stdout_takeover_state:
        return

    raw_stdout_write = sys.stdout.write
    raw_stderr_write = sys.stderr.write
    original_stdout_write = sys.stdout.write

    def _redirected_write(chunk: str, /) -> int:
        return raw_stderr_write(chunk)

    sys.stdout.write = _redirected_write  # type: ignore

    _stdout_takeover_state = {
        "raw_stdout_write": raw_stdout_write,
        "raw_stderr_write": raw_stderr_write,
        "original_stdout_write": original_stdout_write,
    }


def restore_stdout() -> None:
    global _stdout_takeover_state
    if not _stdout_takeover_state:
        return

    sys.stdout.write = _stdout_takeover_state["original_stdout_write"]  # type: ignore
    _stdout_takeover_state = None


def is_stdout_taken_over() -> bool:
    return _stdout_takeover_state is not None


def write_raw_stdout(text: str) -> None:
    global _raw_stdout_write_tail
    if len(text) == 0:
        return

    new_tail: asyncio.Future[None] = asyncio.Future()

    def _do_write() -> None:
        _raw_stdout_write_tail.add_done_callback(
            lambda _: asyncio.ensure_future(
                _write_raw_stdout_chunk(text)
            ).add_done_callback(
                lambda f: new_tail.set_result(None) if not f.exception() else None
            )
        )

    loop = asyncio.get_event_loop()
    loop.call_soon(_do_write)
    _raw_stdout_write_tail = new_tail


async def wait_for_raw_stdout_backpressure() -> None:
    while True:
        tail = _raw_stdout_write_tail
        await tail
        if tail == _raw_stdout_write_tail:
            return


async def flush_raw_stdout() -> None:
    await wait_for_raw_stdout_backpressure()
    await _write_raw_stdout_chunk("")
