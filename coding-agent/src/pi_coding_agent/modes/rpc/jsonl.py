"""JSONL (JSON Lines) transport for RPC mode.

Provides strict JSONL framing with LF (``\\n``) as the only record delimiter.
This intentionally does not use ``asyncio.StreamReader.readline`` because
readline splits on additional Unicode separators that are valid inside JSON
strings.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from asyncio import StreamReader, StreamWriter


def serialize_json_line(value: object) -> str:
    """Serialize a single strict JSONL record.

    Framing is LF-only. Payload strings may contain other Unicode separators
    such as U+2028 and U+2029. Clients must split records on ``\\n`` only.
    """
    return f"{json.dumps(value, ensure_ascii=False)}\n"


class JsonlReader:
    """Strict JSONL reader that splits on ``\\n`` only.

    Usage::

        reader = JsonlReader(stream_reader)
        async for line in reader:
            data = json.loads(line)
    """

    def __init__(self, stream: StreamReader) -> None:
        self._stream = stream
        self._buffer = ""

    async def read_line(self) -> str | None:
        """Read a single JSONL line from the stream.

        Returns ``None`` when the stream is exhausted.
        """
        while True:
            newline_index = self._buffer.find("\n")
            if newline_index >= 0:
                line = self._buffer[:newline_index]
                self._buffer = self._buffer[newline_index + 1 :]
                # Strip trailing CR if present
                if line.endswith("\r"):
                    line = line[:-1]
                return line

            chunk = await self._stream.read(4096)
            if not chunk:
                # End of stream; emit remaining buffer
                if self._buffer:
                    remaining = self._buffer
                    self._buffer = ""
                    if remaining.endswith("\r"):
                        remaining = remaining[:-1]
                    return remaining
                return None

            self._buffer += chunk.decode("utf-8") if isinstance(chunk, bytes) else chunk

    def __aiter__(self) -> JsonlReader:
        return self

    async def __anext__(self) -> str:
        line = await self.read_line()
        if line is None:
            raise StopAsyncIteration
        return line


class JsonlWriter:
    """Strict JSONL writer that writes LF-terminated records."""

    def __init__(self, stream: StreamWriter) -> None:
        self._stream = stream

    def write(self, value: object) -> None:
        """Serialize and write a JSONL record (non-blocking)."""
        data = serialize_json_line(value)
        self._stream.write(data.encode("utf-8"))

    async def drain(self) -> None:
        """Wait for the write buffer to drain."""
        await self._stream.drain()

    def write_line(self, line: str) -> None:
        """Write a pre-serialized line (must end with ``\\n``)."""
        self._stream.write(line.encode("utf-8"))

    async def write_and_drain(self, value: object) -> None:
        """Serialize, write, and drain."""
        self.write(value)
        await self.drain()