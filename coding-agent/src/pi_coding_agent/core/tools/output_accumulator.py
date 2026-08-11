"""Output accumulator for streaming tool output.

Incrementally tracks streaming output with bounded memory.
Appends decoded chunks, keeps only a decoded tail for display snapshots,
and opens a temp file when the full output needs to be preserved.
"""

from __future__ import annotations

import os
import tempfile
import uuid

from .truncate import (
    DEFAULT_MAX_BYTES,
    DEFAULT_MAX_LINES,
    TruncationOptions,
    TruncationResult,
    truncate_tail,
)


class OutputAccumulatorOptions:
    """Options for the OutputAccumulator."""

    def __init__(
        self,
        max_lines: int | None = None,
        max_bytes: int | None = None,
        temp_file_prefix: str | None = None,
    ) -> None:
        self.max_lines = max_lines if max_lines is not None else DEFAULT_MAX_LINES
        self.max_bytes = max_bytes if max_bytes is not None else DEFAULT_MAX_BYTES
        self.temp_file_prefix = temp_file_prefix or "pi-output"


class OutputSnapshot:
    """A snapshot of the accumulated output."""

    def __init__(
        self,
        content: str,
        truncation: TruncationResult,
        full_output_path: str | None = None,
    ) -> None:
        self.content = content
        self.truncation = truncation
        self.full_output_path = full_output_path


class OutputAccumulator:
    """Incrementally tracks streaming output with bounded memory.

    Appends decoded chunks, keeps only a decoded tail for display snapshots,
    and opens a temp file when the full output needs to be preserved.
    """

    def __init__(self, options: OutputAccumulatorOptions | None = None) -> None:
        opts = options or OutputAccumulatorOptions()
        self._max_lines = opts.max_lines
        self._max_bytes = opts.max_bytes
        self._max_rolling_bytes = max(self._max_bytes * 2, 1)
        self._temp_file_prefix = opts.temp_file_prefix

        self._tail_text = ""
        self._tail_bytes = 0
        self._tail_starts_at_line_boundary = True
        self._total_decoded_bytes = 0
        self._completed_lines = 0
        self._total_lines = 0
        self._current_line_bytes = 0
        self._has_open_line = False
        self._finished = False

        self._temp_file_path: str | None = None
        self._temp_file: object = None  # Will be set to a file handle if needed
        self._raw_chunks: list[bytes] = []

    def append(self, data: bytes) -> None:
        """Append a chunk of raw bytes."""
        if self._finished:
            raise RuntimeError("Cannot append to a finished output accumulator")

        text = data.decode("utf-8", errors="replace")
        self._append_decoded_text(text)

        if self._temp_file_path is not None or self._should_use_temp_file():
            self._ensure_temp_file()
            if self._temp_file is not None:
                import io

                handle = self._temp_file
                if isinstance(handle, io.TextIOBase):
                    handle.write(text)
        elif data:
            self._raw_chunks.append(data)

    def finish(self) -> None:
        """Mark the output as finished."""
        if self._finished:
            return
        self._finished = True
        if self._should_use_temp_file():
            self._ensure_temp_file()

    def snapshot(self, persist_if_truncated: bool = False) -> OutputSnapshot:
        """Get a snapshot of the current output."""
        tail_truncation = truncate_tail(
            self._get_snapshot_text(),
            TruncationOptions(max_lines=self._max_lines, max_bytes=self._max_bytes),
        )
        truncated = (
            self._total_lines > self._max_lines
            or self._total_decoded_bytes > self._max_bytes
        )
        truncated_by = (
            tail_truncation.truncated_by
            if truncated
            else (
                "bytes"
                if self._total_decoded_bytes > self._max_bytes
                else "lines"
                if self._total_lines > self._max_lines
                else None
            )
        )
        truncation = TruncationResult(
            content=tail_truncation.content,
            truncated=truncated,
            truncated_by=truncated_by,
            total_lines=self._total_lines,
            total_bytes=self._total_decoded_bytes,
            output_lines=tail_truncation.output_lines,
            output_bytes=tail_truncation.output_bytes,
            last_line_partial=tail_truncation.last_line_partial,
            first_line_exceeds_limit=tail_truncation.first_line_exceeds_limit,
            max_lines=self._max_lines,
            max_bytes=self._max_bytes,
        )

        if persist_if_truncated and truncation.truncated:
            self._ensure_temp_file()

        return OutputSnapshot(
            content=truncation.content,
            truncation=truncation,
            full_output_path=self._temp_file_path,
        )

    async def close_temp_file(self) -> None:
        """Close the temp file if open."""
        if self._temp_file is None:
            return
        import io

        handle = self._temp_file
        self._temp_file = None
        if isinstance(handle, io.TextIOBase):
            handle.close()

    def get_last_line_bytes(self) -> int:
        """Get the byte count of the current (last) line."""
        return self._current_line_bytes

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _append_decoded_text(self, text: str) -> None:
        if not text:
            return

        bytes_len = len(text.encode("utf-8"))
        self._total_decoded_bytes += bytes_len
        self._tail_text += text
        self._tail_bytes += bytes_len
        if self._tail_bytes > self._max_rolling_bytes * 2:
            self._trim_tail()

        newlines = text.count("\n")
        last_newline = text.rfind("\n")
        if newlines == 0:
            self._current_line_bytes += bytes_len
            self._has_open_line = True
        else:
            self._completed_lines += newlines
            tail = text[last_newline + 1 :]
            self._current_line_bytes = len(tail.encode("utf-8"))
            self._has_open_line = bool(tail)
        self._total_lines = self._completed_lines + (1 if self._has_open_line else 0)

    def _trim_tail(self) -> None:
        encoded = self._tail_text.encode("utf-8")
        if len(encoded) <= self._max_rolling_bytes:
            self._tail_bytes = len(encoded)
            return

        start = len(encoded) - self._max_rolling_bytes
        while start < len(encoded) and (encoded[start] & 0xC0) == 0x80:
            start += 1

        self._tail_starts_at_line_boundary = (
            self._tail_starts_at_line_boundary
            if start == 0
            else encoded[start - 1] == 0x0A
        )
        self._tail_text = encoded[start:].decode("utf-8", errors="replace")
        self._tail_bytes = len(self._tail_text.encode("utf-8"))

    def _get_snapshot_text(self) -> str:
        if self._tail_starts_at_line_boundary:
            return self._tail_text

        first_newline = self._tail_text.find("\n")
        return (
            self._tail_text
            if first_newline == -1
            else self._tail_text[first_newline + 1 :]
        )

    def _should_use_temp_file(self) -> bool:
        return (
            self._total_decoded_bytes > self._max_bytes
            or self._total_lines > self._max_lines
        )

    def _ensure_temp_file(self) -> None:
        if self._temp_file_path is not None:
            return
        suffix = f"-{uuid.uuid4().hex[:8]}.log"
        self._temp_file_path = os.path.join(
            tempfile.gettempdir(), f"{self._temp_file_prefix}{suffix}"
        )
        handle = open(self._temp_file_path, "w", encoding="utf-8")
        handle.writelines(
            chunk.decode("utf-8", errors="replace") for chunk in self._raw_chunks
        )
        self._raw_chunks = []
        self._temp_file = handle
