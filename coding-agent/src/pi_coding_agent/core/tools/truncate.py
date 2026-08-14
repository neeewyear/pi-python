"""Shared truncation utilities for tool outputs.

Truncation is based on two independent limits - whichever is hit first wins:
- Line limit (default: 2000 lines)
- Byte limit (default: 50KB)

Never returns partial lines (except bash tail truncation edge case).
"""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500


class TruncationResult(BaseModel):
    """Truncation result."""

    content: str
    truncated: bool
    truncated_by: str | None = None
    total_lines: int
    total_bytes: int
    output_lines: int
    output_bytes: int
    last_line_partial: bool = False
    first_line_exceeds_limit: bool = False
    max_lines: int
    max_bytes: int


class TruncationOptions(BaseModel):
    """Truncation options."""

    max_lines: int | None = None
    max_bytes: int | None = None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _utf8_byte_length(content: str) -> int:
    return len(content.encode("utf-8"))


def _split_lines_for_counting(content: str) -> list[str]:
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    if max_bytes <= 0:
        return ""

    output_bytes = 0
    start = len(text)

    i = len(text)
    while i > 0:
        char_start = i - 1
        code = ord(text[char_start])
        char_bytes: int

        if 0xDC00 <= code <= 0xDFFF and char_start > 0:
            previous = ord(text[char_start - 1])
            if 0xD800 <= previous <= 0xDBFF:
                char_start -= 1
                char_bytes = 4
            else:
                char_bytes = 3
        elif 0xD800 <= code <= 0xDFFF:
            char_bytes = 3
        elif code <= 0x7F:
            char_bytes = 1
        elif code <= 0x7FF:
            char_bytes = 2
        else:
            char_bytes = 3

        if output_bytes + char_bytes > max_bytes:
            break
        output_bytes += char_bytes
        start = char_start
        i = char_start

    return text[start:]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def format_size(byte_count: int) -> str:
    """Format bytes as human-readable size."""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def truncate_head(content: str, options: TruncationOptions | None = None) -> TruncationResult:
    """Truncate from the head (keep first N lines/bytes).

    Suitable for file reads where you want to see the beginning.
    Never returns partial lines. If first line exceeds byte limit,
    returns empty content with first_line_exceeds_limit=True.
    """
    opts = options or TruncationOptions()
    max_lines = opts.max_lines if opts.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = opts.max_bytes if opts.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = _utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    # No truncation needed
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # First line alone exceeds byte limit
    first_line_bytes = _utf8_byte_length(lines[0])
    if first_line_bytes > max_bytes:
        return TruncationResult(
            content="",
            truncated=True,
            truncated_by="bytes",
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=0,
            output_bytes=0,
            last_line_partial=False,
            first_line_exceeds_limit=True,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Collect complete lines that fit
    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"

    for i in range(min(len(lines), max_lines)):
        line = lines[i]
        line_bytes = _utf8_byte_length(line) + (1 if i > 0 else 0)  # +1 for newline

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            break

        output_lines_arr.append(line)
        output_bytes_count += line_bytes

    # Exited due to line limit
    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = _utf8_byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=False,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_tail(content: str, options: TruncationOptions | None = None) -> TruncationResult:
    """Truncate from the tail (keep last N lines/bytes).

    Suitable for bash output where you want to see the end.
    May return partial first line if the last line of original content exceeds byte limit.
    """
    opts = options or TruncationOptions()
    max_lines = opts.max_lines if opts.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = opts.max_bytes if opts.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = _utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    # No truncation needed
    if total_lines <= max_lines and total_bytes <= max_bytes:
        return TruncationResult(
            content=content,
            truncated=False,
            truncated_by=None,
            total_lines=total_lines,
            total_bytes=total_bytes,
            output_lines=total_lines,
            output_bytes=total_bytes,
            last_line_partial=False,
            first_line_exceeds_limit=False,
            max_lines=max_lines,
            max_bytes=max_bytes,
        )

    # Work backwards from the end
    output_lines_arr: list[str] = []
    output_bytes_count = 0
    truncated_by: str = "lines"
    last_line_partial = False

    i = len(lines) - 1
    while i >= 0 and len(output_lines_arr) < max_lines:
        line = lines[i]
        line_bytes = _utf8_byte_length(line) + (1 if output_lines_arr else 0)  # +1 for newline

        if output_bytes_count + line_bytes > max_bytes:
            truncated_by = "bytes"
            # Edge case: if we haven't added ANY lines yet and this line exceeds maxBytes,
            # take the end of the line (partial)
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = _utf8_byte_length(truncated_line)
                last_line_partial = True
            break

        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes
        i -= 1

    # Exited due to line limit
    if len(output_lines_arr) >= max_lines and output_bytes_count <= max_bytes:
        truncated_by = "lines"

    output_content = "\n".join(output_lines_arr)
    final_output_bytes = _utf8_byte_length(output_content)

    return TruncationResult(
        content=output_content,
        truncated=True,
        truncated_by=truncated_by,
        total_lines=total_lines,
        total_bytes=total_bytes,
        output_lines=len(output_lines_arr),
        output_bytes=final_output_bytes,
        last_line_partial=last_line_partial,
        first_line_exceeds_limit=False,
        max_lines=max_lines,
        max_bytes=max_bytes,
    )


def truncate_line(line: str, max_chars: int = GREP_MAX_LINE_LENGTH) -> tuple[str, bool]:
    """Truncate a single line to max characters, adding [truncated] suffix.

    Used for grep match lines.
    Returns (text, was_truncated).
    """
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True