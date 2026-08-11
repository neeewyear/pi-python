"""共享文本截断工具（对应 ``harness/utils/truncate.ts``）。

截断基于两个独立上限——先触发者生效：
- 行数上限（默认 2000 行）
- 字节数上限（默认 50KB）

从不返回部分行（bash tail 截断的边缘情况除外）。
"""

from __future__ import annotations

from pydantic import BaseModel

DEFAULT_MAX_LINES = 2000
DEFAULT_MAX_BYTES = 50 * 1024  # 50KB
GREP_MAX_LINE_LENGTH = 500

_UNPAIRED_SURROGATE_REPLACEMENT = "\ufffd"


class TruncationResult(BaseModel):
    """截断结果（对应 TS ``TruncationResult``）。"""

    content: str
    """截断后的内容。"""
    truncated: bool
    """是否发生了截断。"""
    truncated_by: str | None = None
    """触发截断的上限：``"lines"`` / ``"bytes"`` / ``None``。"""
    total_lines: int
    """原始内容的总行数。"""
    total_bytes: int
    """原始内容的总字节数（UTF-8）。"""
    output_lines: int
    """截断后输出的完整行数。"""
    output_bytes: int
    """截断后输出的字节数（UTF-8）。"""
    last_line_partial: bool = False
    """最后一行是否被部分截断（仅 tail 截断的边缘情况）。"""
    first_line_exceeds_limit: bool = False
    """首行是否超过字节上限（仅 head 截断的边缘情况）。"""
    max_lines: int
    """应用的行数上限。"""
    max_bytes: int
    """应用的字节数上限。"""


class TruncationOptions(BaseModel):
    """截断选项（对应 TS ``TruncationOptions``）。"""

    max_lines: int | None = None
    """最大行数（默认 2000）。"""
    max_bytes: int | None = None
    """最大字节数（默认 50KB）。"""


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------

def _utf8_byte_length(content: str) -> int:
    """返回字符串的 UTF-8 字节长度。"""
    return len(content.encode("utf-8"))


def _split_lines_for_counting(content: str) -> list[str]:
    """按 ``\\n`` 分割，空字符串返回空列表，末尾 ``\\n`` 不产生空行。"""
    if not content:
        return []
    lines = content.split("\n")
    if content.endswith("\n"):
        lines.pop()
    return lines


def _replace_unpaired_surrogates(content: str) -> str:
    """替换孤立 surrogate 字符为 U+FFFD。

    Python 3 内部使用 UTF-8，但输入可能来自外部编码含孤立 surrogate。
    """
    result: list[str] = []
    i = 0
    while i < len(content):
        code = ord(content[i])
        if 0xD800 <= code <= 0xDBFF:
            if i + 1 < len(content):
                next_code = ord(content[i + 1])
                if 0xDC00 <= next_code <= 0xDFFF:
                    result.append(content[i])
                    result.append(content[i + 1])
                    i += 2
                    continue
            result.append(_UNPAIRED_SURROGATE_REPLACEMENT)
            i += 1
        elif 0xDC00 <= code <= 0xDFFF:
            result.append(_UNPAIRED_SURROGATE_REPLACEMENT)
            i += 1
        else:
            result.append(content[i])
            i += 1
    return "".join(result)


def _truncate_string_to_bytes_from_end(text: str, max_bytes: int) -> str:
    """从末尾截断字符串至不超过 ``max_bytes`` 字节，正确处理多字节 UTF-8 字符。"""
    if max_bytes <= 0:
        return ""

    output_bytes = 0
    start = len(text)
    needs_replacement = False

    i = len(text)
    while i > 0:
        char_start = i - 1
        code = ord(text[char_start])
        char_bytes: int
        unpaired_surrogate = False

        if 0xDC00 <= code <= 0xDFFF and char_start > 0:
            previous = ord(text[char_start - 1])
            if 0xD800 <= previous <= 0xDBFF:
                char_start -= 1
                char_bytes = 4
            else:
                char_bytes = 3
                unpaired_surrogate = True
        elif 0xD800 <= code <= 0xDFFF:
            char_bytes = 3
            unpaired_surrogate = True
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
        if unpaired_surrogate:
            needs_replacement = True
        i = char_start

    output = text[start:]
    return _replace_unpaired_surrogates(output) if needs_replacement else output


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------

def format_size(byte_count: int) -> str:
    """把字节数格式化为人类可读的大小。"""
    if byte_count < 1024:
        return f"{byte_count}B"
    if byte_count < 1024 * 1024:
        return f"{byte_count / 1024:.1f}KB"
    return f"{byte_count / (1024 * 1024):.1f}MB"


def truncate_head(content: str, options: TruncationOptions | None = None) -> TruncationResult:
    """从头截断（保留前 N 行/字节），适用于文件读取场景。

    从不返回部分行。如果首行超过字节上限，返回空内容并标记
    ``first_line_exceeds_limit=True``。
    """
    opts = options or TruncationOptions()
    max_lines = opts.max_lines if opts.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = opts.max_bytes if opts.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = _utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    # 无需截断
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

    # 首行单独超过字节上限
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

    # 收集完整行
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

    # 因行数上限退出
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
    """从尾截断（保留后 N 行/字节），适用于 bash 输出场景。

    可能返回部分行：如果原始内容的最后一行超过字节上限，取该行的末尾。
    """
    opts = options or TruncationOptions()
    max_lines = opts.max_lines if opts.max_lines is not None else DEFAULT_MAX_LINES
    max_bytes = opts.max_bytes if opts.max_bytes is not None else DEFAULT_MAX_BYTES

    total_bytes = _utf8_byte_length(content)
    lines = _split_lines_for_counting(content)
    total_lines = len(lines)

    # 无需截断
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

    # 从末尾反向遍历
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
            # 边缘情况：还未添加任何行，且当前行超过字节上限 → 取该行末尾部分
            if not output_lines_arr:
                truncated_line = _truncate_string_to_bytes_from_end(line, max_bytes)
                output_lines_arr.insert(0, truncated_line)
                output_bytes_count = _utf8_byte_length(truncated_line)
                last_line_partial = True
            break

        output_lines_arr.insert(0, line)
        output_bytes_count += line_bytes
        i -= 1

    # 因行数上限退出
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
    """截断单行到最多 ``max_chars`` 字符，追加 ``... [truncated]`` 后缀。

    用于 grep 匹配行。
    返回 ``(text, was_truncated)``。
    """
    if len(line) <= max_chars:
        return line, False
    return f"{line[:max_chars]}... [truncated]", True