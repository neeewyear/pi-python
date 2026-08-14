"""Diff/Patch 核心算法。  

提供编辑工具所需的精确替换、模糊匹配、Unified Patch 生成与
diff 字符串格式化。
"""

from __future__ import annotations

import difflib
from typing import Literal

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class Edit(BaseModel):
    """一次编辑操作。"""

    old_text: str
    new_text: str


class FuzzyMatchResult(BaseModel):
    """模糊匹配结果。"""

    found: bool
    index: int
    match_length: int
    used_fuzzy_match: bool
    content_for_replacement: str


class AppliedEditsResult(BaseModel):
    """编辑应用结果。"""

    base_content: str
    new_content: str


# ---------------------------------------------------------------------------
# 行结尾处理
# ---------------------------------------------------------------------------


def detect_line_ending(content: str) -> Literal["\r\n", "\n"]:
    """检测行结尾风格。"""
    crlf_idx = content.find("\r\n")
    lf_idx = content.find("\n")
    if lf_idx == -1:
        return "\n"
    if crlf_idx == -1:
        return "\n"
    return "\r\n" if crlf_idx < lf_idx else "\n"


def normalize_to_lf(text: str) -> str:
    """统一为 LF 行结尾。"""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def restore_line_endings(text: str, ending: Literal["\r\n", "\n"]) -> str:
    """恢复原始行结尾。"""  
    return text.replace("\n", "\r\n") if ending == "\r\n" else text


# ---------------------------------------------------------------------------
# 模糊匹配
# ---------------------------------------------------------------------------


def normalize_for_fuzzy_match(text: str) -> str:
    """模糊匹配归一化。

    渐进式变换：
    - 每行去掉尾部空白
    - 智能引号 → ASCII 引号
    - Unicode 破折号/连字符 → ASCII 连字符
    - 特殊 Unicode 空格 → 常规空格
    """
    lines = [line.rstrip() for line in text.split("\n")]
    result = "\n".join(lines)
    # 智能单引号 → '
    result = result.translate(str.maketrans("\u2018\u2019\u201a\u201b", "''''"))
    # 智能双引号 → "
    result = result.translate(str.maketrans("\u201c\u201d\u201e\u201f", '""""'))
    # 各种破折号/连字符 → -
    result = result.translate(
        str.maketrans("\u2010\u2011\u2012\u2013\u2014\u2015\u2212", "-------")
    )
    # 特殊空格 → 常规空格
    special_spaces = (
        "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008\u2009\u200a\u202f\u205f\u3000"
    )
    result = result.translate(str.maketrans(special_spaces, " " * len(special_spaces)))
    return result


# ---------------------------------------------------------------------------
# 行跨度
# ---------------------------------------------------------------------------


def _split_lines_with_endings(content: str) -> list[str]:
    """按行分割，保留行结尾。"""
    lines: list[str] = []
    current: list[str] = []
    for ch in content:
        current.append(ch)
        if ch == "\n":
            lines.append("".join(current))
            current = []
    if current:
        lines.append("".join(current))
    return lines


class _LineSpan:
    __slots__ = ("end", "start")

    def __init__(self, start: int, end: int) -> None:
        self.start = start
        self.end = end


class _MatchedEdit:
    __slots__ = ("edit_index", "match_index", "match_length", "new_text")

    def __init__(
        self, edit_index: int, match_index: int, match_length: int, new_text: str
    ) -> None:
        self.edit_index = edit_index
        self.match_index = match_index
        self.match_length = match_length
        self.new_text = new_text


class _TextReplacement:
    __slots__ = ("match_index", "match_length", "new_text")

    def __init__(self, match_index: int, match_length: int, new_text: str) -> None:
        self.match_index = match_index
        self.match_length = match_length
        self.new_text = new_text


def _get_line_spans(content: str) -> list[_LineSpan]:
    offset = 0
    spans: list[_LineSpan] = []
    for line in _split_lines_with_endings(content):
        spans.append(_LineSpan(offset, offset + len(line)))
        offset = spans[-1].end
    return spans


def _get_replacement_line_range(
    lines: list[_LineSpan], replacement: _TextReplacement
) -> tuple[int, int]:
    replacement_start = replacement.match_index
    replacement_end = replacement.match_index + replacement.match_length

    start_line = -1
    for i, line in enumerate(lines):
        if replacement_start >= line.start and replacement_start < line.end:
            start_line = i
            break
    if start_line == -1:
        raise ValueError("Replacement range is outside the base content.")

    end_line = start_line
    while end_line < len(lines) and lines[end_line].end < replacement_end:
        end_line += 1
    if end_line >= len(lines):
        raise ValueError("Replacement range is outside the base content.")

    return start_line, end_line + 1


def _apply_replacements(
    content: str, replacements: list[_TextReplacement], offset: int = 0
) -> str:
    result = content
    for replacement in reversed(replacements):
        match_index = replacement.match_index - offset
        result = (
            result[:match_index]
            + replacement.new_text
            + result[match_index + replacement.match_length :]
        )
    return result


def apply_replacements_preserving_unchanged_lines(
    original_content: str,
    base_content: str,
    replacements: list[_TextReplacement],
) -> str:
    """在保留未变更行的前提下应用替换。"""
    original_lines = _split_lines_with_endings(original_content)
    base_line_spans = _get_line_spans(base_content)

    if len(original_lines) != len(base_line_spans):
        raise ValueError(
            "Cannot preserve unchanged lines because the base content has a different line count."
        )

    # 按 start_line 分组
    sorted_replacements = sorted(replacements, key=lambda r: r.match_index)
    groups: list[tuple[int, int, list[_TextReplacement]]] = []
    for replacement in sorted_replacements:
        start_line, end_line = _get_replacement_line_range(base_line_spans, replacement)
        if groups and start_line < groups[-1][1]:
            prev_start, prev_end, prev_reps = groups[-1]
            groups[-1] = (
                prev_start,
                max(prev_end, end_line),
                prev_reps + [replacement],
            )
        else:
            groups.append((start_line, end_line, [replacement]))

    original_line_index = 0
    result_parts: list[str] = []
    for group_start, group_end, group_reps in groups:
        result_parts.append("".join(original_lines[original_line_index:group_start]))

        group_start_offset = base_line_spans[group_start].start
        group_end_offset = base_line_spans[group_end - 1].end
        result_parts.append(
            _apply_replacements(
                base_content[group_start_offset:group_end_offset],
                group_reps,
                group_start_offset,
            )
        )
        original_line_index = group_end
    result_parts.append("".join(original_lines[original_line_index:]))

    return "".join(result_parts)


# ---------------------------------------------------------------------------
# 模糊查找
# ---------------------------------------------------------------------------


def fuzzy_find_text(content: str, old_text: str) -> FuzzyMatchResult:
    """在 content 中查找 old_text，先精确匹配再模糊匹配。"""
    exact_index = content.find(old_text)
    if exact_index != -1:
        return FuzzyMatchResult(
            found=True,
            index=exact_index,
            match_length=len(old_text),
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    fuzzy_index = fuzzy_content.find(fuzzy_old_text)

    if fuzzy_index == -1:
        return FuzzyMatchResult(
            found=False,
            index=-1,
            match_length=0,
            used_fuzzy_match=False,
            content_for_replacement=content,
        )

    return FuzzyMatchResult(
        found=True,
        index=fuzzy_index,
        match_length=len(fuzzy_old_text),
        used_fuzzy_match=True,
        content_for_replacement=fuzzy_content,
    )


# ---------------------------------------------------------------------------
# BOM 处理
# ---------------------------------------------------------------------------


def strip_bom(content: str) -> dict[str, str]:
    """去除 UTF-8 BOM。"""
    if content.startswith("\ufeff"):
        return {"bom": "\ufeff", "text": content[1:]}
    return {"bom": "", "text": content}


# ---------------------------------------------------------------------------
# 出现次数统计
# ---------------------------------------------------------------------------


def _count_occurrences(content: str, old_text: str) -> int:
    fuzzy_content = normalize_for_fuzzy_match(content)
    fuzzy_old_text = normalize_for_fuzzy_match(old_text)
    return fuzzy_content.count(fuzzy_old_text)


# ---------------------------------------------------------------------------
# 错误消息
# ---------------------------------------------------------------------------


def _get_not_found_error(path: str, edit_index: int, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Could not find the exact text in {path}. "
            "The old text must match exactly including all whitespace and newlines."
        )
    return ValueError(
        f"Could not find edits[{edit_index}] in {path}. "
        "The oldText must match exactly including all whitespace and newlines."
    )


def _get_duplicate_error(
    path: str, edit_index: int, total_edits: int, occurrences: int
) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"Found {occurrences} occurrences of the text in {path}. "
            "The text must be unique. Please provide more context to make it unique."
        )
    return ValueError(
        f"Found {occurrences} occurrences of edits[{edit_index}] in {path}. "
        "Each oldText must be unique. Please provide more context to make it unique."
    )


def _get_empty_old_text_error(
    path: str, edit_index: int, total_edits: int
) -> ValueError:
    if total_edits == 1:
        return ValueError(f"oldText must not be empty in {path}.")
    return ValueError(f"edits[{edit_index}].oldText must not be empty in {path}.")


def _get_no_change_error(path: str, total_edits: int) -> ValueError:
    if total_edits == 1:
        return ValueError(
            f"No changes made to {path}. The replacement produced identical content. "
            "This might indicate an issue with special characters or the text not existing as expected."
        )
    return ValueError(
        f"No changes made to {path}. The replacements produced identical content."
    )


# ---------------------------------------------------------------------------
# 编辑应用
# ---------------------------------------------------------------------------


def apply_edits_to_normalized_content(
    normalized_content: str,
    edits: list[Edit],
    path: str,
) -> AppliedEditsResult:
    """对 LF 归一化内容应用一组精确替换。"""
    normalized_edits = [
        Edit(old_text=normalize_to_lf(e.old_text), new_text=normalize_to_lf(e.new_text))
        for e in edits
    ]

    for i, edit in enumerate(normalized_edits):
        if len(edit.old_text) == 0:
            raise _get_empty_old_text_error(path, i, len(normalized_edits))

    initial_matches = [
        fuzzy_find_text(normalized_content, edit.old_text) for edit in normalized_edits
    ]
    used_fuzzy_match = any(m.used_fuzzy_match for m in initial_matches)
    replacement_base_content = (
        normalize_for_fuzzy_match(normalized_content)
        if used_fuzzy_match
        else normalized_content
    )

    matched_edits: list[_MatchedEdit] = []
    for i, edit in enumerate(normalized_edits):
        match_result = fuzzy_find_text(replacement_base_content, edit.old_text)
        if not match_result.found:
            raise _get_not_found_error(path, i, len(normalized_edits))

        occurrences = _count_occurrences(replacement_base_content, edit.old_text)
        if occurrences > 1:
            raise _get_duplicate_error(path, i, len(normalized_edits), occurrences)

        matched_edits.append(
            _MatchedEdit(
                edit_index=i,
                match_index=match_result.index,
                match_length=match_result.match_length,
                new_text=edit.new_text,
            )
        )

    matched_edits.sort(key=lambda m: m.match_index)
    for i in range(1, len(matched_edits)):
        prev = matched_edits[i - 1]
        curr = matched_edits[i]
        if prev.match_index + prev.match_length > curr.match_index:
            raise ValueError(
                f"edits[{prev.edit_index}] and edits[{curr.edit_index}] overlap in {path}. "
                "Merge them into one edit or target disjoint regions."
            )

    base_content = normalized_content
    replacements = [
        _TextReplacement(m.match_index, m.match_length, m.new_text)
        for m in matched_edits
    ]

    new_content = (
        apply_replacements_preserving_unchanged_lines(
            normalized_content, replacement_base_content, replacements
        )
        if used_fuzzy_match
        else _apply_replacements(replacement_base_content, replacements)
    )

    if base_content == new_content:
        raise _get_no_change_error(path, len(normalized_edits))

    return AppliedEditsResult(base_content=base_content, new_content=new_content)


# ---------------------------------------------------------------------------
# Unified Patch
# ---------------------------------------------------------------------------


def generate_unified_patch(
    path: str,
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> str:
    """生成标准 unified patch。"""
    diff = difflib.unified_diff(
        old_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=path,
        tofile=path,
        n=context_lines,
    )
    return "".join(diff)


# ---------------------------------------------------------------------------
# Diff 字符串
# ---------------------------------------------------------------------------


def generate_diff_string(
    old_content: str,
    new_content: str,
    context_lines: int = 4,
) -> dict[str, object]:
    """生成带行号的显示用 diff 字符串。"""
    old_lines = old_content.split("\n")
    new_lines = new_content.split("\n")

    seq_matcher = difflib.SequenceMatcher(None, old_lines, new_lines)
    opcodes = seq_matcher.get_opcodes()

    max_line_num = max(len(old_lines), len(new_lines))
    line_num_width = len(str(max_line_num))
    output: list[str] = []
    first_changed_line: int | None = None

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            # 上下文行
            for k in range(i2 - i1):
                line_num = str(i1 + k + 1).rjust(line_num_width)
                output.append(f" {line_num} {old_lines[i1 + k]}")
            continue

        # 捕获第一个变更行
        if first_changed_line is None:
            first_changed_line = j1 + 1

        if tag == "replace":
            for k in range(i2 - i1):
                line_num = str(i1 + k + 1).rjust(line_num_width)
                output.append(f"-{line_num} {old_lines[i1 + k]}")
            for k in range(j2 - j1):
                line_num = str(j1 + k + 1).rjust(line_num_width)
                output.append(f"+{line_num} {new_lines[j1 + k]}")
        elif tag == "delete":
            for k in range(i2 - i1):
                line_num = str(i1 + k + 1).rjust(line_num_width)
                output.append(f"-{line_num} {old_lines[i1 + k]}")
        elif tag == "insert":
            for k in range(j2 - j1):
                line_num = str(j1 + k + 1).rjust(line_num_width)
                output.append(f"+{line_num} {new_lines[j1 + k]}")

    return {"diff": "\n".join(output), "firstChangedLine": first_changed_line}
