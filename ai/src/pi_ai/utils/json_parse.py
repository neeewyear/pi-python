"""JSON 解析与修复工具。

提供 ``repair_json``、``parse_json_with_repair``、``parse_streaming_json``。
"""

from __future__ import annotations

import json
import re
from typing import cast

VALID_JSON_ESCAPES: set[str] = {'"', "\\", "/", "b", "f", "n", "r", "t", "u"}


def _is_control_character(char: str) -> bool:
    """检查是否为控制字符（0x00-0x1F）。"""
    code_point = ord(char)
    return 0x00 <= code_point <= 0x1F


def _escape_control_character(char: str) -> str:
    """转义控制字符。"""
    escape_map = {
        "\b": "\\b",
        "\f": "\\f",
        "\n": "\\n",
        "\r": "\\r",
        "\t": "\\t",
    }
    if char in escape_map:
        return escape_map[char]
    return f"\\u{ord(char):04x}"


def repair_json(json_str: str) -> str:
    """修复格式错误的 JSON 字符串。

    处理：
    - 字符串中的原始控制字符
    - 无效转义字符前的反斜杠
    """
    repaired: list[str] = []
    in_string = False
    i = 0

    while i < len(json_str):
        char = json_str[i]

        if not in_string:
            repaired.append(char)
            if char == '"':
                in_string = True
            i += 1
            continue

        if char == '"':
            repaired.append(char)
            in_string = False
            i += 1
            continue

        if char == "\\":
            if i + 1 >= len(json_str):
                repaired.append("\\\\")
                i += 1
                continue

            next_char = json_str[i + 1]

            if next_char == "u":
                unicode_digits = json_str[i + 2 : i + 6]
                if re.match(r"^[0-9a-fA-F]{4}$", unicode_digits):
                    repaired.append(f"\\u{unicode_digits}")
                    i += 6
                    continue

            if next_char in VALID_JSON_ESCAPES:
                repaired.append(f"\\{next_char}")
                i += 2
                continue

            repaired.append("\\\\")
            i += 1
            continue

        if _is_control_character(char):
            repaired.append(_escape_control_character(char))
        else:
            repaired.append(char)
        i += 1

    return "".join(repaired)


def parse_json_with_repair(json_str: str) -> object:
    """解析 JSON，失败时尝试修复后重试。"""
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        repaired_json = repair_json(json_str)
        if repaired_json != json_str:
            return json.loads(repaired_json)
        raise


def _parse_partial_json(text: str) -> object | None:
    """尝试解析可能不完整的 JSON。

    从末尾开始尝试截断，直到 JSON 解析成功。
    """
    text = text.strip()
    if not text:
        return None

    # 尝试完整解析
    try:
        return cast(object, json.loads(text))
    except json.JSONDecodeError:
        pass

    # 尝试截断末尾的非 JSON 字符
    for end_idx in range(len(text), 0, -1):
        try:
            result = json.loads(text[:end_idx])
            if isinstance(result, dict):
                return result
            return None
        except json.JSONDecodeError:
            continue

    return None


def parse_streaming_json(partial_json: str | None) -> dict[str, object]:
    """尝试解析流式 JSON。

    总是返回有效的对象，即使 JSON 不完整。
    """
    if not partial_json or partial_json.strip() == "":
        return {}

    try:
        result = parse_json_with_repair(partial_json)
        if isinstance(result, dict):
            return result
        return {}
    except (json.JSONDecodeError, ValueError):
        pass

    result = _parse_partial_json(partial_json)
    if result is not None and isinstance(result, dict):
        return result

    try:
        result = _parse_partial_json(repair_json(partial_json))
        if result is not None and isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    return {}
