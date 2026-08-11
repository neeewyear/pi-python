"""TypeBox 辅助工具（对应 ``utils/typebox-helpers.ts``）。

Python 侧使用 Pydantic 替代 TypeBox，``StringEnum`` 可通过 Pydantic 的
``Literal`` 类型或 ``Field`` 约束实现。
"""

from __future__ import annotations

from typing import Any


def string_enum(
    values: list[str],
    description: str | None = None,
    default: str | None = None,
) -> dict[str, object]:
    """创建字符串枚举 schema（对应 TS ``StringEnum``）。

    返回字典形式的 JSON Schema 片段，兼容 Pydantic / JSON Schema 验证器。
    """
    schema: dict[str, object] = {
        "type": "string",
        "enum": list(values),
    }
    if description is not None:
        schema["description"] = description
    if default is not None:
        schema["default"] = default
    return schema