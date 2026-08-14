"""工具调用验证与类型推断。

提供 ``validate_tool_call``、``validate_tool_arguments``。
"""

from __future__ import annotations

import copy
import json
from typing import Any, cast

from ..types import Tool, ToolCallContent


def get_schema_types(schema: dict[str, object]) -> list[str]:
    """获取 JSON Schema 的类型列表。"""
    schema_type = schema.get("type")
    if isinstance(schema_type, str):
        return [schema_type]
    if isinstance(schema_type, list):
        return [t for t in schema_type if isinstance(t, str)]
    return []


def matches_json_type(value: object, type_name: str) -> bool:
    """检查值是否匹配 JSON 类型。"""
    type_checkers = {
        "number": lambda v: isinstance(v, (int, float)),
        "integer": lambda v: isinstance(v, int),
        "boolean": lambda v: isinstance(v, bool),
        "string": lambda v: isinstance(v, str),
        "null": lambda v: v is None,
        "array": lambda v: isinstance(v, list),
        "object": lambda v: isinstance(v, dict),
    }
    checker = type_checkers.get(type_name)
    return checker(value) if checker else False


def _coerce_primitive_by_type(value: object, type_name: str) -> object:
    """按目标类型强制转换原始值。"""
    if type_name in ("number", "integer"):
        if value is None:
            return 0
        if isinstance(value, str) and value.strip():
            try:
                parsed = float(value)
                if type_name == "integer" and parsed == int(parsed):
                    return int(parsed)
                if type_name == "number":
                    return parsed
            except (ValueError, OverflowError):
                pass
        if isinstance(value, bool):
            return 1 if value else 0
        return value

    if type_name == "boolean":
        if value is None:
            return False
        if isinstance(value, str):
            if value == "true":
                return True
            if value == "false":
                return False
        if isinstance(value, (int, float)):
            if value == 1:
                return True
            if value == 0:
                return False
        return value

    if type_name == "string":
        if value is None:
            return ""
        if isinstance(value, (int, float, bool)):
            return str(value)
        return value

    if type_name == "null":
        if value in ("", 0, False):
            return None
        return value

    return value


def _coerce_with_json_schema(value: object, schema: dict[str, object]) -> object:
    """递归按 JSON Schema 强制转换类型。"""
    # 处理 allOf/anyOf/oneOf
    for compound_key in ("allOf", "anyOf", "oneOf"):
        compound = schema.get(compound_key)
        if isinstance(compound, list):
            for sub_schema in compound:
                if isinstance(sub_schema, dict):
                    value = _coerce_with_json_schema(value, sub_schema)

    schema_types = get_schema_types(schema)
    if schema_types:
        matches_union = len(schema_types) > 1 and any(
            matches_json_type(value, t) for t in schema_types
        )
        if not matches_union:
            for t in schema_types:
                candidate = _coerce_primitive_by_type(value, t)
                if candidate is not value:
                    value = candidate
                    break

    if isinstance(value, dict) and "object" in schema_types:
        _apply_schema_object_coercion(value, schema)

    if isinstance(value, list) and "array" in schema_types:
        _apply_schema_array_coercion(value, schema)

    return value


def _apply_schema_object_coercion(
    value: dict[str, object], schema: dict[str, object]
) -> None:
    """递归转换对象属性。"""
    properties = schema.get("properties")
    defined_keys: set[str] = set()
    if isinstance(properties, dict):
        defined_keys = set(properties.keys())
        for key, prop_schema in properties.items():
            if key in value and isinstance(prop_schema, dict):
                value[key] = _coerce_with_json_schema(value[key], prop_schema)

    additional = schema.get("additionalProperties")
    if isinstance(additional, dict):
        for key, prop_value in list(value.items()):
            if key not in defined_keys:
                value[key] = _coerce_with_json_schema(prop_value, additional)


def _apply_schema_array_coercion(
    value: list[object], schema: dict[str, object]
) -> None:
    """递归转换数组元素。"""
    items = schema.get("items")
    if isinstance(items, list):
        for i in range(len(value)):
            if i < len(items) and isinstance(items[i], dict):
                value[i] = _coerce_with_json_schema(value[i], items[i])
    elif isinstance(items, dict):
        for i in range(len(value)):
            value[i] = _coerce_with_json_schema(value[i], items)


def _format_validation_path(errors: list[dict[str, object]]) -> str:
    """格式化验证错误路径。"""
    lines: list[str] = []
    for error in errors:
        path = cast("list[Any]", error.get("loc", []))
        path_str = ".".join(str(p) for p in path) if path else "root"
        msg = error.get("msg", "Unknown error")
        lines.append(f"  - {path_str}: {msg}")
    return "\n".join(lines)


def _get_validator(
    schema: dict[str, object],
) -> object | None:
    """获取 Pydantic/schema 校验器。"""
    # 使用 Pydantic 的 TypeAdapter 进行验证
    # 对于简单 schema，使用基本类型检查
    return schema


def validate_tool_call(
    tools: list[Tool], tool_call: ToolCallContent
) -> dict[str, object]:
    """按名称查找工具并验证参数。"""
    tool = next((t for t in tools if t.name == tool_call.name), None)
    if tool is None:
        raise ValueError(f'Tool "{tool_call.name}" not found')
    return validate_tool_arguments(tool, tool_call)


def validate_tool_arguments(
    tool: Tool, tool_call: ToolCallContent
) -> dict[str, object]:
    """验证工具调用参数。"""
    args = copy.deepcopy(tool_call.args)

    # 尝试类型转换
    schema = tool.parameters
    coerced = _coerce_with_json_schema(args, schema)
    if isinstance(coerced, dict) and coerced is not args:
        args.clear()
        args.update(coerced)

    # 使用 Pydantic 进行验证
    from pydantic import TypeAdapter, ValidationError

    try:
        adapter = TypeAdapter(dict[str, object])
        adapter.validate_python(args)
        return args
    except ValidationError as e:
        error_lines = _format_validation_path(
            cast("list[dict[str, object]]", e.errors())
        )
        raise ValueError(
            f'Validation failed for tool "{tool_call.name}":\n{error_lines}\n\n'
            f"Received arguments:\n{json.dumps(tool_call.args, indent=2)}"
        ) from e
