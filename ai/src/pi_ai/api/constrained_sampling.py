"""受限采样（对应 ``constrained-sampling.ts``）。"""

from __future__ import annotations

import json
from typing import Any


class GrammarConstrainedSampling:
    """语法约束采样配置。"""

    def __init__(self, fmt: str, definition: str, input_property: str) -> None:
        self.format = fmt  # "lark" | "regex"
        self.definition = definition
        self.input_property = input_property


class GrammarToolInputJsonBuffer:
    """语法工具输入 JSON 缓冲区。"""

    def __init__(self) -> None:
        self.input: str = ""
        self.started: bool = False
        self.closed: bool = False


def get_grammar_tool_input(
    tool_name: str,
    arguments: dict[str, Any],
    input_property: str,
) -> str:
    """获取语法工具输入。"""
    inp = arguments.get(input_property)
    if not isinstance(inp, str):
        raise ValueError(
            f'Grammar tool call "{tool_name}" requires argument "{input_property}" to be a string.'
        )
    return inp


def append_grammar_tool_input_json_delta(
    buffer: GrammarToolInputJsonBuffer,
    input_property: str,
    next_input: str,
    close: bool,
) -> str | None:
    """追加语法工具输入 JSON delta。"""
    if buffer.closed:
        if close and next_input == buffer.input:
            return None
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed after it was closed'
        )
    if not next_input.startswith(buffer.input):
        raise ValueError(
            f'grammar tool input for property "{input_property}" changed non-monotonically'
        )

    input_delta = next_input[len(buffer.input) :]
    if not close and len(input_delta) == 0:
        return None

    delta = ""
    if not buffer.started:
        delta += f'{{{json.dumps(input_property)}:"'
        buffer.started = True
    delta += json.dumps(input_delta)[1:-1]
    buffer.input = next_input

    if close:
        delta += '"}'
        buffer.closed = True
    return delta


def _infer_grammar_input_property(parameters: dict[str, Any]) -> str:
    """推断语法工具的输入属性名。"""
    schema = parameters
    if schema.get("type") != "object":
        raise ValueError(
            "grammar constrained sampling requires an object parameter schema"
        )
    required = schema.get("required")
    if (
        not isinstance(required, list)
        or len(required) != 1
        or not isinstance(required[0], str)
    ):
        raise ValueError(
            "grammar constrained sampling requires exactly one required string property"
        )
    input_property = required[0]
    properties = schema.get("properties")
    if not isinstance(properties, dict) or input_property not in properties:
        raise ValueError(
            f"grammar constrained sampling requires a properties entry for {input_property}"
        )
    prop = properties[input_property]
    if not isinstance(prop, dict) or prop.get("type") != "string":
        raise ValueError(
            f"grammar constrained sampling property {input_property} must have type string"
        )
    return input_property


def resolve_json_schema_strict_sampling(
    tool: Any,
    supports_strict_mode: bool,
) -> bool | None:
    """解析 JSON Schema 严格采样。"""
    config = getattr(tool, "constrained_sampling", None) or (
        tool.get("constrained_sampling") if isinstance(tool, dict) else None
    )
    if not config or config.get("type") != "json_schema":
        return None
    if supports_strict_mode:
        return True
    if config.get("strict") == "require":
        raise ValueError(
            f'Tool "{tool.name if hasattr(tool, "name") else tool.get("name")}" '
            "requires JSON-schema constrained sampling, but strict tools are unsupported."
        )
    return None


def resolve_grammar_constrained_sampling(
    tool: Any,
    supports_openai_grammar_tools: bool,
) -> GrammarConstrainedSampling | None:
    """解析语法约束采样。"""
    config = getattr(tool, "constrained_sampling", None) or (
        tool.get("constrained_sampling") if isinstance(tool, dict) else None
    )
    if not config or config.get("type") != "grammar":
        return None
    if not supports_openai_grammar_tools:
        return None

    variants = config.get("variants", {})
    lark_definition = variants.get("openai_lark")
    regex_definition = variants.get("openai_regex")
    has_lark = isinstance(lark_definition, str) and len(lark_definition.strip()) > 0
    has_regex = isinstance(regex_definition, str) and len(regex_definition.strip()) > 0
    if not has_lark and not has_regex:
        raise ValueError(
            f'Tool "{tool.name if hasattr(tool, "name") else tool.get("name")}" '
            "cannot use grammar constrained sampling: no supported grammar variant was provided."
        )

    try:
        raw_params = getattr(tool, "parameters", None) or (
            tool.get("parameters") if isinstance(tool, dict) else {}
        )
        parameters: dict[str, Any] = raw_params if isinstance(raw_params, dict) else {}
        return GrammarConstrainedSampling(
            fmt="lark" if has_lark else "regex",
            definition=lark_definition if has_lark else regex_definition,
            input_property=_infer_grammar_input_property(parameters),
        )
    except ValueError as e:
        raise ValueError(
            f'Tool "{tool.name if hasattr(tool, "name") else tool.get("name")}" '
            f"cannot use grammar constrained sampling: {e}."
        ) from e


def create_grammar_tool_input_properties(
    tools: list[Any] | None,
    supports_openai_grammar_tools: bool,
) -> dict[str, str]:
    """创建语法工具输入属性映射。"""
    properties: dict[str, str] = {}
    for tool in tools or []:
        grammar = resolve_grammar_constrained_sampling(
            tool, supports_openai_grammar_tools
        )
        if grammar:
            name = tool.name if hasattr(tool, "name") else tool.get("name", "")
            properties[name] = grammar.input_property
    return properties
