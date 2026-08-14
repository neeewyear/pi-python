"""配置值解析。

支持以下格式的配置值：
- 字面量
- 环境变量插值（``$VAR`` 或 ``${VAR}``）
- shell 命令（``!command``）

仅在进程初次访问时缓存 shell 命令结果。
"""

from __future__ import annotations

import os
import re
import subprocess
from typing import Literal, Union

# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------

_command_result_cache: dict[str, str | None] = {}

# ---------------------------------------------------------------------------
# 解析器
# ---------------------------------------------------------------------------

_ENV_VAR_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ENV_VAR_NAME_PREFIX_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*")


class _LiteralPart:
    type: str = "literal"
    value: str

    def __init__(self, value: str) -> None:
        self.value = value


class _EnvPart:
    type: str = "env"
    name: str

    def __init__(self, name: str) -> None:
        self.name = name


_TemplatePart = Union[_LiteralPart, _EnvPart]
_ConfigValueReference = Union[
    tuple[Literal["command"], None],  # ("command", None) — shell command
    tuple[Literal["template"], list[_TemplatePart]],  # ("template", parts)
]


def _append_literal(parts: list[_TemplatePart], value: str) -> None:
    if not value:
        return
    if parts and isinstance(parts[-1], _LiteralPart):
        parts[-1].value += value
        return
    parts.append(_LiteralPart(value))


def _parse_config_value_template(config: str) -> list[_TemplatePart]:
    parts: list[_TemplatePart] = []
    index = 0

    while index < len(config):
        dollar_index = config.find("$", index)
        if dollar_index < 0:
            _append_literal(parts, config[index:])
            break

        _append_literal(parts, config[index:dollar_index])
        next_char = config[dollar_index + 1] if dollar_index + 1 < len(config) else ""

        if next_char in ("$", "!"):
            _append_literal(parts, next_char)
            index = dollar_index + 2
            continue

        if next_char == "{":
            end_index = config.find("}", dollar_index + 2)
            if end_index < 0:
                _append_literal(parts, "$")
                index = dollar_index + 1
                continue

            name = config[dollar_index + 2 : end_index]
            if _ENV_VAR_NAME_RE.match(name):
                parts.append(_EnvPart(name))
            else:
                _append_literal(parts, config[dollar_index : end_index + 1])
            index = end_index + 1
            continue

        match = _ENV_VAR_NAME_PREFIX_RE.match(config, dollar_index + 1)
        if match:
            parts.append(_EnvPart(match.group(0)))
            index = dollar_index + 1 + match.end() - match.start()
            continue

        _append_literal(parts, "$")
        index = dollar_index + 1

    return parts


def _parse_config_value_reference(config: str) -> _ConfigValueReference:
    if config.startswith("!"):
        return ("command", None)
    return ("template", _parse_config_value_template(config))


def _resolve_env_config_value(
    name: str, env: dict[str, str] | None = None
) -> str | None:
    if env and name in env:
        return env[name]
    return os.environ.get(name)


def _get_template_env_var_names(parts: list[_TemplatePart]) -> list[str]:
    names: list[str] = []
    for part in parts:
        if not isinstance(part, _EnvPart):
            continue
        if part.name not in names:
            names.append(part.name)
    return names


def _resolve_template(
    parts: list[_TemplatePart], env: dict[str, str] | None = None
) -> str | None:
    resolved = ""
    for part in parts:
        if isinstance(part, _LiteralPart):
            resolved += part.value
            continue
        env_value = _resolve_env_config_value(part.name, env)
        if env_value is None:
            return None
        resolved += env_value
    return resolved


def _execute_command(command_config: str) -> str | None:
    if command_config in _command_result_cache:
        return _command_result_cache[command_config]

    command = command_config[1:]  # strip leading "!"
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        value = result.stdout.strip() or None
    except (subprocess.TimeoutExpired, OSError):
        value = None

    _command_result_cache[command_config] = value
    return value


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def get_config_value_env_var_name(config: str) -> str | None:
    """获取配置值对应的环境变量名（仅当值为纯环境变量引用时返回）。"""
    ref_type, ref_value = _parse_config_value_reference(config)
    if ref_type == "command":
        return None
    assert ref_value is not None
    parts = ref_value
    if len(parts) == 1 and isinstance(parts[0], _EnvPart):
        return parts[0].name
    return None


def get_config_value_env_var_names(config: str) -> list[str]:
    """获取配置值中引用的所有环境变量名。"""
    ref_type, ref_value = _parse_config_value_reference(config)
    if ref_type == "command":
        return []
    assert ref_value is not None
    return _get_template_env_var_names(ref_value)


def get_missing_config_value_env_var_names(
    config: str, env: dict[str, str] | None = None
) -> list[str]:
    """获取配置值中缺失的环境变量名。"""
    return [
        name
        for name in get_config_value_env_var_names(config)
        if _resolve_env_config_value(name, env) is None
    ]


def is_command_config_value(config: str) -> bool:
    """判断配置值是否为 shell 命令。"""
    ref_type, _ = _parse_config_value_reference(config)
    return ref_type == "command"


def is_config_value_configured(config: str, env: dict[str, str] | None = None) -> bool:
    """判断配置值是否已配置（所有引用的环境变量都存在）。"""
    return len(get_missing_config_value_env_var_names(config, env)) == 0


def resolve_config_value(config: str, env: dict[str, str] | None = None) -> str | None:
    """解析配置值。

    支持：
    - ``!command``：执行 shell 命令并使用 stdout
    - ``$VAR`` / ``${VAR}``：环境变量插值
    - ``$$``：转义字面量 ``$``
    - 其他：视为字面量
    """
    ref_type, ref_value = _parse_config_value_reference(config)
    if ref_type == "command":
        return _execute_command(config)
    assert ref_value is not None
    return _resolve_template(ref_value, env)


def resolve_config_value_uncached(
    config: str, env: dict[str, str] | None = None
) -> str | None:
    """解析配置值（不使用缓存）。"""
    ref_type, ref_value = _parse_config_value_reference(config)
    if ref_type == "command":
        command = config[1:]
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.stdout.strip() or None
        except (subprocess.TimeoutExpired, OSError):
            return None
    assert ref_value is not None
    return _resolve_template(ref_value, env)


def resolve_config_value_or_throw(
    config: str, description: str, env: dict[str, str] | None = None
) -> str:
    """解析配置值，失败时抛出异常。"""
    resolved_value = resolve_config_value_uncached(config, env)
    if resolved_value is not None:
        return resolved_value

    ref_type, ref_value = _parse_config_value_reference(config)
    if ref_type == "command":
        raise ValueError(
            f"Failed to resolve {description} from shell command: {config[1:]}"
        )

    missing_env_vars = get_missing_config_value_env_var_names(config, env)
    if len(missing_env_vars) == 1:
        raise ValueError(
            f"Failed to resolve {description} from environment variable: {missing_env_vars[0]}"
        )
    if len(missing_env_vars) > 1:
        raise ValueError(
            f"Failed to resolve {description} from environment variables: "
            f"{', '.join(missing_env_vars)}"
        )

    raise ValueError(f"Failed to resolve {description}")


def resolve_headers(
    headers: dict[str, str] | None,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """解析头信息中的所有值。"""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved_value = resolve_config_value(value, env)
        if resolved_value:
            resolved[key] = resolved_value
    return resolved if resolved else None


def resolve_headers_or_throw(
    headers: dict[str, str] | None,
    description: str,
    env: dict[str, str] | None = None,
) -> dict[str, str] | None:
    """解析头信息中的所有值，失败时抛出异常。"""
    if not headers:
        return None
    resolved: dict[str, str] = {}
    for key, value in headers.items():
        resolved[key] = resolve_config_value_or_throw(
            value, f'{description} header "{key}"', env
        )
    return resolved if resolved else None


def clear_config_value_cache() -> None:
    """清除 shell 命令结果缓存。"""
    _command_result_cache.clear()
