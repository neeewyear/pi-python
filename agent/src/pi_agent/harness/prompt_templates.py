"""提示词模板加载与参数替换（对应 ``harness/prompt-templates.ts``）。

目录输入非递归加载直接子级 ``.md`` 文件；文件输入加载显式 ``.md`` 文件。
支持 ``$1`` / ``$@`` / ``$ARGUMENTS`` / ``${@:N}`` / ``${@:N:L}`` 占位符替换。
"""

from __future__ import annotations

import re
from typing import Callable, Literal, TypeAlias, TypeVar

import yaml
from pydantic import BaseModel

from ..result import Result, err, ok, to_error
from .types import ExecutionEnv, FileInfo, PromptTemplate

PromptTemplateDiagnosticCode: TypeAlias = Literal[
    "file_info_failed",
    "list_failed",
    "read_failed",
    "parse_failed",
]


class PromptTemplateDiagnostic(BaseModel):
    """加载提示词模板时产生的警告。"""

    type: Literal["warning"] = "warning"
    code: PromptTemplateDiagnosticCode
    message: str
    path: str


async def load_prompt_templates(
    env: ExecutionEnv, paths: str | list[str]
) -> tuple[list[PromptTemplate], list[PromptTemplateDiagnostic]]:
    """从一个或多个路径加载提示词模板。

    缺失路径与非 markdown 文件被跳过；读取/解析失败以诊断返回。
    """
    templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path in [paths] if isinstance(paths, str) else paths:
        info = await env.file_info(path)
        if not info.is_ok():
            if info.error.code != "not_found":
                diagnostics.append(_diagnostic("file_info_failed", info.error.message, path))
            continue
        kind = await _resolve_kind(env, info.value, diagnostics)
        if kind == "directory":
            result = await _load_templates_from_dir(env, info.value.path)
            templates.extend(result[0])
            diagnostics.extend(result[1])
        elif kind == "file" and info.value.name.endswith(".md"):
            file_result = await _load_template_from_file(env, info.value.path, info.value.name)
            if file_result[0] is not None:
                templates.append(file_result[0])
            diagnostics.extend(file_result[1])
    return templates, diagnostics


TSource = TypeVar("TSource")


async def load_sourced_prompt_templates(
    env: ExecutionEnv,
    inputs: list[tuple[str, TSource]],
    map_template: Callable[[PromptTemplate, TSource], PromptTemplate] | None = None,
) -> tuple[list[tuple[PromptTemplate, TSource]], list[PromptTemplateDiagnostic]]:
    """从带来源标记的路径加载模板；来源值原样保留并附着到每个模板与诊断上。"""
    templates: list[tuple[PromptTemplate, TSource]] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    for path, source in inputs:
        loaded, diags = await load_prompt_templates(env, path)
        for template in loaded:
            templates.append((map_template(template, source) if map_template else template, source))
        diagnostics.extend(diags)
    return templates, diagnostics


async def _load_templates_from_dir(
    env: ExecutionEnv, directory: str
) -> tuple[list[PromptTemplate], list[PromptTemplateDiagnostic]]:
    templates: list[PromptTemplate] = []
    diagnostics: list[PromptTemplateDiagnostic] = []
    entries = await env.list_dir(directory)
    if not entries.is_ok():
        diagnostics.append(_diagnostic("list_failed", entries.error.message, directory))
        return templates, diagnostics
    for entry in sorted(entries.value, key=lambda e: e.name):
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file" or not entry.name.endswith(".md"):
            continue
        result = await _load_template_from_file(env, entry.path, entry.name)
        if result[0] is not None:
            templates.append(result[0])
        diagnostics.extend(result[1])
    return templates, diagnostics


async def _load_template_from_file(
    env: ExecutionEnv, file_path: str, file_name: str
) -> tuple[PromptTemplate | None, list[PromptTemplateDiagnostic]]:
    diagnostics: list[PromptTemplateDiagnostic] = []
    raw = await env.read_text_file(file_path)
    if not raw.is_ok():
        diagnostics.append(_diagnostic("read_failed", raw.error.message, file_path))
        return None, diagnostics

    parsed = _parse_frontmatter(raw.value)
    if not parsed.is_ok():
        diagnostics.append(_diagnostic("parse_failed", str(parsed.error), file_path))
        return None, diagnostics
    frontmatter, body = parsed.value

    first_line = next((line for line in body.splitlines() if line.strip()), None)
    description = frontmatter.get("description")
    description = description if isinstance(description, str) else ""
    if not description and first_line:
        description = first_line[:60]
        if len(first_line) > 60:
            description += "..."

    template = PromptTemplate(name=file_name[:-3] if file_name.lower().endswith(".md") else file_name, description=description, content=body)
    return template, diagnostics


def _diagnostic(code: PromptTemplateDiagnosticCode, message: str, path: str) -> PromptTemplateDiagnostic:
    return PromptTemplateDiagnostic(code=code, message=message, path=path)


async def _resolve_kind(
    env: ExecutionEnv, info: FileInfo, diagnostics: list[PromptTemplateDiagnostic]
) -> str | None:
    """把 symlink 解析为 file/directory；无法解析返回 None。"""
    if info.kind == "file" or info.kind == "directory":
        return info.kind
    canonical = await env.canonical_path(info.path)
    if not canonical.is_ok():
        if canonical.error.code != "not_found":
            diagnostics.append(_diagnostic("file_info_failed", canonical.error.message, info.path))
        return None
    target = await env.file_info(canonical.value)
    if not target.is_ok():
        if target.error.code != "not_found":
            diagnostics.append(_diagnostic("file_info_failed", target.error.message, info.path))
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


def _parse_frontmatter(content: str) -> Result[tuple[dict[str, object], str], Exception]:
    try:
        normalized = content.replace("\r\n", "\n").replace("\r", "\n")
        if not normalized.startswith("---"):
            return ok(({}, normalized))
        end_index = normalized.find("\n---", 3)
        if end_index == -1:
            return ok(({}, normalized))
        yaml_string = normalized[4:end_index]
        body = normalized[end_index + 4 :].strip()
        data = yaml.safe_load(yaml_string) or {}
        return ok((data if isinstance(data, dict) else {}, body))
    except Exception as exc:  # noqa: BLE001
        return err(to_error(exc))


def parse_command_args(args_string: str) -> list[str]:
    """解析参数串（支持简单 shell 风格的单双引号）。"""
    args: list[str] = []
    current = ""
    in_quote: str | None = None
    for char in args_string:
        if in_quote is not None:
            if char == in_quote:
                in_quote = None
            else:
                current += char
        elif char in ('"', "'"):
            in_quote = char
        elif char in (" ", "\t"):
            if current:
                args.append(current)
                current = ""
        else:
            current += char
    if current:
        args.append(current)
    return args


def substitute_args(content: str, args: list[str]) -> str:
    """替换模板占位符（``$1`` / ``$@`` / ``$ARGUMENTS`` / ``${@:N}`` / ``${@:N:L}``）。"""
    result = re.sub(r"\$(\d+)", lambda m: args[int(m.group(1)) - 1] if int(m.group(1)) - 1 < len(args) else "", content)
    result = re.sub(
        r"\$\{@:(\d+)(?::(\d+))?\}",
        lambda m: _slice_args(args, int(m.group(1)), int(m.group(2)) if m.group(2) else None),
        result,
    )
    all_args = " ".join(args)
    result = result.replace("$ARGUMENTS", all_args).replace("$@", all_args)
    return result


def _slice_args(args: list[str], start_1: int, length: int | None) -> str:
    start = max(start_1 - 1, 0)
    return " ".join(args[start : start + length] if length is not None else args[start:])


def format_prompt_template_invocation(template: PromptTemplate, args: list[str] | None = None) -> str:
    """用位置参数格式化模板调用。"""
    return substitute_args(template.content, args or [])
