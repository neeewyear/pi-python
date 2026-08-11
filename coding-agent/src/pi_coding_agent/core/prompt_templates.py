"""提示词模板加载与参数替换（对应 TS ``core/prompt-templates.ts``）。

目录输入非递归加载直接子级 ``.md`` 文件；文件输入加载显式 ``.md`` 文件。
支持 ``$1`` / ``$@`` / ``$ARGUMENTS`` / ``${@:N}`` / ``${@:N:L}`` / ``${N:-default}`` 占位符替换。
"""

from __future__ import annotations

import re
from pathlib import Path

import aiofiles
import yaml
from pydantic import BaseModel, ConfigDict

from ..config import CONFIG_DIR_NAME, get_agent_dir
from .diagnostics import ResourceDiagnostic


class PromptTemplate(BaseModel):
    """提示词模板（对应 TS ``PromptTemplate``）。"""

    name: str
    description: str
    argument_hint: str | None = None
    content: str
    source_info: object
    file_path: str


class LoadPromptTemplatesOptions(BaseModel):
    """``load_prompt_templates`` 选项（对应 TS ``LoadPromptTemplatesOptions``）。"""

    cwd: str
    agent_dir: str
    prompt_paths: list[str]
    include_defaults: bool


# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------


def parse_command_args(args_string: str) -> list[str]:
    """解析参数串（支持单双引号）（对应 TS ``parseCommandArgs``）。"""
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
    """替换模板占位符（对应 TS ``substituteArgs``）。

    支持：
    - ``$1``, ``$2``, ... 位置参数
    - ``$@`` 和 ``$ARGUMENTS`` 全部参数
    - ``${N:-default}`` 带默认值的位置参数
    - ``${@:-default}`` 和 ``${ARGUMENTS:-default}`` 带默认值的全部参数
    - ``${@:N}`` 从第 N 个参数开始的切片
    - ``${@:N:L}`` 从第 N 个参数开始取 L 个
    """
    all_args = " ".join(args)

    def replacer(m: re.Match[str]) -> str:
        default_target = m.group(1)
        default_value = m.group(2)
        slice_start = m.group(3)
        slice_length = m.group(4)
        simple = m.group(5)

        if default_target is not None:
            if default_target in ("@", "ARGUMENTS"):
                value = all_args
            else:
                idx = int(default_target) - 1
                value = args[idx] if idx < len(args) else ""
            return value if value else (default_value or "")

        if slice_start is not None:
            start = int(slice_start) - 1
            if start < 0:
                start = 0
            if slice_length is not None:
                length = int(slice_length)
                return " ".join(args[start : start + length])
            return " ".join(args[start:])

        if simple in ("ARGUMENTS", "@"):
            return all_args

        index = int(simple) - 1
        return args[index] if index < len(args) else ""

    return re.sub(
        r"\$\{(\d+|ARGUMENTS|@):-([^}]*)\}|\$\{@:(\d+)(?::(\d+))?\}|\$(ARGUMENTS|@|\d+)",
        replacer,
        content,
    )


# ---------------------------------------------------------------------------
# 文件加载
# ---------------------------------------------------------------------------


async def _load_template_from_file(file_path: Path) -> PromptTemplate | None:
    """从单个文件加载提示词模板（对应 TS ``loadTemplateFromFile``）。"""
    try:
        async with aiofiles.open(str(file_path), mode="r", encoding="utf-8") as f:
            raw_content = await f.read()
    except Exception:  # noqa: BLE001
        return None

    frontmatter, body = _parse_frontmatter(raw_content)

    name = file_path.stem  # Remove .md extension

    # Get description from frontmatter or first non-empty line
    description = frontmatter.get("description", "")
    description = description if isinstance(description, str) else ""
    if not description:
        first_line = next((line for line in body.split("\n") if line.strip()), None)
        if first_line is not None:
            description = first_line[:60]
            if len(first_line) > 60:
                description += "..."

    argument_hint = frontmatter.get("argument-hint")
    argument_hint = argument_hint if isinstance(argument_hint, str) else None

    return PromptTemplate(
        name=name,
        description=description,
        argument_hint=argument_hint,
        content=body,
        source_info=frontmatter,
        file_path=str(file_path),
    )


async def _load_templates_from_dir(directory: Path) -> list[PromptTemplate]:
    """从目录非递归加载 .md 文件（对应 TS ``loadTemplatesFromDir``）。"""
    templates: list[PromptTemplate] = []

    if not directory.exists():
        return templates

    try:
        entries = sorted(directory.iterdir(), key=lambda e: e.name)
    except Exception:  # noqa: BLE001
        return templates

    for entry in entries:
        # For symlinks, check if they point to a file
        is_file = entry.is_file()
        if entry.is_symlink():
            try:
                is_file = entry.resolve().is_file()
            except Exception:  # noqa: BLE001
                continue

        if is_file and entry.name.endswith(".md"):
            template = await _load_template_from_file(entry)
            if template is not None:
                templates.append(template)

    return templates


def _parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
    """解析 YAML frontmatter。"""
    normalized = content.replace("\r\n", "\n").replace("\r", "\n")

    if not normalized.startswith("---"):
        return {}, normalized

    end_index = normalized.find("\n---", 3)
    if end_index == -1:
        return {}, normalized

    yaml_string = normalized[4:end_index]
    body = normalized[end_index + 4 :].strip()

    try:
        data = yaml.safe_load(yaml_string) or {}
        if isinstance(data, dict):
            return data, body
        return {}, body
    except Exception:  # noqa: BLE001
        return {}, body


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


async def load_prompt_templates(options: LoadPromptTemplatesOptions) -> list[PromptTemplate]:
    """加载所有提示词模板（对应 TS ``loadPromptTemplates``）。

    加载顺序：
    1. 全局：``agentDir/prompts/``
    2. 项目：``cwd/.pi/prompts/``
    3. 显式提示词路径
    """
    resolved_cwd = Path(options.cwd).resolve()
    resolved_agent_dir = Path(options.agent_dir).resolve()

    templates: list[PromptTemplate] = []

    global_prompts_dir = resolved_agent_dir / "prompts"
    project_prompts_dir = resolved_cwd / CONFIG_DIR_NAME / "prompts"

    def is_under_path(target: str, root: str) -> bool:
        normalized_root = str(Path(root).resolve())
        if target == normalized_root:
            return True
        prefix = f"{normalized_root}/"
        return target.startswith(prefix)

    if options.include_defaults:
        templates.extend(await _load_templates_from_dir(global_prompts_dir))
        templates.extend(await _load_templates_from_dir(project_prompts_dir))

    # Load explicit prompt paths
    for raw_path in options.prompt_paths:
        resolved_path = str(Path(raw_path).resolve())
        p = Path(resolved_path)
        if not p.exists():
            continue

        try:
            if p.is_dir():
                templates.extend(await _load_templates_from_dir(p))
            elif p.is_file() and resolved_path.endswith(".md"):
                template = await _load_template_from_file(p)
                if template is not None:
                    templates.append(template)
        except Exception:  # noqa: BLE001
            pass

    return templates


def expand_prompt_template(text: str, templates: list[PromptTemplate]) -> str:
    """展开提示词模板（对应 TS ``expandPromptTemplate``）。

    如果 ``text`` 以 ``/`` 开头，尝试匹配模板名称并替换参数。
    否则返回原文本。
    """
    if not text.startswith("/"):
        return text

    match = re.match(r"^/([^\s]+)(?:\s+([\s\S]*))?$", text)
    if not match:
        return text

    template_name = match.group(1)
    args_string = match.group(2) or ""

    template = next((t for t in templates if t.name == template_name), None)
    if template is not None:
        args = parse_command_args(args_string)
        return substitute_args(template.content, args)

    return text


__all__ = [
    "LoadPromptTemplatesOptions",
    "PromptTemplate",
    "expand_prompt_template",
    "load_prompt_templates",
    "parse_command_args",
    "substitute_args",
]