"""技能加载（对应 ``harness/skills.ts``）。

递归遍历目录加载 ``SKILL.md``，遵守 ignore 文件（.gitignore/.ignore/.fdignore），
非法的技能文件以诊断信息返回而不是失败。
"""

from __future__ import annotations

import re
from typing import Callable, Literal, TypeAlias, TypeVar

import pathspec
import yaml
from pydantic import BaseModel

from ..result import Result, err, ok, to_error
from .types import ExecutionEnv, FileInfo, Skill

MAX_NAME_LENGTH = 64
MAX_DESCRIPTION_LENGTH = 1024
IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")

SkillDiagnosticCode: TypeAlias = Literal[
    "file_info_failed",
    "list_failed",
    "read_failed",
    "parse_failed",
    "invalid_metadata",
]
"""技能加载诊断码。"""


class SkillDiagnostic(BaseModel):
    """加载技能时产生的警告。"""

    type: Literal["warning"] = "warning"
    code: SkillDiagnosticCode
    message: str
    path: str


def _make_diagnostic(code: str, message: str, path: str) -> SkillDiagnostic:
    return SkillDiagnostic(code=code, message=message, path=path)


def format_skill_invocation(skill: Skill, additional_instructions: str | None = None) -> str:
    """格式化技能调用提示词，可追加额外用户指令。"""
    skill_block = (
        f'<skill name="{skill.name}" location="{skill.file_path}">\n'
        f"References are relative to {_dirname(skill.file_path)}.\n\n"
        f"{skill.content}\n</skill>"
    )
    return f"{skill_block}\n\n{additional_instructions}" if additional_instructions else skill_block


async def load_skills(
    env: ExecutionEnv, dirs: str | list[str]
) -> tuple[list[Skill], list[SkillDiagnostic]]:
    """从一个或多个目录加载技能。

    返回 ``(skills, diagnostics)``；缺失的输入目录被跳过。
    """
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []
    for directory in [dirs] if isinstance(dirs, str) else dirs:
        root_info = await env.file_info(directory)
        if not root_info.is_ok():
            if root_info.error.code != "not_found":
                diagnostics.append(_make_diagnostic("file_info_failed", root_info.error.message, directory))
            continue
        if await _resolve_kind(env, root_info.value, diagnostics) != "directory":
            continue
        patterns: list[str] = []
        result = await _load_skills_from_dir(env, root_info.value.path, True, patterns, root_info.value.path)
        skills.extend(result[0])
        diagnostics.extend(result[1])
    return skills, diagnostics


TSource = TypeVar("TSource")


async def load_sourced_skills(
    env: ExecutionEnv,
    inputs: list[tuple[str, TSource]],
    map_skill: Callable[[Skill, TSource], Skill] | None = None,
) -> tuple[list[tuple[Skill, TSource]], list[SkillDiagnostic]]:
    """从带来源标记的目录加载技能；来源值原样保留并附着到每个技能与诊断上。"""
    skills: list[tuple[Skill, TSource]] = []
    diagnostics: list[SkillDiagnostic] = []
    for path, source in inputs:
        loaded, diags = await load_skills(env, path)
        for skill in loaded:
            skills.append((map_skill(skill, source) if map_skill else skill, source))
        diagnostics.extend(diags)
    return skills, diagnostics


async def _load_skills_from_dir(
    env: ExecutionEnv,
    directory: str,
    include_root_files: bool,
    ignore_patterns: list[str],
    root_dir: str,
) -> tuple[list[Skill], list[SkillDiagnostic]]:
    skills: list[Skill] = []
    diagnostics: list[SkillDiagnostic] = []

    dir_info = await env.file_info(directory)
    if not dir_info.is_ok():
        if dir_info.error.code != "not_found":
            diagnostics.append(_make_diagnostic("file_info_failed", dir_info.error.message, directory))
        return skills, diagnostics
    if await _resolve_kind(env, dir_info.value, diagnostics) != "directory":
        return skills, diagnostics

    await _add_ignore_rules(env, ignore_patterns, directory, root_dir, diagnostics)
    ignore_matcher = pathspec.PathSpec.from_lines("gitwildmatch", ignore_patterns)

    entries = await env.list_dir(directory)
    if not entries.is_ok():
        diagnostics.append(_make_diagnostic("list_failed", entries.error.message, directory))
        return skills, diagnostics
    file_entries = entries.value

    for entry in file_entries:
        if entry.name != "SKILL.md":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind != "file":
            continue
        rel = _relative(root_dir, entry.path)
        if ignore_matcher.match_file(rel):
            continue
        skill, diags = await _load_skill_from_file(env, entry.path, dir_info.value.name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(diags)
        return skills, diagnostics

    for entry in sorted(file_entries, key=lambda e: e.name):
        if entry.name.startswith(".") or entry.name == "node_modules":
            continue
        kind = await _resolve_kind(env, entry, diagnostics)
        if kind is None:
            continue
        rel = _relative(root_dir, entry.path)
        ignore_path = f"{rel}/" if kind == "directory" else rel
        if ignore_matcher.match_file(ignore_path):
            continue
        if kind == "directory":
            sub_skills, sub_diags = await _load_skills_from_dir(
                env, entry.path, False, ignore_patterns, root_dir
            )
            skills.extend(sub_skills)
            diagnostics.extend(sub_diags)
            continue
        if kind != "file" or not include_root_files or not entry.name.endswith(".md"):
            continue
        skill, diags = await _load_skill_from_file(env, entry.path, dir_info.value.name)
        if skill is not None:
            skills.append(skill)
        diagnostics.extend(diags)

    return skills, diagnostics


async def _add_ignore_rules(
    env: ExecutionEnv,
    ignore_patterns: list[str],
    directory: str,
    root_dir: str,
    diagnostics: list[SkillDiagnostic],
) -> None:
    relative_dir = _relative(root_dir, directory)
    prefix = f"{relative_dir}/" if relative_dir else ""
    for filename in IGNORE_FILE_NAMES:
        ignore_path = await env.join_path([directory, filename])
        if not ignore_path.is_ok():
            diagnostics.append(_make_diagnostic("file_info_failed", ignore_path.error.message, directory))
            continue
        info = await env.file_info(ignore_path.value)
        if not info.is_ok():
            if info.error.code != "not_found":
                diagnostics.append(_make_diagnostic("file_info_failed", info.error.message, ignore_path.value))
            continue
        if info.value.kind != "file":
            continue
        content = await env.read_text_file(ignore_path.value)
        if not content.is_ok():
            diagnostics.append(_make_diagnostic("read_failed", content.error.message, ignore_path.value))
            continue
        for line in content.value.splitlines():
            prefixed = _prefix_ignore_pattern(line, prefix)
            if prefixed:
                ignore_patterns.append(prefixed)


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    trimmed = line.strip()
    if not trimmed:
        return None
    if trimmed.startswith("#") and not trimmed.startswith("\\#"):
        return None
    pattern = line
    negated = False
    if pattern.startswith("!"):
        negated = True
        pattern = pattern[1:]
    elif pattern.startswith("\\!"):
        pattern = pattern[1:]
    if pattern.startswith("/"):
        pattern = pattern[1:]
    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


async def _load_skill_from_file(
    env: ExecutionEnv, file_path: str, parent_dir_name: str
) -> tuple[Skill | None, list[SkillDiagnostic]]:
    diagnostics: list[SkillDiagnostic] = []
    raw = await env.read_text_file(file_path)
    if not raw.is_ok():
        diagnostics.append(_make_diagnostic("read_failed", raw.error.message, file_path))
        return None, diagnostics

    parsed = _parse_frontmatter(raw.value)
    if not parsed.is_ok():
        diagnostics.append(_make_diagnostic("parse_failed", str(parsed.error), file_path))
        return None, diagnostics
    frontmatter, body = parsed.value

    description = frontmatter.get("description")
    description = description if isinstance(description, str) else None
    for error in _validate_description(description):
        diagnostics.append(_make_diagnostic("invalid_metadata", error, file_path))

    frontmatter_name = frontmatter.get("name")
    name = frontmatter_name if isinstance(frontmatter_name, str) else parent_dir_name
    for error in _validate_name(name, parent_dir_name):
        diagnostics.append(_make_diagnostic("invalid_metadata", error, file_path))

    if not description or not description.strip():
        return None, diagnostics

    skill = Skill(
        name=name,
        description=description,
        content=body,
        file_path=file_path,
        disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
    )
    return skill, diagnostics


def _validate_name(name: str, parent_dir_name: str) -> list[str]:
    errors: list[str] = []
    if name != parent_dir_name:
        errors.append(f'name "{name}" does not match parent directory "{parent_dir_name}"')
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append("name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)")
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    errors: list[str] = []
    if not description or not description.strip():
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})")
    return errors


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


async def _resolve_kind(
    env: ExecutionEnv, info: FileInfo, diagnostics: list[SkillDiagnostic]
) -> str | None:
    """把 symlink 解析为 file/directory；无法解析返回 None。"""
    if info.kind == "file" or info.kind == "directory":
        return info.kind
    canonical = await env.canonical_path(info.path)
    if not canonical.is_ok():
        if canonical.error.code != "not_found":
            diagnostics.append(_make_diagnostic("file_info_failed", canonical.error.message, info.path))
        return None
    target = await env.file_info(canonical.value)
    if not target.is_ok():
        if target.error.code != "not_found":
            diagnostics.append(_make_diagnostic("file_info_failed", target.error.message, info.path))
        return None
    return target.value.kind if target.value.kind in ("file", "directory") else None


def _dirname(path: str) -> str:
    """返回路径所在目录（兼容 ``/`` 与 ``\\`` 分隔符）。"""
    normalized = path.rstrip("/\\")
    idx = max(normalized.rfind("/"), normalized.rfind("\\"))
    if idx == 2 and normalized[1] == ":":
        return normalized[:3]
    return normalized[:idx] if idx > 0 else "/"


def _relative(root: str, path: str) -> str:
    normalized_root = root.replace("\\", "/").rstrip("/")
    normalized_path = path.replace("\\", "/").rstrip("/")
    if normalized_path == normalized_root:
        return ""
    if normalized_path.startswith(f"{normalized_root}/"):
        return normalized_path[len(normalized_root) + 1 :]
    return normalized_path.lstrip("/")
