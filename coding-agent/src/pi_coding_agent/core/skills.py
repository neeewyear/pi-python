"""技能加载（对应 TS ``core/skills.ts``）。

递归遍历目录加载 ``SKILL.md`` 或 ``.md`` 文件，遵守 ignore 文件（.gitignore/.ignore/.fdignore），
非法的技能文件以诊断信息返回而不是失败。
"""

from __future__ import annotations

import re
from pathlib import Path

import aiofiles
import pathspec
import yaml
from pydantic import BaseModel, ConfigDict

from ..config import CONFIG_DIR_NAME
from .diagnostics import ResourceCollision, ResourceDiagnostic

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MAX_NAME_LENGTH = 64
"""技能名称最大长度。"""

MAX_DESCRIPTION_LENGTH = 1024
"""技能描述最大长度。"""

IGNORE_FILE_NAMES = (".gitignore", ".ignore", ".fdignore")
"""需要读取的 ignore 文件名列表。"""


# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class SkillFrontmatter(BaseModel):
    """技能文件 frontmatter 元数据。"""

    name: str | None = None
    description: str | None = None
    disable_model_invocation: bool | None = None
    model_config = ConfigDict(extra="allow")


class SourceInfo(BaseModel):
    """来源信息（对应 TS ``SourceInfo``）。"""

    path: str
    source: str
    scope: str = "temporary"
    origin: str = "top-level"
    base_dir: str | None = None


class Skill(BaseModel):
    """已加载的技能（对应 TS ``Skill``）。"""

    name: str
    description: str
    file_path: str
    base_dir: str
    source_info: SourceInfo
    disable_model_invocation: bool = False


class SkillsResult(BaseModel):
    """技能加载结果。"""

    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []


class LoadSkillsFromDirOptions(BaseModel):
    """``load_skills_from_dir`` 选项。"""

    dir_path: Path
    source: str


class LoadSkillsOptions(BaseModel):
    """``load_skills`` 选项（对应 TS ``LoadSkillsOptions``）。"""

    cwd: str
    agent_dir: str
    skill_paths: list[str]
    include_defaults: bool


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _to_posix_path(p: str) -> str:
    """将路径分隔符转换为 POSIX 风格。"""
    return p.replace("\\", "/")


def _relative(root: str, target: str) -> str:
    """计算相对路径（POSIX 风格）。"""
    normalized_root = _to_posix_path(root).rstrip("/")
    normalized_target = _to_posix_path(target).rstrip("/")
    if normalized_target == normalized_root:
        return ""
    if normalized_target.startswith(f"{normalized_root}/"):
        return normalized_target[len(normalized_root) + 1 :]
    return normalized_target.lstrip("/")


def _prefix_ignore_pattern(line: str, prefix: str) -> str | None:
    """为忽略规则添加目录前缀（对应 TS ``prefixIgnorePattern``）。"""
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

    pattern = pattern.removeprefix("/")

    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _create_skill_source_info(file_path: str, base_dir: str, source: str) -> SourceInfo:
    """根据 source 类型创建 SourceInfo（对应 TS ``createSkillSourceInfo``）。"""
    if source == "user":
        return SourceInfo(
            path=file_path,
            source="local",
            scope="user",
            base_dir=base_dir,
        )
    if source == "project":
        return SourceInfo(
            path=file_path,
            source="local",
            scope="project",
            base_dir=base_dir,
        )
    if source == "path":
        return SourceInfo(
            path=file_path,
            source="local",
            base_dir=base_dir,
        )
    return SourceInfo(
        path=file_path,
        source=source,
        base_dir=base_dir,
    )


def _validate_name(name: str) -> list[str]:
    """验证技能名称（对应 TS ``validateName``）。"""
    errors: list[str] = []
    if len(name) > MAX_NAME_LENGTH:
        errors.append(f"name exceeds {MAX_NAME_LENGTH} characters ({len(name)})")
    if not re.fullmatch(r"[a-z0-9-]+", name):
        errors.append(
            "name contains invalid characters (must be lowercase a-z, 0-9, hyphens only)"
        )
    if name.startswith("-") or name.endswith("-"):
        errors.append("name must not start or end with a hyphen")
    if "--" in name:
        errors.append("name must not contain consecutive hyphens")
    return errors


def _validate_description(description: str | None) -> list[str]:
    """验证技能描述（对应 TS ``validateDescription``）。"""
    errors: list[str] = []
    if not description or description.strip() == "":
        errors.append("description is required")
    elif len(description) > MAX_DESCRIPTION_LENGTH:
        errors.append(
            f"description exceeds {MAX_DESCRIPTION_LENGTH} characters ({len(description)})"
        )
    return errors


def _escape_xml(value: str) -> str:
    """转义 XML 特殊字符。"""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


# ---------------------------------------------------------------------------
# 忽略规则处理
# ---------------------------------------------------------------------------


async def _add_ignore_rules(
    patterns: list[str],
    directory: Path,
    root_dir: Path,
) -> None:
    """从目录中的 ignore 文件读取规则并添加到 patterns 列表（对应 TS ``addIgnoreRules``）。"""
    relative_dir = _relative(str(root_dir), str(directory))
    prefix = f"{relative_dir}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = directory / filename
        if not ignore_path.exists():
            continue
        try:
            async with aiofiles.open(str(ignore_path), mode="r", encoding="utf-8") as f:
                content = await f.read()
            for line in content.splitlines():
                prefixed = _prefix_ignore_pattern(line, prefix)
                if prefixed:
                    patterns.append(prefixed)
        except Exception:  # noqa: BLE001
            pass


# ---------------------------------------------------------------------------
# 核心加载函数
# ---------------------------------------------------------------------------


async def _load_skill_from_file(
    file_path: Path,
    source: str,
) -> tuple[Skill | None, list[ResourceDiagnostic]]:
    """从单个文件加载技能（对应 TS ``loadSkillFromFile``）。"""
    diagnostics: list[ResourceDiagnostic] = []

    try:
        async with aiofiles.open(str(file_path), mode="r", encoding="utf-8") as f:
            raw_content = await f.read()
    except Exception as e:  # noqa: BLE001
        diagnostics.append(
            ResourceDiagnostic(
                type="warning",
                message=f"failed to read skill file: {e}",
                path=str(file_path),
            )
        )
        return None, diagnostics

    frontmatter, body = _parse_frontmatter(raw_content)
    skill_dir = file_path.parent
    parent_dir_name = skill_dir.name

    # Validate description
    description = frontmatter.get("description")
    description = description if isinstance(description, str) else None
    desc_errors = _validate_description(description)
    for error in desc_errors:
        diagnostics.append(
            ResourceDiagnostic(type="warning", message=error, path=str(file_path))
        )

    # Use name from frontmatter, or fall back to parent directory name
    fm_name = frontmatter.get("name")
    name = fm_name if isinstance(fm_name, str) else parent_dir_name

    # Validate name
    name_errors = _validate_name(name)
    for error in name_errors:
        diagnostics.append(
            ResourceDiagnostic(type="warning", message=error, path=str(file_path))
        )

    # Still load the skill even with warnings (unless description is completely missing)
    if not description or description.strip() == "":
        return None, diagnostics

    skill = Skill(
        name=name,
        description=description,
        file_path=str(file_path),
        base_dir=str(skill_dir),
        source_info=_create_skill_source_info(str(file_path), str(skill_dir), source),
        disable_model_invocation=frontmatter.get("disable-model-invocation") is True,
    )
    return skill, diagnostics


def _parse_frontmatter(content: str) -> tuple[dict[str, object], str]:
    """解析 YAML frontmatter（对应 TS ``parseFrontmatter``）。"""
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


async def _load_skills_from_dir(
    directory: Path,
    source: str,
    include_root_files: bool,
    ignore_patterns: list[str] | None = None,
    root_dir: Path | None = None,
) -> SkillsResult:
    """从目录递归加载技能（对应 TS ``loadSkillsFromDirInternal``）。"""
    skills: list[Skill] = []
    diagnostics: list[ResourceDiagnostic] = []

    if not directory.exists():
        return SkillsResult()

    root = root_dir or directory
    ig = ignore_patterns or []
    await _add_ignore_rules(ig, directory, root)
    spec = (
        pathspec.PathSpec.from_lines("gitwildmatch", ig)
        if ig
        else pathspec.PathSpec.from_lines("gitwildmatch", [])
    )

    try:
        entries = sorted(directory.iterdir(), key=lambda e: e.name)
    except Exception:  # noqa: BLE001
        return SkillsResult()

    # First pass: look for SKILL.md
    for entry in entries:
        if entry.name != "SKILL.md":
            continue

        is_file = entry.is_file()
        if entry.is_symlink():
            try:
                is_file = entry.resolve().is_file()
            except Exception:  # noqa: BLE001
                continue

        rel_path = _to_posix_path(str(entry.relative_to(root)))
        if not is_file or spec.match_file(rel_path):
            continue

        result = await _load_skill_from_file(entry, source)
        if result[0] is not None:
            skills.append(result[0])
        diagnostics.extend(result[1])
        return SkillsResult(skills=skills, diagnostics=diagnostics)

    # Second pass: recurse into subdirectories
    for entry in entries:
        if entry.name.startswith("."):
            continue
        if entry.name == "node_modules":
            continue

        is_directory = entry.is_dir()
        is_file = entry.is_file()
        if entry.is_symlink():
            try:
                resolved = entry.resolve()
                is_directory = resolved.is_dir()
                is_file = resolved.is_file()
            except Exception:  # noqa: BLE001
                continue

        rel_path = _to_posix_path(str(entry.relative_to(root)))
        ignore_path = f"{rel_path}/" if is_directory else rel_path
        if spec.match_file(ignore_path):
            continue

        if is_directory:
            sub_result = await _load_skills_from_dir(entry, source, False, ig, root)
            skills.extend(sub_result.skills)
            diagnostics.extend(sub_result.diagnostics)
            continue

        if not is_file or not include_root_files or not entry.name.endswith(".md"):
            continue

        result = await _load_skill_from_file(entry, source)
        if result[0] is not None:
            skills.append(result[0])
        diagnostics.extend(result[1])

    return SkillsResult(skills=skills, diagnostics=diagnostics)


async def load_skills_from_dir(options: LoadSkillsFromDirOptions) -> SkillsResult:
    """从目录加载技能（对应 TS ``loadSkillsFromDir``）。"""
    return await _load_skills_from_dir(options.dir_path, options.source, True)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------


async def load_skills(options: LoadSkillsOptions) -> SkillsResult:
    """从所有配置位置加载技能（对应 TS ``loadSkills``）。

    加载顺序：
    1. 默认用户技能目录（``agentDir/skills/``）
    2. 默认项目技能目录（``cwd/.pi/skills/``）
    3. 显式技能路径
    """
    resolved_cwd = Path(options.cwd).resolve()
    resolved_agent_dir = Path(options.agent_dir).resolve()

    skill_map: dict[str, Skill] = {}
    real_path_set: set[str] = set()
    all_diagnostics: list[ResourceDiagnostic] = []
    collision_diagnostics: list[ResourceDiagnostic] = []

    def add_skills(result: SkillsResult) -> None:
        all_diagnostics.extend(result.diagnostics)
        for skill in result.skills:
            real_path = _canonicalize_path(skill.file_path)
            if real_path in real_path_set:
                continue
            existing = skill_map.get(skill.name)
            if existing is not None:
                collision_diagnostics.append(
                    ResourceDiagnostic(
                        type="collision",
                        message=f'name "{skill.name}" collision',
                        path=skill.file_path,
                        collision=ResourceCollision(
                            resource_type="skill",
                            name=skill.name,
                            winner_path=existing.file_path,
                            loser_path=skill.file_path,
                        ),
                    )
                )
            else:
                skill_map[skill.name] = skill
                real_path_set.add(real_path)

    if options.include_defaults:
        user_skills_dir = resolved_agent_dir / "skills"
        add_skills(await _load_skills_from_dir(user_skills_dir, "user", True))

        project_skills_dir = resolved_cwd / CONFIG_DIR_NAME / "skills"
        add_skills(await _load_skills_from_dir(project_skills_dir, "project", True))

    user_skills_dir = resolved_agent_dir / "skills"
    project_skills_dir = resolved_cwd / CONFIG_DIR_NAME / "skills"

    def is_under_path(target: str, root: str) -> bool:
        normalized_root = str(Path(root).resolve())
        if target == normalized_root:
            return True
        prefix = f"{normalized_root}/"
        return target.startswith(prefix)

    def get_source(resolved_path: str) -> str:
        if not options.include_defaults:
            if is_under_path(resolved_path, str(user_skills_dir)):
                return "user"
            if is_under_path(resolved_path, str(project_skills_dir)):
                return "project"
        return "path"

    for raw_path in options.skill_paths:
        resolved_path = str(Path(raw_path).resolve())
        p = Path(resolved_path)
        if not p.exists():
            all_diagnostics.append(
                ResourceDiagnostic(
                    type="warning",
                    message="skill path does not exist",
                    path=resolved_path,
                )
            )
            continue

        try:
            source = get_source(resolved_path)
            if p.is_dir():
                add_skills(await _load_skills_from_dir(p, source, True))
            elif p.is_file() and resolved_path.endswith(".md"):
                result = await _load_skill_from_file(p, source)
                if result[0] is not None:
                    add_skills(SkillsResult(skills=[result[0]], diagnostics=result[1]))
                else:
                    all_diagnostics.extend(result[1])
            else:
                all_diagnostics.append(
                    ResourceDiagnostic(
                        type="warning",
                        message="skill path is not a markdown file",
                        path=resolved_path,
                    )
                )
        except Exception as e:  # noqa: BLE001
            message = f"failed to read skill path: {e}"
            all_diagnostics.append(
                ResourceDiagnostic(type="warning", message=message, path=resolved_path)
            )

    return SkillsResult(
        skills=list(skill_map.values()),
        diagnostics=[*all_diagnostics, *collision_diagnostics],
    )


def _canonicalize_path(path: str) -> str:
    """解析真实路径（跟随符号链接），失败时返回原路径（对应 TS ``canonicalizePath``）。"""
    try:
        return str(Path(path).resolve())
    except Exception:  # noqa: BLE001
        return path


# ---------------------------------------------------------------------------
# 格式化
# ---------------------------------------------------------------------------


def format_skills_for_prompt(skills: list[Skill]) -> str:
    """格式化技能列表用于系统提示词（对应 TS ``formatSkillsForPrompt``）。

    使用 XML 格式，遵循 Agent Skills 标准。
    被 ``disable_model_invocation`` 标记的技能不展示给模型。
    """
    visible_skills = [s for s in skills if not s.disable_model_invocation]

    if not visible_skills:
        return ""

    lines = [
        "",
        "",
        "The following skills provide specialized instructions for specific tasks.",
        "Use the read tool to load a skill's file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]

    for skill in visible_skills:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")

    lines.append("</available_skills>")

    return "\n".join(lines)


__all__ = [
    "LoadSkillsFromDirOptions",
    "LoadSkillsOptions",
    "Skill",
    "SkillFrontmatter",
    "SkillsResult",
    "SourceInfo",
    "format_skills_for_prompt",
    "load_skills",
    "load_skills_from_dir",
]
