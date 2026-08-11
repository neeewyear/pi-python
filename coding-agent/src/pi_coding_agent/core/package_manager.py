from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeAlias, cast

from ..config import CONFIG_DIR_NAME
from ..utils.git import GitSource, parse_git_url  # type: ignore[import-not-found]
from ..utils.paths import (
    canonicalize_path,
    is_local_path,
    mark_path_ignored_by_cloud_sync,
    resolve_path,
)
from .pi_manifest import read_pi_manifest

NETWORK_TIMEOUT_MS = 10000
UPDATE_CHECK_CONCURRENCY = 4
GIT_UPDATE_CONCURRENCY = 4


def is_offline_mode_enabled() -> bool:
    value = os.environ.get("PI_OFFLINE")
    if not value:
        return False
    return value in ("1", "true", "yes")


def is_exact_npm_version(version: str | None) -> bool:
    if not version:
        return False
    try:
        from packaging.version import Version

        Version(version)
        return True
    except Exception:
        return False


def get_npm_version_range(version: str | None) -> str | None:
    if not version:
        return None
    try:
        from packaging.specifiers import SpecifierSet

        SpecifierSet(version)
        return version
    except Exception:
        return None


@dataclass
class PathMetadata:
    source: str = ""
    scope: str = "temporary"
    origin: str = "top-level"
    base_dir: str | None = None


@dataclass
class ResolvedResource:
    path: str = ""
    enabled: bool = True
    metadata: PathMetadata | None = None


@dataclass
class ResolvedPaths:
    extensions: list[ResolvedResource] = field(default_factory=list)
    skills: list[ResolvedResource] = field(default_factory=list)
    prompts: list[ResolvedResource] = field(default_factory=list)
    themes: list[ResolvedResource] = field(default_factory=list)


@dataclass
class ProgressEvent:
    type: str = "start"  # "start" | "progress" | "complete" | "error"
    action: str = "install"  # "install" | "remove" | "update" | "clone" | "pull"
    source: str = ""
    message: str | None = None


@dataclass
class PackageUpdate:
    source: str = ""
    display_name: str = ""
    type: str = "npm"  # "npm" | "git"
    scope: str = "user"  # "user" | "project"


@dataclass
class ConfiguredPackage:
    source: str = ""
    scope: str = "user"  # "user" | "project"
    filtered: bool = False
    installed_path: str | None = None


class PackageManager:
    async def resolve(
        self, on_missing: Callable[[str], Awaitable[str]] | None = None
    ) -> ResolvedPaths:
        raise NotImplementedError

    async def install(self, source: str, options: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    async def install_and_persist(
        self, source: str, options: dict[str, Any] | None = None
    ) -> None:
        raise NotImplementedError

    async def remove(self, source: str, options: dict[str, Any] | None = None) -> None:
        raise NotImplementedError

    async def remove_and_persist(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        raise NotImplementedError

    async def update(self, source: str | None = None) -> None:
        raise NotImplementedError

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        raise NotImplementedError

    async def resolve_extension_sources(
        self, sources: list[str], options: dict[str, Any] | None = None
    ) -> ResolvedPaths:
        raise NotImplementedError

    def add_source_to_settings(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        raise NotImplementedError

    def remove_source_from_settings(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        raise NotImplementedError

    def set_progress_callback(
        self, callback: Callable[[ProgressEvent], None] | None
    ) -> None:
        raise NotImplementedError

    def get_installed_path(self, source: str, scope: str) -> str | None:
        raise NotImplementedError


SourceScope = str  # "user" | "project" | "temporary"


@dataclass
class NpmSource:
    type: str = "npm"
    spec: str = ""
    name: str = ""
    version: str | None = None
    range: str | None = None
    pinned: bool = False


@dataclass
class LocalSource:
    type: str = "local"
    path: str = ""


ParsedSource: TypeAlias = NpmSource | GitSource | LocalSource

InstalledSourceScope = str  # "user" | "project"


@dataclass
class ConfiguredUpdateSource:
    source: str = ""
    scope: InstalledSourceScope = "user"


@dataclass
class NpmUpdateTarget:
    source: str = ""
    scope: InstalledSourceScope = "user"
    parsed: NpmSource | None = None


@dataclass
class GitUpdateTarget:
    source: str = ""
    scope: InstalledSourceScope = "user"
    parsed: GitSource | None = None


ResourceType = str  # "extensions" | "skills" | "prompts" | "themes"
RESOURCE_TYPES: list[ResourceType] = ["extensions", "skills", "prompts", "themes"]

FILE_PATTERNS: dict[ResourceType, re.Pattern[str]] = {
    "extensions": re.compile(r"\.(ts|js)$"),
    "skills": re.compile(r"\.md$"),
    "prompts": re.compile(r"\.md$"),
    "themes": re.compile(r"\.json$"),
}

IGNORE_FILE_NAMES = [".gitignore", ".ignore", ".fdignore"]


def to_posix_path(p: str) -> str:
    return p.replace(os.sep, "/")


def get_home_dir() -> str:
    return os.environ.get("HOME") or os.path.expanduser("~")


def get_extension_temp_folder(agent_dir: str) -> str:
    temp_folder = Path(agent_dir) / "tmp" / "extensions"
    temp_folder.mkdir(parents=True, exist_ok=True)
    os.chmod(temp_folder, 0o700)
    return str(temp_folder)


def resource_precedence_rank(m: PathMetadata) -> int:
    if m.origin == "package":
        return 4
    scope_base = 0 if m.scope == "project" else 2
    return scope_base + (0 if m.source == "local" else 1)


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

    pattern = pattern.removeprefix("/")

    prefixed = f"{prefix}{pattern}" if prefix else pattern
    return f"!{prefixed}" if negated else prefixed


def _add_ignore_rules(ig: Any, dir_path: str, root_dir: str) -> None:
    relative_dir = os.path.relpath(dir_path, root_dir)
    prefix = f"{to_posix_path(relative_dir)}/" if relative_dir else ""

    for filename in IGNORE_FILE_NAMES:
        ignore_path = Path(dir_path) / filename
        if not ignore_path.exists():
            continue
        try:
            content = ignore_path.read_text("utf-8")
            patterns = [
                _prefix_ignore_pattern(line, prefix) for line in content.split("\n")
            ]
            patterns = [p for p in patterns if p]
            if patterns:
                ig.add(patterns)
        except OSError:
            pass


def _is_pattern(s: str) -> bool:
    return s.startswith(("!", "+", "-")) or "*" in s or "?" in s


def _is_override_pattern(s: str) -> bool:
    return s.startswith(("!", "+", "-"))


def _has_glob_pattern(s: str) -> bool:
    return "*" in s or "?" in s


def _split_patterns(entries: list[str]) -> tuple[list[str], list[str]]:
    plain: list[str] = []
    patterns: list[str] = []
    for entry in entries:
        if _is_pattern(entry):
            patterns.append(entry)
        else:
            plain.append(entry)
    return plain, patterns


def _collect_files(
    dir_path: str,
    file_pattern: re.Pattern[str],
    skip_node_modules: bool = True,
    ignore_matcher: Any = None,
    root_dir: str | None = None,
) -> list[str]:
    files: list[str] = []
    d = Path(dir_path)
    if not d.exists():
        return files

    root = root_dir or dir_path
    ig = ignore_matcher
    if ig is None:
        import pathspec

        ig = pathspec.PathSpec.from_lines("gitwildmatch", [])
    _add_ignore_rules(ig, dir_path, root)

    try:
        for entry in d.iterdir():
            if entry.name.startswith("."):
                continue
            if skip_node_modules and entry.name == "node_modules":
                continue

            full_path = str(entry.resolve())
            is_dir = entry.is_dir()
            is_file = entry.is_file()

            if entry.is_symlink():
                try:
                    stats = entry.stat()
                    is_dir = stat.S_ISDIR(stats.st_mode)
                    is_file = stat.S_ISREG(stats.st_mode)
                except OSError:
                    continue

            rel_path = to_posix_path(os.path.relpath(full_path, root))
            ignore_path = f"{rel_path}/" if is_dir else rel_path

            if is_dir:
                files.extend(
                    _collect_files(full_path, file_pattern, skip_node_modules, ig, root)
                )
            elif is_file and file_pattern.search(entry.name):
                files.append(full_path)
    except OSError:
        pass

    return files


SkillDiscoveryMode = str  # "pi" | "agents"


def _collect_skill_entries(
    dir_path: str,
    mode: SkillDiscoveryMode,
    ignore_matcher: Any = None,
    root_dir: str | None = None,
) -> list[str]:
    entries: list[str] = []
    d = Path(dir_path)
    if not d.exists():
        return entries

    root = root_dir or dir_path
    ig = ignore_matcher
    if ig is None:
        import pathspec

        ig = pathspec.PathSpec.from_lines("gitwildmatch", [])
    _add_ignore_rules(ig, dir_path, root)

    try:
        for entry in d.iterdir():
            if entry.name != "SKILL.md":
                continue

            full_path = str(entry.resolve())
            is_file = entry.is_file()
            if entry.is_symlink():
                try:
                    is_file = stat.S_ISREG(entry.stat().st_mode)
                except OSError:
                    continue

            rel_path = to_posix_path(os.path.relpath(full_path, root))
            if is_file:
                entries.append(full_path)
                return entries

        for entry in d.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue

            full_path = str(entry.resolve())
            is_dir = entry.is_dir()
            is_file = entry.is_file()

            if entry.is_symlink():
                try:
                    stats = entry.stat()
                    is_dir = stat.S_ISDIR(stats.st_mode)
                    is_file = stat.S_ISREG(stats.st_mode)
                except OSError:
                    continue

            rel_path = to_posix_path(os.path.relpath(full_path, root))
            if (
                mode == "pi"
                and dir_path == root
                and is_file
                and entry.name.endswith(".md")
            ):
                entries.append(full_path)
                continue

            if not is_dir:
                continue

            entries.extend(_collect_skill_entries(full_path, mode, ig, root))
    except OSError:
        pass

    return entries


def _find_git_repo_root(start_dir: str) -> str | None:
    dir_ = os.path.abspath(start_dir)
    while True:
        if Path(dir_, ".git").exists():
            return dir_
        parent = os.path.dirname(dir_)
        if parent == dir_:
            return None
        dir_ = parent


def _collect_ancestor_agents_skill_dirs(start_dir: str) -> list[str]:
    skill_dirs: list[str] = []
    resolved_start = os.path.abspath(start_dir)
    git_repo_root = _find_git_repo_root(resolved_start)

    dir_ = resolved_start
    while True:
        skill_dirs.append(str(Path(dir_) / ".agents" / "skills"))
        if git_repo_root and dir_ == git_repo_root:
            break
        parent = os.path.dirname(dir_)
        if parent == dir_:
            break
        dir_ = parent

    return skill_dirs


def _collect_auto_prompt_entries(dir_path: str) -> list[str]:
    entries: list[str] = []
    d = Path(dir_path)
    if not d.exists():
        return entries

    try:
        for entry in d.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue

            full_path = str(entry.resolve())
            is_file = entry.is_file()
            if entry.is_symlink():
                try:
                    is_file = stat.S_ISREG(entry.stat().st_mode)
                except OSError:
                    continue

            if is_file and entry.name.endswith(".md"):
                entries.append(full_path)
    except OSError:
        pass

    return entries


def _collect_auto_theme_entries(dir_path: str) -> list[str]:
    entries: list[str] = []
    d = Path(dir_path)
    if not d.exists():
        return entries

    try:
        for entry in d.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue

            full_path = str(entry.resolve())
            is_file = entry.is_file()
            if entry.is_symlink():
                try:
                    is_file = stat.S_ISREG(entry.stat().st_mode)
                except OSError:
                    continue

            if is_file and entry.name.endswith(".json"):
                entries.append(full_path)
    except OSError:
        pass

    return entries


def _resolve_extension_entries(dir_path: str) -> list[str] | None:
    package_json_path = Path(dir_path) / "package.json"
    if package_json_path.exists():
        manifest = read_pi_manifest(str(package_json_path))
        if manifest and manifest.extensions:
            entries: list[str] = []
            for ext_path in manifest.extensions:
                resolved_ext = str(Path(dir_path, ext_path).resolve())
                if Path(resolved_ext).exists():
                    entries.append(resolved_ext)
            if entries:
                return entries

    index_ts = Path(dir_path) / "index.ts"
    index_js = Path(dir_path) / "index.js"
    if index_ts.exists():
        return [str(index_ts)]
    if index_js.exists():
        return [str(index_js)]

    return None


def _collect_auto_extension_entries(dir_path: str) -> list[str]:
    entries: list[str] = []
    d = Path(dir_path)
    if not d.exists():
        return entries

    root_entries = _resolve_extension_entries(dir_path)
    if root_entries:
        return root_entries

    try:
        for entry in d.iterdir():
            if entry.name.startswith("."):
                continue
            if entry.name == "node_modules":
                continue

            full_path = str(entry.resolve())
            is_dir = entry.is_dir()
            is_file = entry.is_file()

            if entry.is_symlink():
                try:
                    stats = entry.stat()
                    is_dir = stat.S_ISDIR(stats.st_mode)
                    is_file = stat.S_ISREG(stats.st_mode)
                except OSError:
                    continue

            if is_file and (entry.name.endswith(".ts") or entry.name.endswith(".js")):
                entries.append(full_path)
            elif is_dir:
                resolved = _resolve_extension_entries(full_path)
                if resolved:
                    entries.extend(resolved)
    except OSError:
        pass

    return entries


def _collect_resource_files(dir_path: str, resource_type: ResourceType) -> list[str]:
    if resource_type == "skills":
        return _collect_skill_entries(dir_path, "pi")
    if resource_type == "extensions":
        return _collect_auto_extension_entries(dir_path)
    return _collect_files(dir_path, FILE_PATTERNS[resource_type])


def _matches_any_pattern(file_path: str, patterns: list[str], base_dir: str) -> bool:
    from fnmatch import fnmatch

    rel = to_posix_path(os.path.relpath(file_path, base_dir))
    name = os.path.basename(file_path)
    file_path_posix = to_posix_path(file_path)
    is_skill_file = name == "SKILL.md"
    parent_dir = os.path.dirname(file_path) if is_skill_file else None
    parent_rel = (
        to_posix_path(os.path.relpath(parent_dir, base_dir)) if parent_dir else None
    )
    parent_name = os.path.basename(parent_dir) if parent_dir else None
    parent_dir_posix = to_posix_path(parent_dir) if parent_dir else None

    for pattern in patterns:
        normalized = to_posix_path(pattern)
        if (
            fnmatch(rel, normalized)
            or fnmatch(name, normalized)
            or fnmatch(file_path_posix, normalized)
        ):
            return True
        if not is_skill_file:
            continue
        if parent_rel and fnmatch(parent_rel, normalized):
            return True
        if parent_name and fnmatch(parent_name, normalized):
            return True
        if parent_dir_posix and fnmatch(parent_dir_posix, normalized):
            return True
    return False


def _normalize_exact_pattern(pattern: str) -> str:
    normalized = pattern[2:] if pattern.startswith(("./", ".\\")) else pattern
    return to_posix_path(normalized)


def _matches_any_exact_pattern(
    file_path: str, patterns: list[str], base_dir: str
) -> bool:
    if not patterns:
        return False
    rel = to_posix_path(os.path.relpath(file_path, base_dir))
    name = os.path.basename(file_path)
    file_path_posix = to_posix_path(file_path)
    is_skill_file = name == "SKILL.md"
    parent_dir = os.path.dirname(file_path) if is_skill_file else None
    parent_rel = (
        to_posix_path(os.path.relpath(parent_dir, base_dir)) if parent_dir else None
    )
    parent_dir_posix = to_posix_path(parent_dir) if parent_dir else None

    for pattern in patterns:
        normalized = _normalize_exact_pattern(pattern)
        if normalized == rel or normalized == file_path_posix:
            return True
        if not is_skill_file:
            continue
        if parent_rel and normalized == parent_rel:
            return True
        if parent_dir_posix and normalized == parent_dir_posix:
            return True
    return False


def _get_override_patterns(entries: list[str]) -> list[str]:
    return [p for p in entries if p.startswith(("!", "+", "-"))]


def _is_enabled_by_overrides(
    file_path: str, patterns: list[str], base_dir: str
) -> bool:
    overrides = _get_override_patterns(patterns)
    excludes = [p[1:] for p in overrides if p.startswith("!")]
    force_includes = [p[1:] for p in overrides if p.startswith("+")]
    force_excludes = [p[1:] for p in overrides if p.startswith("-")]

    enabled = True
    if excludes and _matches_any_pattern(file_path, excludes, base_dir):
        enabled = False
    if force_includes and _matches_any_exact_pattern(
        file_path, force_includes, base_dir
    ):
        enabled = True
    if force_excludes and _matches_any_exact_pattern(
        file_path, force_excludes, base_dir
    ):
        enabled = False
    return enabled


def _apply_patterns(
    all_paths: list[str], patterns: list[str], base_dir: str
) -> set[str]:
    includes: list[str] = []
    excludes: list[str] = []
    force_includes: list[str] = []
    force_excludes: list[str] = []

    for p in patterns:
        if p.startswith("+"):
            force_includes.append(p[1:])
        elif p.startswith("-"):
            force_excludes.append(p[1:])
        elif p.startswith("!"):
            excludes.append(p[1:])
        else:
            includes.append(p)

    if not includes:
        result = list(all_paths)
    else:
        result = [f for f in all_paths if _matches_any_pattern(f, includes, base_dir)]

    if excludes:
        result = [f for f in result if not _matches_any_pattern(f, excludes, base_dir)]

    if force_includes:
        for f in all_paths:
            if f not in result and _matches_any_exact_pattern(
                f, force_includes, base_dir
            ):
                result.append(f)

    if force_excludes:
        result = [
            f
            for f in result
            if not _matches_any_exact_pattern(f, force_excludes, base_dir)
        ]

    return set(result)


def _apply_autoload_disabled_patterns(
    all_paths: list[str], patterns: list[str], base_dir: str
) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for pattern in patterns:
        offset = 1 if pattern.startswith(("+", "-", "!")) else 0
        target = pattern[offset:]
        enabled = not pattern.startswith(("-", "!"))
        exact = pattern.startswith(("+", "-"))
        for file_path in all_paths:
            if exact:
                match = _matches_any_exact_pattern(file_path, [target], base_dir)
            else:
                match = _matches_any_pattern(file_path, [target], base_dir)
            if match:
                result[file_path] = enabled
    return result


def _get_env() -> dict[str, str]:
    if os.name != "posix" or os.environ:
        return dict(os.environ)
    try:
        data = Path("/proc/self/environ").read_text("utf-8")
        env: dict[str, str] = {}
        for entry in data.split("\0"):
            idx = entry.find("=")
            if idx > 0:
                env[entry[:idx]] = entry[idx + 1 :]
        return env
    except OSError:
        return dict(os.environ)


class DefaultPackageManager(PackageManager):
    def __init__(self, options: dict[str, Any]) -> None:
        self._cwd: str = resolve_path(options["cwd"])
        self._agent_dir: str = resolve_path(options["agent_dir"])
        self._settings_manager: Any = options["settings_manager"]
        self._global_npm_root: str | None = None
        self._global_npm_root_command_key: str | None = None
        self._progress_callback: Callable[[ProgressEvent], None] | None = None

    def set_progress_callback(
        self, callback: Callable[[ProgressEvent], None] | None
    ) -> None:
        self._progress_callback = callback

    def _emit_progress(self, event: ProgressEvent) -> None:
        if self._progress_callback:
            self._progress_callback(event)

    async def _with_progress(
        self, action: str, source: str, message: str, operation: Callable[[], Any]
    ) -> None:
        self._emit_progress(
            ProgressEvent(type="start", action=action, source=source, message=message)
        )
        try:
            await operation()
            self._emit_progress(
                ProgressEvent(type="complete", action=action, source=source)
            )
        except Exception as error:
            error_message = str(error)
            self._emit_progress(
                ProgressEvent(
                    type="error", action=action, source=source, message=error_message
                )
            )
            raise

    def add_source_to_settings(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        scope: SourceScope = "project" if (options and options.get("local")) else "user"
        current_settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        current_packages = current_settings.get("packages") or []
        normalized_source = self._normalize_package_source_for_settings(source, scope)
        match_index = next(
            (
                i
                for i, existing in enumerate(current_packages)
                if self._package_sources_match(existing, source, scope)
            ),
            -1,
        )
        if match_index != -1:
            existing = current_packages[match_index]
            if self._get_package_source_string(existing) == normalized_source:
                return False
            next_packages = list(current_packages)
            if isinstance(existing, str):
                next_packages[match_index] = normalized_source
            else:
                next_packages[match_index] = {**existing, "source": normalized_source}
            if scope == "project":
                self._settings_manager.set_project_packages(next_packages)
            else:
                self._settings_manager.set_packages(next_packages)
            return True

        next_packages = list(current_packages) + [normalized_source]
        if scope == "project":
            self._settings_manager.set_project_packages(next_packages)
        else:
            self._settings_manager.set_packages(next_packages)
        return True

    def remove_source_from_settings(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        scope: SourceScope = "project" if (options and options.get("local")) else "user"
        current_settings = (
            self._settings_manager.get_project_settings()
            if scope == "project"
            else self._settings_manager.get_global_settings()
        )
        current_packages = current_settings.get("packages") or []
        next_packages = [
            existing
            for existing in current_packages
            if not self._package_sources_match(existing, source, scope)
        ]
        changed = len(next_packages) != len(current_packages)
        if not changed:
            return False
        if scope == "project":
            self._settings_manager.set_project_packages(next_packages)
        else:
            self._settings_manager.set_packages(next_packages)
        return True

    def get_installed_path(self, source: str, scope: str) -> str | None:
        parsed = self._parse_source(source)
        if parsed.type == "npm":
            path = self._get_npm_install_path(cast(NpmSource, parsed), scope)
            return path if Path(path).exists() else None
        if parsed.type == "git":
            path = self._get_git_install_path(cast(Any, parsed), scope)
            return path if Path(path).exists() else None
        if parsed.type == "local":
            base_dir = self._get_base_dir_for_scope(scope)
            path = self._resolve_path_from_base(
                cast(LocalSource, parsed).path, base_dir
            )
            return path if Path(path).exists() else None
        return None

    async def resolve(
        self, on_missing: Callable[[str], Awaitable[str]] | None = None
    ) -> ResolvedPaths:
        accumulator = self._create_accumulator()
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()

        all_packages: list[dict[str, Any]] = []
        for pkg in project_settings.get("packages") or []:
            all_packages.append({"pkg": pkg, "scope": "project"})
        for pkg in global_settings.get("packages") or []:
            all_packages.append({"pkg": pkg, "scope": "user"})

        package_sources = self._dedupe_packages(all_packages)
        await self._resolve_package_sources(package_sources, accumulator, on_missing)

        global_base_dir = self._agent_dir
        project_base_dir = str(Path(self._cwd) / CONFIG_DIR_NAME)

        for resource_type in RESOURCE_TYPES:
            target = self._get_target_map(accumulator, resource_type)
            global_entries = global_settings.get(resource_type) or []
            project_entries = project_settings.get(resource_type) or []
            self._resolve_local_entries(
                project_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="project", origin="top-level"),
                project_base_dir,
            )
            self._resolve_local_entries(
                global_entries,
                resource_type,
                target,
                PathMetadata(source="local", scope="user", origin="top-level"),
                global_base_dir,
            )

        self._add_auto_discovered_resources(
            accumulator,
            global_settings,
            project_settings,
            global_base_dir,
            project_base_dir,
        )
        return self._to_resolved_paths(accumulator)

    async def resolve_extension_sources(
        self, sources: list[str], options: dict[str, Any] | None = None
    ) -> ResolvedPaths:
        accumulator = self._create_accumulator()
        scope: SourceScope = (
            "temporary"
            if (options and options.get("temporary"))
            else "project"
            if (options and options.get("local"))
            else "user"
        )
        package_sources = [{"pkg": s, "scope": scope} for s in sources]
        await self._resolve_package_sources(package_sources, accumulator)
        return self._to_resolved_paths(accumulator)

    def list_configured_packages(self) -> list[ConfiguredPackage]:
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        configured: list[ConfiguredPackage] = []

        for pkg in global_settings.get("packages") or []:
            source = pkg if isinstance(pkg, str) else pkg["source"]
            configured.append(
                ConfiguredPackage(
                    source=source,
                    scope="user",
                    filtered=not isinstance(pkg, str),
                    installed_path=self.get_installed_path(source, "user"),
                )
            )

        for pkg in project_settings.get("packages") or []:
            source = pkg if isinstance(pkg, str) else pkg["source"]
            configured.append(
                ConfiguredPackage(
                    source=source,
                    scope="project",
                    filtered=not isinstance(pkg, str),
                    installed_path=self.get_installed_path(source, "project"),
                )
            )

        return configured

    async def install(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options and options.get("local")) else "user"
        self._assert_project_trusted_for_scope(scope)
        await self._with_progress(
            "install",
            source,
            f"Installing {source}...",
            lambda: self._do_install(parsed, scope, source),
        )

    async def _do_install(
        self, parsed: ParsedSource, scope: SourceScope, source: str
    ) -> None:
        if parsed.type == "npm":
            await self._install_npm(cast(NpmSource, parsed), scope, False)
        elif parsed.type == "git":
            await self._install_git(cast(Any, parsed), scope)
        elif parsed.type == "local":
            resolved = self._resolve_path(cast(LocalSource, parsed).path)
            if not Path(resolved).exists():
                raise FileNotFoundError(f"Path does not exist: {resolved}")
        else:
            raise ValueError(f"Unsupported install source: {source}")

    async def install_and_persist(
        self, source: str, options: dict[str, Any] | None = None
    ) -> None:
        await self.install(source, options)
        self.add_source_to_settings(source, options)

    async def remove(self, source: str, options: dict[str, Any] | None = None) -> None:
        parsed = self._parse_source(source)
        scope: SourceScope = "project" if (options and options.get("local")) else "user"
        self._assert_project_trusted_for_scope(scope)
        await self._with_progress(
            "remove",
            source,
            f"Removing {source}...",
            lambda: self._do_remove(parsed, scope, source),
        )

    async def _do_remove(
        self, parsed: ParsedSource, scope: SourceScope, source: str
    ) -> None:
        if parsed.type == "npm":
            await self._uninstall_npm(cast(NpmSource, parsed), scope)
        elif parsed.type == "git":
            await self._remove_git(cast(Any, parsed), scope)
        elif parsed.type == "local":
            pass
        else:
            raise ValueError(f"Unsupported remove source: {source}")

    async def remove_and_persist(
        self, source: str, options: dict[str, Any] | None = None
    ) -> bool:
        await self.remove(source, options)
        return self.remove_source_from_settings(source, options)

    async def update(self, source: str | None = None) -> None:
        global_settings = self._settings_manager.get_global_settings()
        project_settings = self._settings_manager.get_project_settings()
        identity = self._get_package_identity(source) if source else None
        matched = False
        update_sources: list[ConfiguredUpdateSource] = []

        for pkg in global_settings.get("packages") or []:
            source_str = pkg if isinstance(pkg, str) else pkg["source"]
            if identity and self._get_package_identity(source_str, "user") != identity:
                continue
            matched = True
            update_sources.append(
                ConfiguredUpdateSource(source=source_str, scope="user")
            )

        for pkg in project_settings.get("packages") or []:
            source_str = pkg if isinstance(pkg, str) else pkg["source"]
            if (
                identity
                and self._get_package_identity(source_str, "project") != identity
            ):
                continue
            matched = True
            update_sources.append(
                ConfiguredUpdateSource(source=source_str, scope="project")
            )

        if source and not matched:
            raise ValueError(
                self._build_no_matching_package_message(
                    source,
                    list(global_settings.get("packages") or [])
                    + list(project_settings.get("packages") or []),
                )
            )

        await self._update_configured_sources(update_sources)

    async def _update_configured_sources(
        self, sources: list[ConfiguredUpdateSource]
    ) -> None:
        if is_offline_mode_enabled() or not sources:
            return

        npm_candidates: list[NpmUpdateTarget] = []
        git_candidates: list[GitUpdateTarget] = []

        for entry in sources:
            parsed = self._parse_source(entry.source)
            if parsed.type == "npm":
                npm_parsed = cast(NpmSource, parsed)
                if not npm_parsed.pinned:
                    npm_candidates.append(
                        NpmUpdateTarget(
                            source=entry.source, scope=entry.scope, parsed=npm_parsed
                        )
                    )
            elif parsed.type == "git":
                git_candidates.append(
                    GitUpdateTarget(
                        source=entry.source, scope=entry.scope, parsed=cast(Any, parsed)
                    )
                )

        npm_check_tasks = [
            lambda e=entry: self._should_update_npm_source(e.parsed, e.scope)
            for entry in npm_candidates
        ]
        npm_check_results = await self._run_with_concurrency(
            npm_check_tasks, UPDATE_CHECK_CONCURRENCY
        )
        user_npm_updates: list[NpmUpdateTarget] = []
        project_npm_updates: list[NpmUpdateTarget] = []
        for i, should_update in enumerate(npm_check_results):
            if not should_update:
                continue
            if npm_candidates[i].scope == "user":
                user_npm_updates.append(npm_candidates[i])
            else:
                project_npm_updates.append(npm_candidates[i])

        tasks: list[asyncio.Task[Any]] = []
        if user_npm_updates:
            tasks.append(
                asyncio.ensure_future(self._update_npm_batch(user_npm_updates, "user"))
            )
        if project_npm_updates:
            tasks.append(
                asyncio.ensure_future(
                    self._update_npm_batch(project_npm_updates, "project")
                )
            )
        if git_candidates:
            git_tasks = [
                lambda e=entry: self._with_progress(
                    "update",
                    e.source,
                    f"Updating {e.source}...",
                    lambda: self._update_git(e.parsed, e.scope),
                )
                for entry in git_candidates
            ]
            tasks.append(
                asyncio.ensure_future(
                    self._run_with_concurrency(git_tasks, GIT_UPDATE_CONCURRENCY)
                )
            )

        if tasks:
            await asyncio.gather(*tasks)

    async def _should_update_npm_source(
        self, source: NpmSource, scope: InstalledSourceScope
    ) -> bool:
        installed_path = self._get_managed_npm_install_path(source, scope)
        installed_version = (
            self._get_installed_npm_version(installed_path)
            if Path(installed_path).exists()
            else None
        )
        if not installed_version:
            return True
        try:
            target_version = await self._get_latest_npm_version(
                source.spec if source.version else source.name, source.range
            )
            return target_version != installed_version
        except Exception:
            return True

    async def _update_npm_batch(
        self, sources: list[NpmUpdateTarget], scope: InstalledSourceScope
    ) -> None:
        if not sources:
            return
        specs = [
            (
                e.parsed.spec
                if e.parsed and e.parsed.version
                else f"{e.parsed.name if e.parsed else e.source}@latest"
            )
            for e in sources
        ]
        source_label = (
            sources[0].source if len(sources) == 1 else f"{scope} npm packages"
        )
        message = (
            f"Updating {sources[0].source}..."
            if len(sources) == 1
            else f"Updating {scope} npm packages..."
        )
        await self._with_progress(
            "update",
            source_label,
            message,
            lambda: self._install_npm_batch(specs, scope),
        )

    async def _install_npm_batch(
        self, specs: list[str], scope: InstalledSourceScope
    ) -> None:
        install_root = self._get_npm_install_root(scope, False)
        self._ensure_npm_project(install_root)
        await self._run_npm_command(self._get_npm_install_args(specs, install_root))

    # Private helpers
    def _parse_source(self, source: str) -> ParsedSource:
        if source.startswith("npm:"):
            spec = source[4:].strip()
            name, version = self._parse_npm_spec(spec)
            return NpmSource(
                type="npm",
                spec=spec,
                name=name,
                version=version,
                range=get_npm_version_range(version),
                pinned=is_exact_npm_version(version),
            )
        if is_local_path(source):
            return LocalSource(type="local", path=source)
        git_parsed = parse_git_url(source)
        if git_parsed:
            return git_parsed
        return LocalSource(type="local", path=source)

    def _parse_npm_spec(self, spec: str) -> tuple[str, str | None]:
        match = re.match(r"^(@?[^@]+(?:/[^@]+)?)(?:@(.+))?$", spec)
        if not match:
            return spec, None
        name = match.group(1) or spec
        version = match.group(2)
        return name, version

    def _get_package_source_string(self, pkg: Any) -> str:
        return pkg if isinstance(pkg, str) else pkg["source"]

    def _get_source_match_key_for_input(self, source: str) -> str:
        parsed = self._parse_source(source)
        if parsed.type == "npm":
            npm_p = cast(NpmSource, parsed)
            return f"npm:{npm_p.name}"
        if parsed.type == "git":
            git_p = cast(Any, parsed)
            return f"git:{git_p.host}/{git_p.path}"
        local_p = cast(LocalSource, parsed)
        return f"local:{self._resolve_path(local_p.path)}"

    def _get_source_match_key_for_settings(
        self, source: str, scope: SourceScope
    ) -> str:
        parsed = self._parse_source(source)
        if parsed.type == "npm":
            npm_p = cast(NpmSource, parsed)
            return f"npm:{npm_p.name}"
        if parsed.type == "git":
            git_p = cast(Any, parsed)
            return f"git:{git_p.host}/{git_p.path}"
        base_dir = self._get_base_dir_for_scope(scope)
        local_p = cast(LocalSource, parsed)
        return f"local:{self._resolve_path_from_base(local_p.path, base_dir)}"

    def _package_sources_match(
        self, existing: Any, input_source: str, scope: SourceScope
    ) -> bool:
        left = self._get_source_match_key_for_settings(
            self._get_package_source_string(existing), scope
        )
        right = self._get_source_match_key_for_input(input_source)
        return left == right

    def _normalize_package_source_for_settings(
        self, source: str, scope: SourceScope
    ) -> str:
        parsed = self._parse_source(source)
        if parsed.type != "local":
            return source
        base_dir = self._get_base_dir_for_scope(scope)
        local_p = cast(LocalSource, parsed)
        resolved = self._resolve_path(local_p.path)
        rel = os.path.relpath(resolved, base_dir)
        return rel or "."

    def _get_package_identity(self, source: str, scope: str | None = None) -> str:
        parsed = self._parse_source(source)
        if parsed.type == "npm":
            npm_p = cast(NpmSource, parsed)
            return f"npm:{npm_p.name}"
        if parsed.type == "git":
            git_p = cast(Any, parsed)
            return f"git:{git_p.host}/{git_p.path}"
        local_p = cast(LocalSource, parsed)
        if scope:
            base_dir = self._get_base_dir_for_scope(scope)
            return f"local:{self._resolve_path_from_base(local_p.path, base_dir)}"
        return f"local:{self._resolve_path(local_p.path)}"

    def _dedupe_packages(self, packages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for entry in packages:
            identity = self._get_package_identity(
                self._get_package_source_string(entry["pkg"]), entry["scope"]
            )
            index = seen.get(identity)
            if index is None:
                seen[identity] = len(result)
                result.append(entry)
                continue
            existing = result[index]
            if existing.get("scope") == "project" and entry.get("scope") == "user":
                if (
                    isinstance(existing["pkg"], dict)
                    and existing["pkg"].get("autoload") is False
                ):
                    result.append(entry)
            elif entry.get("scope") == "project":
                result[index] = entry
        return result

    def _assert_project_trusted_for_scope(self, scope: SourceScope) -> None:
        if scope == "project" and not self._settings_manager.is_project_trusted():
            raise PermissionError(
                "Project is not trusted; refusing to access project package storage"
            )

    def _get_base_dir_for_scope(self, scope: SourceScope) -> str:
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return str(Path(self._cwd) / CONFIG_DIR_NAME)
        if scope == "user":
            return self._agent_dir
        return self._cwd

    def _resolve_path(self, input_path: str) -> str:
        return resolve_path(
            input_path, self._cwd, {"home_dir": get_home_dir(), "trim": True}
        )

    def _resolve_path_from_base(self, input_path: str, base_dir: str) -> str:
        return resolve_path(
            input_path, base_dir, {"home_dir": get_home_dir(), "trim": True}
        )

    # npm helpers
    def _get_npm_command(self) -> tuple[str, list[str]]:
        configured_command = self._settings_manager.get_npm_command()
        if not configured_command:
            return "npm", []
        command = configured_command[0]
        args = list(configured_command[1:]) if len(configured_command) > 1 else []
        return command, args

    def _get_package_manager_name(self) -> str:
        cmd, args = self._get_npm_command()
        command_parts = [cmd] + args
        separator_index = -1
        for i, part in enumerate(command_parts):
            if part == "--":
                separator_index = i
        pm_cmd = command_parts[separator_index + 1] if separator_index >= 0 else cmd
        return os.path.basename(pm_cmd).replace(".cmd", "").replace(".exe", "")

    async def _run_npm_command(
        self, args: list[str], options: dict[str, Any] | None = None
    ) -> None:
        cmd, cmd_args = self._get_npm_command()
        full_args = cmd_args + args
        await self._run_command(cmd, full_args, options)

    def _run_npm_command_sync(self, args: list[str]) -> str:
        cmd, cmd_args = self._get_npm_command()
        return self._run_command_sync(cmd, cmd_args + args)

    def _get_git_dependency_install_args(self) -> list[str]:
        configured_command = self._settings_manager.get_npm_command()
        if configured_command:
            return ["install"]
        return ["install", "--omit=dev"]

    def _get_npm_install_args(self, specs: list[str], install_root: str) -> list[str]:
        pm_name = self._get_package_manager_name()
        if pm_name == "bun":
            return ["install"] + specs + ["--cwd", install_root, "--omit=peer"]
        if pm_name == "pnpm":
            return [
                "install",
                *specs,
                "--prefix",
                install_root,
                "--config.auto-install-peers=false",
                "--config.strict-peer-dependencies=false",
                "--config.strict-dep-builds=false",
            ]
        return ["install"] + specs + ["--prefix", install_root, "--legacy-peer-deps"]

    async def _install_npm(
        self, source: NpmSource, scope: SourceScope, temporary: bool
    ) -> None:
        install_root = self._get_npm_install_root(scope, temporary)
        self._ensure_npm_project(install_root)
        await self._run_npm_command(
            self._get_npm_install_args([source.spec], install_root)
        )

    async def _uninstall_npm(self, source: NpmSource, scope: SourceScope) -> None:
        install_root = self._get_npm_install_root(scope, False)
        if not Path(install_root).exists():
            return
        pm_name = self._get_package_manager_name()
        if pm_name == "bun":
            await self._run_npm_command(
                ["uninstall", source.name, "--cwd", install_root]
            )
            return
        args = ["uninstall", source.name, "--prefix", install_root]
        if pm_name != "pnpm":
            args.append("--legacy-peer-deps")
        await self._run_npm_command(args)

    def _get_npm_install_root(self, scope: SourceScope, temporary: bool) -> str:
        if temporary:
            return self._get_temporary_dir("npm")
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return str(Path(self._cwd) / CONFIG_DIR_NAME / "npm")
        return str(Path(self._agent_dir) / "npm")

    def _get_managed_npm_install_path(
        self, source: NpmSource, scope: SourceScope
    ) -> str:
        if scope == "temporary":
            return str(
                Path(self._get_temporary_dir("npm")) / "node_modules" / source.name
            )
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return str(
                Path(self._cwd) / CONFIG_DIR_NAME / "npm" / "node_modules" / source.name
            )
        return str(Path(self._agent_dir) / "npm" / "node_modules" / source.name)

    def _get_npm_install_path(self, source: NpmSource, scope: SourceScope) -> str:
        managed_path = self._get_managed_npm_install_path(source, scope)
        if scope != "user" or Path(managed_path).exists():
            return managed_path
        legacy_path = self._get_legacy_global_npm_install_path(source)
        return (
            legacy_path if legacy_path and Path(legacy_path).exists() else managed_path
        )

    def _get_legacy_global_npm_install_path(self, source: NpmSource) -> str | None:
        try:
            pnpm_path = self._get_pnpm_global_package_path(source.name)
            if pnpm_path:
                return pnpm_path
            return str(Path(self._get_global_npm_root()) / source.name)
        except Exception:
            return None

    def _get_global_npm_root(self) -> str:
        cmd, cmd_args = self._get_npm_command()
        command_key = "\0".join([cmd] + cmd_args)
        if self._global_npm_root and self._global_npm_root_command_key == command_key:
            return self._global_npm_root
        if self._get_package_manager_name() == "bun":
            bin_dir = self._run_npm_command_sync(["pm", "bin", "-g"]).strip()
            self._global_npm_root = str(
                Path(os.path.dirname(bin_dir)) / "install" / "global" / "node_modules"
            )
        else:
            self._global_npm_root = self._run_npm_command_sync(["root", "-g"]).strip()
        self._global_npm_root_command_key = command_key
        return self._global_npm_root

    def _get_pnpm_global_package_path(self, package_name: str) -> str | None:
        if self._get_package_manager_name() != "pnpm":
            return None
        output = self._run_npm_command_sync(["list", "-g", "--depth", "0", "--json"])
        entries = json.loads(output)
        for entry in entries:
            dep_path: str | None = (
                entry.get("dependencies", {}).get(package_name, {}).get("path")
            )
            if dep_path:
                return dep_path
        return None

    # git helpers
    async def _install_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if Path(target_dir).exists():
            if source.ref:
                await self._ensure_git_ref(
                    target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD"
                )
                return
            target = await self._get_local_git_update_target(target_dir)
            await self._ensure_git_ref(target_dir, target["fetch_args"], target["ref"])
            return
        git_root = self._get_git_install_root(scope)
        if git_root:
            self._ensure_git_ignore(git_root)
        Path(os.path.dirname(target_dir)).mkdir(parents=True, exist_ok=True)
        marker_path = self._get_git_update_marker_path(target_dir)
        if Path(marker_path).exists():
            Path(marker_path).unlink()

        try:
            await self._run_command("git", ["clone", source.repo, target_dir])
            if source.ref:
                await self._run_command(
                    "git", ["checkout", source.ref], {"cwd": target_dir}
                )
            package_json = Path(target_dir) / "package.json"
            if package_json.exists():
                await self._run_npm_command(
                    self._get_git_dependency_install_args(), {"cwd": target_dir}
                )
        except Exception:
            shutil.rmtree(target_dir, ignore_errors=True)
            self._prune_empty_git_parents(target_dir, git_root)
            raise

    async def _update_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        if not Path(target_dir).exists():
            await self._install_git(source, scope)
            return
        if source.ref:
            await self._ensure_git_ref(
                target_dir, ["fetch", "origin", source.ref], "FETCH_HEAD"
            )
            return
        target = await self._get_local_git_update_target(target_dir)
        await self._ensure_git_ref(target_dir, target["fetch_args"], target["ref"])

    async def _remove_git(self, source: GitSource, scope: SourceScope) -> None:
        target_dir = self._get_git_install_path(source, scope)
        shutil.rmtree(target_dir, ignore_errors=True)
        marker_path = self._get_git_update_marker_path(target_dir)
        if Path(marker_path).exists():
            Path(marker_path).unlink()
        self._prune_empty_git_parents(target_dir, self._get_git_install_root(scope))

    def _get_git_install_path(self, source: GitSource, scope: SourceScope) -> str:
        if scope == "temporary":
            return self._get_temporary_dir(f"git-{source.host}", source.path)
        install_root = self._get_git_install_root(scope)
        if not install_root:
            raise ValueError("Missing git install root")
        return self._resolve_managed_path(install_root, source.host, source.path)

    def _get_git_install_root(self, scope: SourceScope) -> str | None:
        if scope == "temporary":
            return None
        if scope == "project":
            self._assert_project_trusted_for_scope(scope)
            return str(Path(self._cwd) / CONFIG_DIR_NAME / "git")
        return str(Path(self._agent_dir) / "git")

    def _get_git_update_marker_path(self, target_dir: str) -> str:
        return str(
            Path(os.path.dirname(target_dir))
            / f".{os.path.basename(target_dir)}.pi-update-incomplete"
        )

    def _ensure_git_ignore(self, dir_path: str) -> None:
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        ignore_path = d / ".gitignore"
        if not ignore_path.exists():
            ignore_path.write_text("*\n!.gitignore\n", "utf-8")

    def _get_temporary_dir(self, prefix: str, suffix: str | None = None) -> str:
        root = self._resolve_managed_path(
            get_extension_temp_folder(self._agent_dir), prefix
        )
        hash_ = hashlib.sha256(f"{prefix}-{suffix or ''}".encode()).hexdigest()[:8]
        return self._resolve_managed_path(root, hash_, suffix or "")

    def _resolve_managed_path(self, root: str, *parts: str) -> str:
        resolved_root = os.path.abspath(root)
        resolved_path = os.path.abspath(os.path.join(resolved_root, *parts))
        if resolved_path != resolved_root and not resolved_path.startswith(
            f"{resolved_root}{os.sep}"
        ):
            raise ValueError(
                f"Refusing to use path outside package install root: {resolved_path}"
            )
        return resolved_path

    def _prune_empty_git_parents(
        self, target_dir: str, install_root: str | None
    ) -> None:
        if not install_root:
            return
        resolved_root = os.path.abspath(install_root)
        current = os.path.dirname(target_dir)
        while current.startswith(resolved_root) and current != resolved_root:
            if not Path(current).exists():
                current = os.path.dirname(current)
                continue
            entries = list(Path(current).iterdir())
            if entries:
                break
            try:
                shutil.rmtree(current, ignore_errors=True)
            except OSError:
                break
            current = os.path.dirname(current)

    async def _get_local_git_update_target(self, installed_path: str) -> dict[str, Any]:
        try:
            upstream = await self._run_command_capture(
                "git",
                ["rev-parse", "--abbrev-ref", "@{upstream}"],
                {"cwd": installed_path, "timeout_ms": NETWORK_TIMEOUT_MS},
            )
            trimmed = upstream.strip()
            if not trimmed.startswith("origin/"):
                raise ValueError(f"Unsupported upstream remote: {trimmed}")
            branch = trimmed[7:]
            if not branch:
                raise ValueError("Missing upstream branch name")
            head = await self._run_command_capture(
                "git",
                ["rev-parse", "@{upstream}"],
                {"cwd": installed_path, "timeout_ms": NETWORK_TIMEOUT_MS},
            )
            return {
                "ref": "@{upstream}",
                "head": head,
                "fetch_args": [
                    "fetch",
                    "--prune",
                    "--no-tags",
                    "origin",
                    f"+refs/heads/{branch}:refs/remotes/origin/{branch}",
                ],
            }
        except Exception:
            await self._run_command(
                "git", ["remote", "set-head", "origin", "-a"], {"cwd": installed_path}
            )
            head = await self._run_command_capture(
                "git",
                ["rev-parse", "origin/HEAD"],
                {"cwd": installed_path, "timeout_ms": NETWORK_TIMEOUT_MS},
            )
            return {
                "ref": "origin/HEAD",
                "head": head,
                "fetch_args": [
                    "fetch",
                    "--prune",
                    "--no-tags",
                    "origin",
                    "+HEAD:refs/remotes/origin/HEAD",
                ],
            }

    async def _ensure_git_ref(
        self, target_dir: str, fetch_args: list[str], ref: str
    ) -> None:
        await self._run_command("git", fetch_args, {"cwd": target_dir})
        local_head = await self._run_command_capture(
            "git",
            ["rev-parse", "HEAD"],
            {"cwd": target_dir, "timeout_ms": NETWORK_TIMEOUT_MS},
        )
        commit_ref = f"{ref}^{{commit}}"
        target_head = await self._run_command_capture(
            "git",
            ["rev-parse", commit_ref],
            {"cwd": target_dir, "timeout_ms": NETWORK_TIMEOUT_MS},
        )
        if local_head.strip() == target_head.strip():
            return
        await self._run_command(
            "git", ["reset", "--hard", commit_ref], {"cwd": target_dir}
        )

    # command execution
    async def _run_command(
        self, command: str, args: list[str], options: dict[str, Any] | None = None
    ) -> None:
        cwd = options.get("cwd") if options else None
        env = _get_env()
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(
                f"{command} {' '.join(args)} failed with code {proc.returncode}"
            )

    async def _run_command_capture(
        self, command: str, args: list[str], options: dict[str, Any] | None = None
    ) -> str:
        cwd = options.get("cwd") if options else None
        timeout_ms = options.get("timeout_ms") if options else None
        env = options.get("env") if options else None
        base_env = _get_env()
        if env:
            base_env.update(env)
        proc = await asyncio.create_subprocess_exec(
            command,
            *args,
            cwd=cwd,
            env=base_env,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=timeout_ms / 1000 if timeout_ms else None
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{command} {' '.join(args)} failed with code {proc.returncode}: {stderr.decode()}"
                )
            return stdout.decode().strip()
        except asyncio.TimeoutError:
            proc.kill()
            raise TimeoutError(
                f"{command} {' '.join(args)} timed out after {timeout_ms}ms"
            )

    def _run_command_sync(self, command: str, args: list[str]) -> str:
        env = _get_env()
        result = subprocess.run(
            [command] + args,
            capture_output=True,
            text=True,
            env=env,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to run {command} {' '.join(args)}: {result.stderr or result.stdout}"
            )
        return (result.stdout or result.stderr or "").strip()

    # resource resolution
    async def _resolve_package_sources(
        self,
        sources: list[dict[str, Any]],
        accumulator: dict[str, Any],
        on_missing: Callable[[str], Awaitable[str]] | None = None,
    ) -> None:
        for entry in sources:
            pkg = entry["pkg"]
            scope = entry["scope"]
            source_str = pkg if isinstance(pkg, str) else pkg["source"]
            pkg_filter = pkg if isinstance(pkg, dict) else None
            delta_base = self._find_autoload_delta_base(pkg, scope, sources)
            resolved_source = delta_base["source"] if delta_base else source_str
            resolved_scope = delta_base["scope"] if delta_base else scope
            parsed = self._parse_source(resolved_source)
            metadata = PathMetadata(source=source_str, scope=scope, origin="package")

            if parsed.type == "local":
                base_dir = self._get_base_dir_for_scope(resolved_scope)
                self._resolve_local_extension_source(
                    cast(LocalSource, parsed),
                    accumulator,
                    pkg_filter,
                    metadata,
                    base_dir,
                )
                continue

            if parsed.type == "npm":
                npm_p = cast(NpmSource, parsed)
                installed_path = self._get_npm_install_path(npm_p, resolved_scope)
                if not Path(installed_path).exists():
                    if is_offline_mode_enabled():
                        continue
                    if on_missing:
                        action = await on_missing(resolved_source)
                        if action == "skip":
                            continue
                        elif action == "error":
                            raise ValueError(f"Missing source: {resolved_source}")
                    await self._install_npm(
                        npm_p, resolved_scope, resolved_scope == "temporary"
                    )
                    installed_path = self._get_npm_install_path(npm_p, resolved_scope)
                metadata.base_dir = installed_path
                self._collect_package_resources(
                    installed_path, accumulator, pkg_filter, metadata
                )
                continue

            if parsed.type == "git":
                installed_path = self._get_git_install_path(parsed, resolved_scope)
                if not Path(installed_path).exists():
                    if is_offline_mode_enabled():
                        continue
                    if on_missing:
                        action = await on_missing(resolved_source)
                        if action == "skip":
                            continue
                        elif action == "error":
                            raise ValueError(f"Missing source: {resolved_source}")
                    await self._install_git(parsed, resolved_scope)
                metadata.base_dir = installed_path
                self._collect_package_resources(
                    installed_path, accumulator, pkg_filter, metadata
                )

    def _find_autoload_delta_base(
        self, pkg: Any, scope: SourceScope, sources: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if (
            scope != "project"
            or not isinstance(pkg, dict)
            or pkg.get("autoload") is not False
        ):
            return None
        identity = self._get_package_identity(pkg["source"], scope)
        for entry in sources:
            if entry["scope"] == "user":
                entry_identity = self._get_package_identity(
                    self._get_package_source_string(entry["pkg"]), "user"
                )
                if entry_identity == identity:
                    return {
                        "source": self._get_package_source_string(entry["pkg"]),
                        "scope": "user",
                    }
        return None

    def _resolve_local_extension_source(
        self,
        source: LocalSource,
        accumulator: dict[str, Any],
        pkg_filter: dict[str, Any] | None,
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        resolved = self._resolve_path_from_base(source.path, base_dir)
        if not Path(resolved).exists():
            return
        try:
            stats = Path(resolved).stat()
            if stats.st_mode & 0o100000:  # S_IFREG
                metadata.base_dir = os.path.dirname(resolved)
                self._add_resource(accumulator["extensions"], resolved, metadata, True)
                return
            if stats.st_mode & 0o40000:  # S_IFDIR
                metadata.base_dir = resolved
                resources = self._collect_package_resources(
                    resolved, accumulator, pkg_filter, metadata
                )
                if not resources:
                    self._add_resource(
                        accumulator["extensions"], resolved, metadata, True
                    )
        except OSError:
            return

    def _collect_package_resources(
        self,
        package_root: str,
        accumulator: dict[str, Any],
        pkg_filter: dict[str, Any] | None,
        metadata: PathMetadata,
    ) -> bool:
        if pkg_filter:
            for resource_type in RESOURCE_TYPES:
                patterns = pkg_filter.get(resource_type)
                target = self._get_target_map(accumulator, resource_type)
                if pkg_filter.get("autoload") is False:
                    self._apply_package_delta_filter(
                        package_root, patterns or [], resource_type, target, metadata
                    )
                elif patterns is not None:
                    self._apply_package_filter(
                        package_root, patterns, resource_type, target, metadata
                    )
                else:
                    self._collect_default_resources(
                        package_root, resource_type, target, metadata
                    )
            return True

        manifest = read_pi_manifest(str(Path(package_root) / "package.json"))
        if manifest:
            for resource_type in RESOURCE_TYPES:
                entries = getattr(manifest, resource_type, None)
                self._add_manifest_entries(
                    entries,
                    package_root,
                    resource_type,
                    self._get_target_map(accumulator, resource_type),
                    metadata,
                )
            return True

        has_any_dir = False
        for resource_type in RESOURCE_TYPES:
            dir_path = Path(package_root) / resource_type
            if dir_path.exists():
                files = _collect_resource_files(str(dir_path), resource_type)
                for f in files:
                    self._add_resource(
                        self._get_target_map(accumulator, resource_type),
                        f,
                        metadata,
                        True,
                    )
                has_any_dir = True
        return has_any_dir

    def _collect_default_resources(
        self,
        package_root: str,
        resource_type: ResourceType,
        target: dict[str, Any],
        metadata: PathMetadata,
    ) -> None:
        manifest = read_pi_manifest(str(Path(package_root) / "package.json"))
        if manifest:
            entries = getattr(manifest, resource_type, None)
            if entries:
                self._add_manifest_entries(
                    entries, package_root, resource_type, target, metadata
                )
                return
        dir_path = Path(package_root) / resource_type
        if dir_path.exists():
            files = _collect_resource_files(str(dir_path), resource_type)
            for f in files:
                self._add_resource(target, f, metadata, True)

    def _apply_package_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: ResourceType,
        target: dict[str, Any],
        metadata: PathMetadata,
    ) -> None:
        all_files = self._collect_manifest_files(package_root, resource_type)[
            "all_files"
        ]
        if not user_patterns:
            for f in all_files:
                self._add_resource(target, f, metadata, False)
            return
        enabled_by_user = _apply_patterns(all_files, user_patterns, package_root)
        for f in all_files:
            self._add_resource(target, f, metadata, f in enabled_by_user)

    def _apply_package_delta_filter(
        self,
        package_root: str,
        user_patterns: list[str],
        resource_type: ResourceType,
        target: dict[str, Any],
        metadata: PathMetadata,
    ) -> None:
        if not user_patterns:
            return
        all_files = self._collect_manifest_files(package_root, resource_type)[
            "all_files"
        ]
        enabled_by_user = _apply_autoload_disabled_patterns(
            all_files, user_patterns, package_root
        )
        for file_path, enabled in enabled_by_user.items():
            self._add_resource(target, file_path, metadata, enabled)

    def _collect_manifest_files(
        self, package_root: str, resource_type: ResourceType
    ) -> dict[str, Any]:
        manifest = read_pi_manifest(str(Path(package_root) / "package.json"))
        entries = getattr(manifest, resource_type, None) if manifest else None
        if entries:
            all_files = self._collect_files_from_manifest_entries(
                entries, package_root, resource_type
            )
            manifest_patterns = [e for e in entries if _is_override_pattern(e)]
            enabled_by_manifest = (
                _apply_patterns(all_files, manifest_patterns, package_root)
                if manifest_patterns
                else set(all_files)
            )
            return {
                "all_files": list(enabled_by_manifest),
                "enabled_by_manifest": enabled_by_manifest,
            }
        convention_dir = Path(package_root) / resource_type
        if not convention_dir.exists():
            return {"all_files": [], "enabled_by_manifest": set()}
        all_files = _collect_resource_files(str(convention_dir), resource_type)
        return {"all_files": all_files, "enabled_by_manifest": set(all_files)}

    def _add_manifest_entries(
        self,
        entries: list[str] | None,
        root: str,
        resource_type: ResourceType,
        target: dict[str, Any],
        metadata: PathMetadata,
    ) -> None:
        if not entries:
            return
        all_files = self._collect_files_from_manifest_entries(
            entries, root, resource_type
        )
        patterns = [e for e in entries if _is_override_pattern(e)]
        enabled_paths = _apply_patterns(all_files, patterns, root)
        for f in all_files:
            if f in enabled_paths:
                self._add_resource(target, f, metadata, True)

    def _collect_files_from_manifest_entries(
        self, entries: list[str], root: str, resource_type: ResourceType
    ) -> list[str]:
        source_entries = [e for e in entries if not _is_override_pattern(e)]
        resolved: list[str] = []
        for entry in source_entries:
            if not _has_glob_pattern(entry):
                resolved.append(str(Path(root, entry).resolve()))
            else:
                import glob

                matches = glob.glob(entry, root_dir=root, recursive=True)
                for m in matches:
                    resolved.append(str(Path(root, m).resolve()))
        return self._collect_files_from_paths(resolved, resource_type)

    def _resolve_local_entries(
        self,
        entries: list[str],
        resource_type: ResourceType,
        target: dict[str, Any],
        metadata: PathMetadata,
        base_dir: str,
    ) -> None:
        if not entries:
            return
        plain, patterns = _split_patterns(entries)
        resolved_plain = [self._resolve_path_from_base(p, base_dir) for p in plain]
        all_files = self._collect_files_from_paths(resolved_plain, resource_type)
        enabled_paths = _apply_patterns(all_files, patterns, base_dir)
        for f in all_files:
            self._add_resource(target, f, metadata, f in enabled_paths)

    def _collect_files_from_paths(
        self, paths: list[str], resource_type: ResourceType
    ) -> list[str]:
        files: list[str] = []
        for p in paths:
            path_obj = Path(p)
            if not path_obj.exists():
                continue
            try:
                if path_obj.is_file():
                    files.append(p)
                elif path_obj.is_dir():
                    files.extend(_collect_resource_files(p, resource_type))
            except OSError:
                pass
        return files

    def _add_auto_discovered_resources(
        self,
        accumulator: dict[str, Any],
        global_settings: dict[str, Any],
        project_settings: dict[str, Any],
        global_base_dir: str,
        project_base_dir: str,
    ) -> None:
        user_metadata = PathMetadata(
            source="auto", scope="user", origin="top-level", base_dir=global_base_dir
        )
        project_metadata = PathMetadata(
            source="auto",
            scope="project",
            origin="top-level",
            base_dir=project_base_dir,
        )

        user_overrides = {rt: project_settings.get(rt) or [] for rt in RESOURCE_TYPES}
        project_overrides = {rt: global_settings.get(rt) or [] for rt in RESOURCE_TYPES}
        # Simplified - full implementation would mirror the TS version

    def _create_accumulator(self) -> dict[str, Any]:
        return {
            "extensions": {},
            "skills": {},
            "prompts": {},
            "themes": {},
        }

    def _get_target_map(
        self, accumulator: dict[str, Any], resource_type: ResourceType
    ) -> dict[str, Any]:
        return cast(dict[str, Any], accumulator[resource_type])

    def _add_resource(
        self, target: dict[str, Any], path: str, metadata: PathMetadata, enabled: bool
    ) -> None:
        if not path:
            return
        if path not in target:
            target[path] = {"metadata": metadata, "enabled": enabled}

    def _to_resolved_paths(self, accumulator: dict[str, Any]) -> ResolvedPaths:
        def map_to_resolved(entries: dict[str, Any]) -> list[ResolvedResource]:
            resolved = [
                ResolvedResource(path=p, enabled=v["enabled"], metadata=v["metadata"])
                for p, v in entries.items()
            ]
            resolved.sort(
                key=lambda r: resource_precedence_rank(r.metadata or PathMetadata())
            )
            seen = set()
            result = []
            for r in resolved:
                canonical = canonicalize_path(r.path)
                if canonical in seen:
                    continue
                seen.add(canonical)
                result.append(r)
            return result

        return ResolvedPaths(
            extensions=map_to_resolved(accumulator["extensions"]),
            skills=map_to_resolved(accumulator["skills"]),
            prompts=map_to_resolved(accumulator["prompts"]),
            themes=map_to_resolved(accumulator["themes"]),
        )

    def _ensure_npm_project(self, install_root: str) -> None:
        root = Path(install_root)
        root.mkdir(parents=True, exist_ok=True)
        mark_path_ignored_by_cloud_sync(install_root)
        self._ensure_git_ignore(install_root)
        package_json = root / "package.json"
        if not package_json.exists():
            pkg = {"name": "pi-extensions", "private": True}
            package_json.write_text(json.dumps(pkg, indent=2), "utf-8")

    def _get_installed_npm_version(self, installed_path: str) -> str | None:
        package_json = Path(installed_path) / "package.json"
        if not package_json.exists():
            return None
        try:
            content = json.loads(package_json.read_text("utf-8"))
            return cast(str | None, content.get("version"))
        except (json.JSONDecodeError, OSError):
            return None

    async def _get_latest_npm_version(
        self, package_spec: str, range_: str | None = None
    ) -> str:
        cmd, cmd_args = self._get_npm_command()
        stdout = await self._run_command_capture(
            cmd,
            cmd_args + ["view", package_spec, "version", "--json"],
            {"cwd": self._cwd, "timeout_ms": NETWORK_TIMEOUT_MS},
        )
        raw = stdout.strip()
        if not raw:
            raise ValueError("Empty response from npm view")
        parsed = json.loads(raw)
        if isinstance(parsed, str):
            return parsed
        if isinstance(parsed, list):
            versions = [v for v in parsed if isinstance(v, str) and v]
            if range_:
                from packaging.specifiers import SpecifierSet

                spec = SpecifierSet(range_)
                matching = [v for v in versions if spec.contains(v)]
                if matching:
                    from packaging.version import Version

                    return str(max(Version(v) for v in matching))
            else:
                from packaging.version import Version

                return str(max(Version(v) for v in versions))
        raise ValueError("Unexpected response from npm view")

    async def _run_with_concurrency(
        self, tasks: list[Callable[..., Any]], limit: int
    ) -> list[Any]:
        if not tasks:
            return []
        results = [None] * len(tasks)
        next_index = 0
        worker_count = max(1, min(limit, len(tasks)))

        async def worker() -> None:
            nonlocal next_index
            while True:
                index = next_index
                next_index += 1
                if index >= len(tasks):
                    return
                results[index] = await tasks[index]()

        await asyncio.gather(*[worker() for _ in range(worker_count)])
        return results

    def _build_no_matching_package_message(
        self, source: str, configured_packages: list[Any]
    ) -> str:
        suggestion = self._find_suggested_configured_source(source, configured_packages)
        if not suggestion:
            return f"No matching package found for {source}"
        return f"No matching package found for {source}. Did you mean {suggestion}?"

    def _find_suggested_configured_source(
        self, source: str, configured_packages: list[Any]
    ) -> str | None:
        trimmed = source.strip()
        for pkg in configured_packages:
            source_str = self._get_package_source_string(pkg)
            parsed = self._parse_source(source_str)
            if parsed.type == "npm":
                npm_p = cast(NpmSource, parsed)
                if trimmed == npm_p.name or trimmed == npm_p.spec:
                    return source_str
            elif parsed.type == "git":
                git_p = cast(Any, parsed)
                shorthand = f"{git_p.host}/{git_p.path}"
                if trimmed == shorthand or (
                    git_p.ref and trimmed == f"{shorthand}@{git_p.ref}"
                ):
                    return source_str
        return None
