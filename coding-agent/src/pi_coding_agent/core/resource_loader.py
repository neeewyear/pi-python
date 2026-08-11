import os
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from ..config import CONFIG_DIR_NAME
from ..utils.paths import (
    canonicalize_path,
    is_local_path,
    resolve_path,
)
from .event_bus import EventBus
from .extensions.loader import (
    clear_extension_cache,
    create_extension_runtime,
    load_extension_from_factory,
    load_extensions_cached,
)
from .extensions.types import (
    Extension,
    ExtensionRuntime,
    InlineExtension,
    LoadExtensionsResult,
)
from .footer_data_provider import find_git_paths
from .package_manager import DefaultPackageManager, PathMetadata, ResolvedResource
from .prompt_templates import (
    LoadPromptTemplatesOptions,
    PromptTemplate,
    load_prompt_templates,
)
from .settings_manager import SettingsManager
from .skills import LoadSkillsOptions, Skill, SkillsResult, load_skills
from .skills import SourceInfo as SkillSourceInfo
from .source_info import SourceInfo, create_source_info
from .timings import reset_timings


class ResourceExtensionPaths:
    def __init__(
        self,
        skill_paths: list[dict[str, Any]] | None = None,
        prompt_paths: list[dict[str, Any]] | None = None,
        theme_paths: list[dict[str, Any]] | None = None,
    ) -> None:
        self.skill_paths = skill_paths or []
        self.prompt_paths = prompt_paths or []
        self.theme_paths = theme_paths or []


class ResourceLoaderReloadOptions:
    def __init__(
        self,
        resolve_project_trust: Callable[..., Any] | None = None,
    ) -> None:
        self.resolve_project_trust = resolve_project_trust


class ResourceLoader:
    def get_extensions(self) -> LoadExtensionsResult:
        raise NotImplementedError

    def get_skills(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_prompts(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_themes(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_agents_files(self) -> dict[str, Any]:
        raise NotImplementedError

    def get_system_prompt(self) -> str | None: ...

    def get_system_prompt_source(self) -> dict[str, Any] | None: ...

    def get_append_system_prompt(self) -> list[str]:
        raise NotImplementedError

    def get_append_system_prompt_sources(self) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def extend_resources(self, paths: ResourceExtensionPaths) -> None: ...

    async def reload(
        self, options: ResourceLoaderReloadOptions | None = None
    ) -> None: ...


def _resolve_prompt_input(input_str: str | None, description: str) -> str | None:
    if not input_str:
        return None
    if Path(input_str).exists():
        try:
            return Path(input_str).read_text("utf-8")
        except OSError as error:
            import warnings

            warnings.warn(
                f"Warning: Could not read {description} file {input_str}: {error}"
            )
            return input_str
    return input_str


def _load_context_file_from_dir(dir_path: str) -> dict[str, Any] | None:
    candidates = ["AGENTS.md", "AGENTS.MD", "CLAUDE.md", "CLAUDE.MD"]
    d = Path(dir_path)
    for filename in candidates:
        file_path = d / filename
        if file_path.exists():
            try:
                if not file_path.is_file():
                    continue
                return {
                    "path": str(file_path),
                    "content": file_path.read_text("utf-8"),
                }
            except OSError as error:
                import warnings

                warnings.warn(f"Warning: Could not read {file_path}: {error}")
    return None


def _find_shadowed_context_file(cwd: str) -> str | None:
    git_paths = find_git_paths(cwd)
    if not git_paths:
        return None
    common_git_dir = canonicalize_path(git_paths.common_git_dir)
    worktree_root = canonicalize_path(git_paths.repo_dir)
    main_repo_root = os.path.dirname(common_git_dir)
    if not worktree_root.startswith(f"{main_repo_root}{os.sep}"):
        return None
    if canonicalize_path(str(Path(main_repo_root) / ".git")) != common_git_dir:
        return None
    worktree_context = _load_context_file_from_dir(worktree_root)
    if worktree_context:
        return str(Path(main_repo_root) / os.path.basename(worktree_context["path"]))
    return None


def load_project_context_files(options: dict[str, Any]) -> list[dict[str, Any]]:
    resolved_cwd = resolve_path(options["cwd"])
    resolved_agent_dir = resolve_path(options["agent_dir"])

    context_files: list[dict[str, Any]] = []
    seen_paths: set[str] = set()

    global_context = _load_context_file_from_dir(resolved_agent_dir)
    if global_context:
        context_files.append(global_context)
        seen_paths.add(global_context["path"])

    ancestor_context_files: list[dict[str, Any]] = []
    shadowed = _find_shadowed_context_file(resolved_cwd)
    current_dir = resolved_cwd

    while True:
        context_file = _load_context_file_from_dir(current_dir)
        is_shadowed = (
            shadowed is not None
            and canonicalize_path(context_file["path"] if context_file else "")
            == shadowed
        )
        if context_file and not is_shadowed and context_file["path"] not in seen_paths:
            ancestor_context_files.insert(0, context_file)
            seen_paths.add(context_file["path"])
        parent = os.path.dirname(current_dir)
        if parent == current_dir:
            break
        current_dir = parent

    context_files.extend(ancestor_context_files)
    return context_files


class DefaultResourceLoaderOptions:
    def __init__(
        self,
        cwd: str,
        agent_dir: str,
        settings_manager: SettingsManager | None = None,
        event_bus: EventBus | None = None,
        additional_extension_paths: list[str] | None = None,
        additional_skill_paths: list[str] | None = None,
        additional_prompt_template_paths: list[str] | None = None,
        additional_theme_paths: list[str] | None = None,
        extension_factories: list[InlineExtension] | None = None,
        no_extensions: bool = False,
        no_skills: bool = False,
        no_prompt_templates: bool = False,
        no_themes: bool = False,
        no_context_files: bool = False,
        system_prompt: str | None = None,
        append_system_prompt: list[str] | None = None,
        extensions_override: Callable[..., Any] | None = None,
        skills_override: Callable[..., Any] | None = None,
        prompts_override: Callable[..., Any] | None = None,
        themes_override: Callable[..., Any] | None = None,
        agents_files_override: Callable[..., Any] | None = None,
        system_prompt_override: Callable[..., Any] | None = None,
        append_system_prompt_override: Callable[..., Any] | None = None,
    ) -> None:
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.settings_manager = settings_manager
        self.event_bus = event_bus
        self.additional_extension_paths = additional_extension_paths or []
        self.additional_skill_paths = additional_skill_paths or []
        self.additional_prompt_template_paths = additional_prompt_template_paths or []
        self.additional_theme_paths = additional_theme_paths or []
        self.extension_factories = extension_factories or []
        self.no_extensions = no_extensions
        self.no_skills = no_skills
        self.no_prompt_templates = no_prompt_templates
        self.no_themes = no_themes
        self.no_context_files = no_context_files
        self.system_prompt = system_prompt
        self.append_system_prompt = append_system_prompt
        self.extensions_override = extensions_override
        self.skills_override = skills_override
        self.prompts_override = prompts_override
        self.themes_override = themes_override
        self.agents_files_override = agents_files_override
        self.system_prompt_override = system_prompt_override
        self.append_system_prompt_override = append_system_prompt_override


class DefaultResourceLoader(ResourceLoader):
    def __init__(self, options: DefaultResourceLoaderOptions) -> None:
        self._cwd = resolve_path(options.cwd)
        self._agent_dir = resolve_path(options.agent_dir)
        self._settings_manager = options.settings_manager or SettingsManager.create(
            self._cwd, self._agent_dir
        )
        self._event_bus = options.event_bus or EventBus()
        self._package_manager = DefaultPackageManager(
            {
                "cwd": self._cwd,
                "agent_dir": self._agent_dir,
                "settings_manager": self._settings_manager,
            }
        )
        self._additional_extension_paths = options.additional_extension_paths
        self._additional_skill_paths = options.additional_skill_paths
        self._additional_prompt_template_paths = (
            options.additional_prompt_template_paths
        )
        self._additional_theme_paths = options.additional_theme_paths
        self._extension_factories = options.extension_factories
        self._no_extensions = options.no_extensions
        self._no_skills = options.no_skills
        self._no_prompt_templates = options.no_prompt_templates
        self._no_themes = options.no_themes
        self._no_context_files = options.no_context_files
        self._system_prompt_source = options.system_prompt
        self._append_system_prompt_source = options.append_system_prompt
        self._extensions_override = options.extensions_override
        self._skills_override = options.skills_override
        self._prompts_override = options.prompts_override
        self._themes_override = options.themes_override
        self._agents_files_override = options.agents_files_override
        self._system_prompt_override = options.system_prompt_override
        self._append_system_prompt_override = options.append_system_prompt_override

        self._extensions_result: LoadExtensionsResult = LoadExtensionsResult(
            extensions=[],
            errors=[],
            runtime=create_extension_runtime(),
        )
        self._skills: list[Skill] = []
        self._skill_diagnostics: list[dict[str, Any]] = []
        self._prompts: list[PromptTemplate] = []
        self._prompt_diagnostics: list[dict[str, Any]] = []
        self._themes: list[dict[str, Any]] = []
        self._theme_diagnostics: list[dict[str, Any]] = []
        self._agents_files: list[dict[str, Any]] = []
        self._system_prompt: str | None = None
        self._system_prompt_source_path: str | None = None
        self._append_system_prompt: list[str] = []
        self._append_system_prompt_source_paths: list[str] = []
        self._last_skill_paths: list[str] = []
        self._extension_skill_source_infos: dict[str, SourceInfo] = {}
        self._extension_prompt_source_infos: dict[str, SourceInfo] = {}
        self._extension_theme_source_infos: dict[str, SourceInfo] = {}
        self._resource_metadata_by_path: dict[str, PathMetadata] = {}
        self._last_prompt_paths: list[str] = []
        self._last_theme_paths: list[str] = []
        self._loaded = False

    def get_extensions(self) -> LoadExtensionsResult:
        return self._extensions_result

    def get_skills(self) -> dict[str, Any]:
        return {"skills": self._skills, "diagnostics": self._skill_diagnostics}

    def get_prompts(self) -> dict[str, Any]:
        return {"prompts": self._prompts, "diagnostics": self._prompt_diagnostics}

    def get_themes(self) -> dict[str, Any]:
        return {"themes": self._themes, "diagnostics": self._theme_diagnostics}

    def get_agents_files(self) -> dict[str, Any]:
        return {"agents_files": self._agents_files}

    def get_system_prompt(self) -> str | None:
        return self._system_prompt

    def get_system_prompt_source(self) -> dict[str, Any] | None:
        return (
            {"path": self._system_prompt_source_path}
            if self._system_prompt_source_path
            else None
        )

    def get_append_system_prompt(self) -> list[str]:
        return self._append_system_prompt

    def get_append_system_prompt_sources(self) -> list[dict[str, Any]]:
        return [{"path": p} for p in self._append_system_prompt_source_paths]

    async def extend_resources(self, paths: ResourceExtensionPaths) -> None:
        skill_paths = self._normalize_extension_paths(paths.skill_paths or [])
        prompt_paths = self._normalize_extension_paths(paths.prompt_paths or [])
        theme_paths = self._normalize_extension_paths(paths.theme_paths or [])

        for entry in skill_paths:
            self._extension_skill_source_infos[entry["path"]] = create_source_info(
                entry["path"], entry["metadata"]
            )
        for entry in prompt_paths:
            self._extension_prompt_source_infos[entry["path"]] = create_source_info(
                entry["path"], entry["metadata"]
            )
        for entry in theme_paths:
            self._extension_theme_source_infos[entry["path"]] = create_source_info(
                entry["path"], entry["metadata"]
            )

        if skill_paths:
            self._last_skill_paths = self._merge_paths(
                self._last_skill_paths, [e["path"] for e in skill_paths]
            )
            await self._update_skills_from_paths(
                self._last_skill_paths, self._resource_metadata_by_path
            )

        if prompt_paths:
            self._last_prompt_paths = self._merge_paths(
                self._last_prompt_paths, [e["path"] for e in prompt_paths]
            )
            await self._update_prompts_from_paths(
                self._last_prompt_paths, self._resource_metadata_by_path
            )

        if theme_paths:
            self._last_theme_paths = self._merge_paths(
                self._last_theme_paths, [e["path"] for e in theme_paths]
            )
            self._update_themes_from_paths(
                self._last_theme_paths, self._resource_metadata_by_path
            )

    async def load_project_trust_extensions(self) -> LoadExtensionsResult:
        self._settings_manager.set_project_trusted(False)
        self._settings_manager.reload()
        return await self._load_current_extension_set(
            {"include_inline_factories": True}
        )

    async def reload(self, options: ResourceLoaderReloadOptions | None = None) -> None:
        reset_timings("extensions")

        if self._loaded:
            clear_extension_cache()

        pre_trust_extensions: LoadExtensionsResult | None = None
        if options and options.resolve_project_trust:
            pre_trust_extensions = await self.load_project_trust_extensions()
            project_trusted = await options.resolve_project_trust(
                {"extensions_result": pre_trust_extensions}
            )
            self._settings_manager.set_project_trusted(project_trusted)

        self._settings_manager.reload()

        resolved_paths = await self._package_manager.resolve()
        cli_extension_paths = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, {"temporary": True}
        )
        self._resource_metadata_by_path = {}
        metadata_by_path = self._resource_metadata_by_path

        self._extension_skill_source_infos = {}
        self._extension_prompt_source_infos = {}
        self._extension_theme_source_infos = {}

        def get_enabled_resources(
            resources: list[ResolvedResource],
        ) -> list[ResolvedResource]:
            for r in resources:
                if r.path not in metadata_by_path and r.metadata is not None:
                    metadata_by_path[r.path] = r.metadata
            return [r for r in resources if r.enabled]

        enabled_extensions = [
            r.path for r in get_enabled_resources(resolved_paths.extensions)
        ]
        enabled_skill_resources = get_enabled_resources(resolved_paths.skills)
        enabled_prompts = [
            r.path for r in get_enabled_resources(resolved_paths.prompts)
        ]
        enabled_themes = [r.path for r in get_enabled_resources(resolved_paths.themes)]

        enabled_skills = [
            self._map_skill_path(r, metadata_by_path) for r in enabled_skill_resources
        ]

        for r in cli_extension_paths.extensions:
            if r.path not in metadata_by_path:
                metadata_by_path[r.path] = PathMetadata(
                    source="cli", scope="temporary", origin="top-level"
                )
        for r in cli_extension_paths.skills:
            if r.path not in metadata_by_path:
                metadata_by_path[r.path] = PathMetadata(
                    source="cli", scope="temporary", origin="top-level"
                )

        cli_enabled_extensions = [
            r.path for r in get_enabled_resources(cli_extension_paths.extensions)
        ]
        cli_enabled_skills = [
            r.path for r in get_enabled_resources(cli_extension_paths.skills)
        ]
        cli_enabled_prompts = [
            r.path for r in get_enabled_resources(cli_extension_paths.prompts)
        ]
        cli_enabled_themes = [
            r.path for r in get_enabled_resources(cli_extension_paths.themes)
        ]

        extension_paths = (
            cli_enabled_extensions
            if self._no_extensions
            else self._merge_paths(cli_enabled_extensions, enabled_extensions)
        )

        extensions_result = await self._load_final_extension_set(
            extension_paths, pre_trust_extensions
        )
        for p in self._additional_extension_paths:
            if is_local_path(p):
                resolved = self._resolve_resource_path(p)
                if not Path(resolved).exists():
                    extensions_result.errors.append(
                        {
                            "path": resolved,
                            "error": f"Extension path does not exist: {resolved}",
                        }
                    )

        self._extensions_result = (
            self._extensions_override(extensions_result)
            if self._extensions_override
            else extensions_result
        )
        self._apply_extension_source_info(
            self._extensions_result.extensions, metadata_by_path
        )

        skill_paths = (
            self._merge_paths(cli_enabled_skills, self._additional_skill_paths)
            if self._no_skills
            else self._merge_paths(
                [*cli_enabled_skills, *enabled_skills], self._additional_skill_paths
            )
        )
        self._last_skill_paths = skill_paths
        await self._update_skills_from_paths(skill_paths, metadata_by_path)

        prompt_paths = (
            self._merge_paths(
                cli_enabled_prompts, self._additional_prompt_template_paths
            )
            if self._no_prompt_templates
            else self._merge_paths(
                [*cli_enabled_prompts, *enabled_prompts],
                self._additional_prompt_template_paths,
            )
        )
        self._last_prompt_paths = prompt_paths
        await self._update_prompts_from_paths(prompt_paths, metadata_by_path)

        theme_paths = (
            self._merge_paths(cli_enabled_themes, self._additional_theme_paths)
            if self._no_themes
            else self._merge_paths(
                [*cli_enabled_themes, *enabled_themes], self._additional_theme_paths
            )
        )
        self._last_theme_paths = theme_paths
        self._update_themes_from_paths(theme_paths, metadata_by_path)

        agents_files = {
            "agents_files": []
            if self._no_context_files
            else load_project_context_files(
                {
                    "cwd": self._cwd,
                    "agent_dir": self._agent_dir,
                }
            )
        }
        resolved_agents = (
            self._agents_files_override(agents_files)
            if self._agents_files_override
            else agents_files
        )
        self._agents_files = resolved_agents["agents_files"]

        system_prompt_source = (
            self._system_prompt_source or self._discover_system_prompt_file()
        )
        base_system_prompt = _resolve_prompt_input(
            system_prompt_source, "system prompt"
        )
        self._system_prompt = (
            self._system_prompt_override(base_system_prompt)
            if self._system_prompt_override
            else base_system_prompt
        )
        self._system_prompt_source_path = (
            resolve_path(system_prompt_source)
            if system_prompt_source and Path(system_prompt_source).exists()
            else None
        )

        append_sources = self._append_system_prompt_source
        if not append_sources:
            discovered = self._discover_append_system_prompt_file()
            append_sources = [discovered] if discovered else []

        base_append = [
            s
            for s in [
                _resolve_prompt_input(src, "append system prompt")
                for src in append_sources
            ]
            if s is not None
        ]
        self._append_system_prompt = (
            self._append_system_prompt_override(base_append)
            if self._append_system_prompt_override
            else base_append
        )
        self._append_system_prompt_source_paths = [
            resolve_path(src) for src in append_sources if Path(src).exists()
        ]
        self._loaded = True

    async def _load_current_extension_set(
        self, options: dict[str, Any]
    ) -> LoadExtensionsResult:
        resolved_paths = await self._package_manager.resolve()
        cli_extensions = await self._package_manager.resolve_extension_sources(
            self._additional_extension_paths, {"temporary": True}
        )
        enabled_extensions = [r.path for r in resolved_paths.extensions if r.enabled]
        cli_enabled = [r.path for r in cli_extensions.extensions if r.enabled]
        extension_paths = (
            cli_enabled
            if self._no_extensions
            else self._merge_paths(cli_enabled, enabled_extensions)
        )
        result_dict = await load_extensions_cached(
            extension_paths, self._cwd, self._event_bus
        )
        result = LoadExtensionsResult(**result_dict)
        if not options.get("include_inline_factories"):
            return result
        inline = await self._load_extension_factories(result.runtime)
        result.extensions.extend(inline.extensions)
        result.errors.extend(inline.errors)
        return result

    async def _load_final_extension_set(
        self, extension_paths: list[str], pre_trust: LoadExtensionsResult | None
    ) -> LoadExtensionsResult:
        if not pre_trust:
            result_dict = await load_extensions_cached(
                extension_paths, self._cwd, self._event_bus
            )
            result = LoadExtensionsResult(**result_dict)
            inline = await self._load_extension_factories(result.runtime)
            result.extensions.extend(inline.extensions)
            result.errors.extend(inline.errors)
            self._add_extension_conflict_diagnostics(result)
            return result

        preloaded_by_path = {
            ext.resolved_path: ext
            for ext in pre_trust.extensions
            if not ext.path.startswith("<inline:")
        }
        failed_preload_paths = set(
            self._resolve_extension_load_path(e["path"]) for e in pre_trust.errors
        )
        remaining = [
            p
            for p in extension_paths
            if self._resolve_extension_load_path(p) not in preloaded_by_path
            and self._resolve_extension_load_path(p) not in failed_preload_paths
        ]
        remaining_result_dict = await load_extensions_cached(
            remaining, self._cwd, self._event_bus, pre_trust.runtime
        )
        remaining_result = LoadExtensionsResult(**remaining_result_dict)
        loaded_by_path = dict(preloaded_by_path)
        for ext in remaining_result.extensions:
            loaded_by_path[ext.resolved_path] = ext

        inline_extensions = [
            e for e in pre_trust.extensions if e.path.startswith("<inline:")
        ]
        ordered: list[Extension] = [
            loaded_by_path[self._resolve_extension_load_path(p)]
            for p in extension_paths
            if self._resolve_extension_load_path(p) in loaded_by_path
        ]
        ordered.extend(inline_extensions)

        result = LoadExtensionsResult(
            extensions=ordered,
            errors=pre_trust.errors + remaining_result.errors,
            runtime=pre_trust.runtime,
        )
        self._add_extension_conflict_diagnostics(result)
        return result

    def _add_extension_conflict_diagnostics(self, result: LoadExtensionsResult) -> None:
        conflicts = self._detect_extension_conflicts(result.extensions)
        for conflict in conflicts:
            result.errors.append(
                {"path": conflict["path"], "error": conflict["message"]}
            )

    def _map_skill_path(
        self, resource: ResolvedResource, metadata_by_path: dict[str, Any]
    ) -> str:
        if (
            resource.metadata
            and resource.metadata.source != "auto"
            and resource.metadata.origin != "package"
        ):
            return resource.path
        try:
            if not Path(resource.path).is_dir():
                return resource.path
        except OSError:
            return resource.path
        skill_file = str(Path(resource.path) / "SKILL.md")
        if Path(skill_file).exists():
            if skill_file not in metadata_by_path:
                metadata_by_path[skill_file] = resource.metadata
            return skill_file
        return resource.path

    def _normalize_extension_paths(
        self, entries: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        return [
            {
                "path": self._resolve_resource_path(e["path"]),
                "metadata": PathMetadata(
                    **{
                        **e["metadata"].__dict__,
                        "base_dir": self._resolve_resource_path(e["metadata"].base_dir),
                    }
                )
                if e["metadata"].base_dir
                else e["metadata"],
            }
            for e in entries
        ]

    async def _update_skills_from_paths(
        self, skill_paths: list[str], metadata_by_path: dict[str, Any] | None = None
    ) -> None:
        if self._no_skills and not skill_paths:
            skills_result = SkillsResult(skills=[], diagnostics=[])
        else:
            skills_result = await load_skills(
                LoadSkillsOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    skill_paths=skill_paths,
                    include_defaults=False,
                )
            )
        resolved = (
            self._skills_override(skills_result)
            if self._skills_override
            else skills_result
        )
        self._skills = [
            Skill(
                name=skill.name,
                description=skill.description,
                file_path=skill.file_path,
                base_dir=skill.base_dir,
                source_info=cast(
                    SkillSourceInfo,
                    self._find_source_info_for_path(
                        skill.file_path,
                        self._extension_skill_source_infos,
                        metadata_by_path,
                    )
                    or skill.source_info
                    or self._get_default_source_info_for_path(skill.file_path),
                ),
                disable_model_invocation=skill.disable_model_invocation,
            )
            for skill in resolved.skills
        ]
        self._skill_diagnostics = cast("list[dict[str, Any]]", resolved.diagnostics)

    async def _update_prompts_from_paths(
        self, prompt_paths: list[str], metadata_by_path: dict[str, Any] | None = None
    ) -> None:
        if self._no_prompt_templates and not prompt_paths:
            prompts_result: dict[str, Any] = {"prompts": [], "diagnostics": []}
        else:
            all_prompts = await load_prompt_templates(
                LoadPromptTemplatesOptions(
                    cwd=self._cwd,
                    agent_dir=self._agent_dir,
                    prompt_paths=prompt_paths,
                    include_defaults=False,
                )
            )
            prompts_result = self._dedupe_prompts(all_prompts)
        resolved = (
            self._prompts_override(prompts_result)
            if self._prompts_override
            else prompts_result
        )
        self._prompts = [
            PromptTemplate(
                name=prompt.name,
                description=prompt.description,
                argument_hint=prompt.argument_hint,
                content=prompt.content,
                file_path=prompt.file_path,
                source_info=(
                    self._find_source_info_for_path(
                        prompt.file_path,
                        self._extension_prompt_source_infos,
                        metadata_by_path,
                    )
                    or prompt.source_info
                    or self._get_default_source_info_for_path(prompt.file_path)
                ),
            )
            for prompt in resolved["prompts"]
        ]
        self._prompt_diagnostics = resolved["diagnostics"]

    def _update_themes_from_paths(
        self, theme_paths: list[str], metadata_by_path: dict[str, Any] | None = None
    ) -> None:
        if self._no_themes and not theme_paths:
            themes_result: dict[str, Any] = {"themes": [], "diagnostics": []}
        else:
            loaded = self._load_themes(theme_paths, False)
            deduped = self._dedupe_themes(loaded["themes"])
            themes_result = {
                "themes": deduped["themes"],
                "diagnostics": loaded["diagnostics"] + deduped["diagnostics"],
            }
        resolved = (
            self._themes_override(themes_result)
            if self._themes_override
            else themes_result
        )
        self._themes = [
            {
                **theme,
                "source_info": (
                    self._find_source_info_for_path(
                        theme.get("source_path", ""),
                        self._extension_theme_source_infos,
                        metadata_by_path,
                    )
                    or theme.get("source_info")
                    or self._get_default_source_info_for_path(
                        theme.get("source_path", "")
                    )
                )
                if theme.get("source_path")
                else theme.get("source_info"),
            }
            for theme in resolved["themes"]
        ]
        self._theme_diagnostics = resolved["diagnostics"]

    def _apply_extension_source_info(
        self, extensions: list[Extension], metadata_by_path: dict[str, Any]
    ) -> None:
        for ext in extensions:
            ext.source_info = self._find_source_info_for_path(
                ext.path, None, metadata_by_path
            ) or self._get_default_source_info_for_path(ext.path)
            for cmd in ext.commands.values():
                cmd.source_info = ext.source_info
            for tool in ext.tools.values():
                tool.source_info = ext.source_info

    def _find_source_info_for_path(
        self,
        resource_path: str,
        extra_source_infos: dict[str, SourceInfo] | None = None,
        metadata_by_path: dict[str, PathMetadata] | None = None,
    ) -> SourceInfo | None:
        if not resource_path:
            return None
        if resource_path.startswith("<"):
            return self._get_default_source_info_for_path(resource_path)

        normalized = os.path.abspath(resource_path)
        if extra_source_infos:
            for sp, si in extra_source_infos.items():
                normalized_sp = os.path.abspath(sp)
                if normalized == normalized_sp or normalized.startswith(
                    f"{normalized_sp}{os.sep}"
                ):
                    return SourceInfo(
                        path=resource_path,
                        source=si.source,
                        scope=si.scope,
                        origin=si.origin,
                        base_dir=si.base_dir,
                    )

        if metadata_by_path:
            exact = metadata_by_path.get(normalized) or metadata_by_path.get(
                resource_path
            )
            if exact:
                return create_source_info(resource_path, exact)
            for sp, md in metadata_by_path.items():
                normalized_sp = os.path.abspath(sp)
                if normalized == normalized_sp or normalized.startswith(
                    f"{normalized_sp}{os.sep}"
                ):
                    return create_source_info(resource_path, md)
        return None

    def _get_default_source_info_for_path(self, file_path: str) -> SourceInfo:
        if file_path.startswith("<") and file_path.endswith(">"):
            return SourceInfo(
                path=file_path,
                source=file_path[1:-1].split(":")[0] or "temporary",
                scope="temporary",
                origin="top-level",
            )
        normalized = os.path.abspath(file_path)
        agent_roots = [
            str(Path(self._agent_dir) / "skills"),
            str(Path(self._agent_dir) / "prompts"),
            str(Path(self._agent_dir) / "themes"),
            str(Path(self._agent_dir) / "extensions"),
        ]
        project_roots = [
            str(Path(self._cwd) / CONFIG_DIR_NAME / "skills"),
            str(Path(self._cwd) / CONFIG_DIR_NAME / "prompts"),
            str(Path(self._cwd) / CONFIG_DIR_NAME / "themes"),
            str(Path(self._cwd) / CONFIG_DIR_NAME / "extensions"),
        ]

        for root in agent_roots:
            if self._is_under_path(normalized, root):
                return SourceInfo(
                    path=file_path,
                    source="local",
                    scope="user",
                    origin="top-level",
                    base_dir=root,
                )

        for root in project_roots:
            if self._is_under_path(normalized, root):
                return SourceInfo(
                    path=file_path,
                    source="local",
                    scope="project",
                    origin="top-level",
                    base_dir=root,
                )

        return SourceInfo(
            path=file_path,
            source="local",
            scope="temporary",
            origin="top-level",
            base_dir=normalized
            if Path(normalized).is_dir()
            else os.path.dirname(normalized),
        )

    def _merge_paths(self, primary: list[str], additional: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for p in list(primary) + list(additional):
            resolved = self._resolve_resource_path(p)
            canonical = canonicalize_path(resolved)
            if canonical in seen:
                continue
            seen.add(canonical)
            merged.append(resolved)
        return merged

    def _resolve_resource_path(self, p: str) -> str:
        return resolve_path(p, self._cwd, {"trim": True})

    def _resolve_extension_load_path(self, path: str) -> str:
        return resolve_path(path, self._cwd, {"normalize_unicode_spaces": True})

    def _load_themes(
        self, paths: list[str], include_defaults: bool = True
    ) -> dict[str, Any]:
        themes: list[dict[str, Any]] = []
        diagnostics: list[dict[str, Any]] = []
        if include_defaults:
            default_dirs = [
                str(Path(self._agent_dir) / "themes"),
                str(Path(self._cwd) / CONFIG_DIR_NAME / "themes"),
            ]
            for d in default_dirs:
                self._load_themes_from_dir(d, themes, diagnostics)

        for p in paths:
            resolved = self._resolve_resource_path(p)
            if not Path(resolved).exists():
                diagnostics.append(
                    {
                        "type": "warning",
                        "message": "theme path does not exist",
                        "path": resolved,
                    }
                )
                continue
            try:
                if Path(resolved).is_dir():
                    self._load_themes_from_dir(resolved, themes, diagnostics)
                elif Path(resolved).is_file() and resolved.endswith(".json"):
                    self._load_theme_from_file(resolved, themes, diagnostics)
                else:
                    diagnostics.append(
                        {
                            "type": "warning",
                            "message": "theme path is not a json file",
                            "path": resolved,
                        }
                    )
            except OSError as error:
                diagnostics.append(
                    {"type": "warning", "message": str(error), "path": resolved}
                )

        return {"themes": themes, "diagnostics": diagnostics}

    def _load_themes_from_dir(
        self,
        dir_path: str,
        themes: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
    ) -> None:
        d = Path(dir_path)
        if not d.exists():
            return
        try:
            for entry in d.iterdir():
                is_file = entry.is_file()
                if entry.is_symlink():
                    try:
                        # stat() follow symlinks，使用 st_mode 判断是否为常规文件
                        import stat as stat_mod

                        is_file = stat_mod.S_ISREG(entry.stat().st_mode)
                    except OSError:
                        continue
                if not is_file or not entry.name.endswith(".json"):
                    continue
                self._load_theme_from_file(str(entry), themes, diagnostics)
        except OSError as error:
            diagnostics.append(
                {"type": "warning", "message": str(error), "path": dir_path}
            )

    def _load_theme_from_file(
        self,
        file_path: str,
        themes: list[dict[str, Any]],
        diagnostics: list[dict[str, Any]],
    ) -> None:
        try:
            from .theme import load_theme_from_path  # type: ignore[import-not-found]

            themes.append(load_theme_from_path(file_path))
        except Exception as error:
            diagnostics.append(
                {"type": "warning", "message": str(error), "path": file_path}
            )

    async def _load_extension_factories(
        self, runtime: ExtensionRuntime
    ) -> LoadExtensionsResult:
        extensions: list[Extension] = []
        errors: list[dict[str, str]] = []
        for index, input_ in enumerate(self._extension_factories):
            is_named = not callable(input_)
            assert isinstance(input_, InlineExtension)
            factory = input_.factory if is_named else input_
            ext_path = f"<inline:{input_.name if is_named else index + 1}>"
            try:
                extension = await load_extension_from_factory(
                    factory, self._cwd, self._event_bus, runtime, ext_path
                )
                if is_named and input_.hidden:
                    extension.hidden = True
                extensions.append(extension)
            except Exception as error:
                errors.append({"path": ext_path, "error": str(error)})
        return LoadExtensionsResult(
            extensions=extensions, errors=errors, runtime=runtime
        )

    def _dedupe_prompts(self, prompts: list[PromptTemplate]) -> dict[str, Any]:
        seen: dict[str, PromptTemplate] = {}
        diagnostics: list[dict[str, Any]] = []
        for prompt in prompts:
            existing = seen.get(prompt.name)
            if existing:
                diagnostics.append(
                    {
                        "type": "collision",
                        "message": f'name "/{prompt.name}" collision',
                        "path": prompt.file_path,
                        "collision": {
                            "resource_type": "prompt",
                            "name": prompt.name,
                            "winner_path": existing.file_path,
                            "loser_path": prompt.file_path,
                        },
                    }
                )
            else:
                seen[prompt.name] = prompt
        return {"prompts": list(seen.values()), "diagnostics": diagnostics}

    def _dedupe_themes(self, themes: list[dict[str, Any]]) -> dict[str, Any]:
        seen: dict[str, dict[str, Any]] = {}
        diagnostics: list[dict[str, Any]] = []
        for t in themes:
            name = t.get("name") or "unnamed"
            existing = seen.get(name)
            if existing:
                diagnostics.append(
                    {
                        "type": "collision",
                        "message": f'name "{name}" collision',
                        "path": t.get("source_path", ""),
                        "collision": {
                            "resource_type": "theme",
                            "name": name,
                            "winner_path": existing.get("source_path") or "<builtin>",
                            "loser_path": t.get("source_path") or "<builtin>",
                        },
                    }
                )
            else:
                seen[name] = t
        return {"themes": list(seen.values()), "diagnostics": diagnostics}

    def _discover_system_prompt_file(self) -> str | None:
        project_path = str(Path(self._cwd) / CONFIG_DIR_NAME / "SYSTEM.md")
        if self._settings_manager.is_project_trusted() and Path(project_path).exists():
            return project_path
        global_path = str(Path(self._agent_dir) / "SYSTEM.md")
        if Path(global_path).exists():
            return global_path
        return None

    def _discover_append_system_prompt_file(self) -> str | None:
        project_path = str(Path(self._cwd) / CONFIG_DIR_NAME / "APPEND_SYSTEM.md")
        if self._settings_manager.is_project_trusted() and Path(project_path).exists():
            return project_path
        global_path = str(Path(self._agent_dir) / "APPEND_SYSTEM.md")
        if Path(global_path).exists():
            return global_path
        return None

    def _is_under_path(self, target: str, root: str) -> bool:
        normalized_root = os.path.abspath(root)
        if target == normalized_root:
            return True
        prefix = (
            normalized_root
            if normalized_root.endswith(os.sep)
            else f"{normalized_root}{os.sep}"
        )
        return target.startswith(prefix)

    def _detect_extension_conflicts(
        self, extensions: list[Extension]
    ) -> list[dict[str, Any]]:
        conflicts: list[dict[str, Any]] = []
        tool_owners: dict[str, str] = {}
        flag_owners: dict[str, str] = {}

        for ext in extensions:
            for tool_name in ext.tools.keys():
                existing = tool_owners.get(tool_name)
                if existing and existing != ext.path:
                    conflicts.append(
                        {
                            "path": ext.path,
                            "message": f'Tool "{tool_name}" conflicts with {existing}',
                        }
                    )
                else:
                    tool_owners[tool_name] = ext.path

            for flag_name in ext.flags.keys():
                existing = flag_owners.get(flag_name)
                if existing and existing != ext.path:
                    conflicts.append(
                        {
                            "path": ext.path,
                            "message": f'Flag "--{flag_name}" conflicts with {existing}',
                        }
                    )
                else:
                    flag_owners[flag_name] = ext.path

        return conflicts
