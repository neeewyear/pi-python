"""Settings models and manager (对应 TS ``core/settings-manager.ts`` 的简化版)。

提供 Pydantic 模型描述所有设置类型，以及 ``SettingsManager`` 类用于
读写 ``settings.json``。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, TypeAlias

import orjson
from pi_agent.types import ThinkingLevel
from pi_ai.types import Transport
from pydantic import BaseModel, ConfigDict

from ..config import get_settings_path

# ---------------------------------------------------------------------------
# 设置子类型
# ---------------------------------------------------------------------------

MermaidRenderingMode: TypeAlias = Literal["off", "final", "streaming"]
"""Mermaid 渲染模式。"""

UiMode: TypeAlias = Literal["regular", "fullscreen"]
"""UI 模式。"""

DefaultProjectTrust: TypeAlias = Literal["ask", "always", "never"]
"""默认项目信任。"""

TransportSetting: TypeAlias = Transport
"""传输层设置。"""


class CompactionSettings(BaseModel):
    """上下文压缩设置。"""

    enabled: bool | None = None
    reserve_tokens: int | None = None
    keep_recent_tokens: int | None = None


class BranchSummarySettings(BaseModel):
    """分支摘要设置。"""

    reserve_tokens: int | None = None
    skip_prompt: bool | None = None


class ProviderRetrySettings(BaseModel):
    """Provider 重试设置。"""

    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None


class RetrySettings(BaseModel):
    """重试设置。"""

    enabled: bool | None = None
    max_retries: int | None = None
    base_delay_ms: int | None = None
    provider: ProviderRetrySettings | None = None


class TerminalSettings(BaseModel):
    """终端设置。"""

    show_images: bool | None = None
    image_width_cells: int | None = None
    clear_on_shrink: bool | None = None
    show_terminal_progress: bool | None = None


class ImageSettings(BaseModel):
    """图片设置。"""

    auto_resize: bool | None = None
    block_images: bool | None = None


class ThinkingBudgetsSettings(BaseModel):
    """思考级别 token 预算。"""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


class MarkdownSettings(BaseModel):
    """Markdown 设置。"""

    code_block_indent: str | None = None
    mermaid: MermaidRenderingMode | None = None


class WarningSettings(BaseModel):
    """警告设置。"""

    anthropic_extra_usage: bool | None = None


class PackageSource(BaseModel):
    """包源（npm/git 包）。"""

    source: str
    autoload: bool | None = None
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None


# ---------------------------------------------------------------------------
# 顶层设置模型
# ---------------------------------------------------------------------------


class Settings(BaseModel):
    """顶层设置（对应 TS ``Settings``）。"""

    model_config = ConfigDict(extra="allow")

    last_changelog_version: str | None = None
    default_provider: str | None = None
    default_model: str | None = None
    default_thinking_level: ThinkingLevel | None = None
    transport: TransportSetting | None = None
    steering_mode: Literal["all", "one-at-a-time"] | None = None
    follow_up_mode: Literal["all", "one-at-a-time"] | None = None
    theme: str | None = None
    compaction: CompactionSettings | None = None
    branch_summary: BranchSummarySettings | None = None

    def get(self, key: str, default: Any = None) -> Any:
        """兼容 dict 的 get 方法。

        Args:
            key: 字段名。
            default: 默认值。

        Returns:
            字段值，如果字段不存在或值为 None 则返回 default。
        """
        # 尝试作为已定义字段访问
        value = getattr(self, key, None)
        if value is not None:
            return value
        # 尝试从额外字段获取（extra="allow" 时存储在 model_extra）
        extra = self.model_extra or {}
        if key in extra:
            return extra[key]
        return default

    retry: RetrySettings | None = None
    hide_thinking_block: bool | None = None
    show_cache_miss_notices: bool | None = None
    external_editor: str | None = None
    shell_path: str | None = None
    quiet_startup: bool | None = None
    default_project_trust: DefaultProjectTrust | None = None
    shell_command_prefix: str | None = None
    npm_command: list[str] | None = None
    collapse_changelog: bool | None = None
    enable_install_telemetry: bool | None = None
    enable_analytics: bool | None = None
    tracking_id: str | None = None
    packages: list[PackageSource] | None = None
    extensions: list[str] | None = None
    skills: list[str] | None = None
    prompts: list[str] | None = None
    themes: list[str] | None = None
    enable_skill_commands: bool | None = None
    terminal: TerminalSettings | None = None
    images: ImageSettings | None = None
    enabled_models: list[str] | None = None
    double_escape_action: Literal["fork", "tree", "none"] | None = None
    tree_filter_mode: (
        Literal["default", "no-tools", "user-only", "labeled-only", "all"] | None
    ) = None
    thinking_budgets: ThinkingBudgetsSettings | None = None
    editor_padding_x: int | None = None
    output_pad: Literal[0, 1] | None = None
    autocomplete_max_visible: int | None = None
    show_hardware_cursor: bool | None = None
    markdown: MarkdownSettings | None = None
    warnings: WarningSettings | None = None
    session_dir: str | None = None
    http_proxy: str | None = None
    http_idle_timeout_ms: int | None = None
    websocket_connect_timeout_ms: int | None = None
    ui_mode: UiMode | None = None
    fullscreen_scrollbar: Literal["auto", "always", "hidden"] | None = None


# ---------------------------------------------------------------------------
# SettingsManager
# ---------------------------------------------------------------------------


class SettingsManager:
    """设置管理器（对应 TS ``SettingsManager`` 的简化版）。

    读取/写入 ``settings.json``，提供 getter/setter 方法。
    支持 global 和 project 两级设置，project 设置覆盖 global。
    """

    def __init__(
        self,
        global_path: Path | None = None,
        project_path: Path | None = None,
    ) -> None:
        self._global_path = global_path or get_settings_path()
        self._project_path = project_path
        self._global_settings = Settings()
        self._project_settings = Settings()
        self._merged = Settings()
        self._project_trusted: bool = False
        self._load()

    # ------------------------------------------------------------------
    # 工厂方法
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        cwd: str | None = None,
        agent_dir: str | None = None,
    ) -> SettingsManager:
        """从文件系统创建设置管理器。"""
        if agent_dir:
            global_path = Path(agent_dir) / "settings.json"
        else:
            global_path = get_settings_path()

        project_path: Path | None = None
        if cwd:
            project_candidate = Path(cwd) / ".pi" / "settings.json"
            if project_candidate.exists():
                project_path = project_candidate

        return cls(global_path=global_path, project_path=project_path)

    @classmethod
    def in_memory(cls, settings: Settings | None = None) -> SettingsManager:
        """创建纯内存设置管理器（不读写文件）。"""
        mgr = cls.__new__(cls)
        mgr._global_path = Path("/dev/null")
        mgr._project_path = None
        mgr._global_settings = settings or Settings()
        mgr._project_settings = Settings()
        mgr._merged = cls._deep_merge(mgr._global_settings, mgr._project_settings)
        return mgr

    # ------------------------------------------------------------------
    # 内部
    # ------------------------------------------------------------------

    @staticmethod
    def _deep_merge(base: Settings, overrides: Settings) -> Settings:
        """合并设置：overrides 覆盖 base（嵌套对象递归合并）。"""
        merged = base.model_copy(deep=True)
        for field_name in overrides.model_fields_set:
            override_val = getattr(overrides, field_name)
            if override_val is None:
                continue
            base_val = getattr(merged, field_name)
            if isinstance(base_val, BaseModel) and isinstance(override_val, BaseModel):
                merged_val = base_val.model_copy(deep=True)
                for nested_field in override_val.model_fields_set:
                    nested_val = getattr(override_val, nested_field)
                    if nested_val is not None:
                        setattr(merged_val, nested_field, nested_val)
                setattr(merged, field_name, merged_val)
            else:
                setattr(merged, field_name, override_val)
        return merged

    def _load(self) -> None:
        """从磁盘加载设置。"""
        self._global_settings = self._read_file(self._global_path)
        if self._project_path and self._project_path.exists():
            self._project_settings = self._read_file(self._project_path)
        else:
            self._project_settings = Settings()
        self._merged = self._deep_merge(self._global_settings, self._project_settings)

    @staticmethod
    def _read_file(path: Path) -> Settings:
        """读取并解析 JSON 设置文件。"""
        try:
            raw = path.read_bytes()
            data = orjson.loads(raw)
            if isinstance(data, dict):
                return Settings(**data)
        except (FileNotFoundError, orjson.JSONDecodeError, TypeError):
            pass
        return Settings()

    @staticmethod
    def _write_file(path: Path, settings: Settings) -> None:
        """将设置序列化为 JSON 并写入文件。"""
        raw = settings.model_dump(exclude_none=True, mode="json")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(orjson.dumps(raw, option=orjson.OPT_INDENT_2))

    def _save_global(self) -> None:
        """持久化全局设置。"""
        self._merged = self._deep_merge(self._global_settings, self._project_settings)
        self._write_file(self._global_path, self._global_settings)

    def _save_project(self) -> None:
        """持久化项目设置。"""
        self._merged = self._deep_merge(self._global_settings, self._project_settings)
        if self._project_path:
            self._write_file(self._project_path, self._project_settings)

    # ------------------------------------------------------------------
    # 查询方法
    # ------------------------------------------------------------------

    def get_global_settings(self) -> Settings:
        """获取全局设置（深拷贝）。"""
        return self._global_settings.model_copy(deep=True)

    def get_project_settings(self) -> Settings:
        """获取项目设置（深拷贝）。"""
        return self._project_settings.model_copy(deep=True)

    # ------------------------------------------------------------------
    # 设置 getter/setter
    # ------------------------------------------------------------------

    def get_default_provider(self) -> str | None:
        return self._merged.default_provider

    def set_default_provider(self, provider: str) -> None:
        self._global_settings.default_provider = provider
        self._save_global()

    def get_default_model(self) -> str | None:
        return self._merged.default_model

    def set_default_model(self, model_id: str) -> None:
        self._global_settings.default_model = model_id
        self._save_global()

    def get_default_thinking_level(self) -> ThinkingLevel | None:
        return self._merged.default_thinking_level

    def set_default_thinking_level(self, level: ThinkingLevel) -> None:
        self._global_settings.default_thinking_level = level
        self._save_global()

    def get_transport(self) -> TransportSetting:
        return self._merged.transport or "auto"

    def set_transport(self, transport: TransportSetting) -> None:
        self._global_settings.transport = transport
        self._save_global()

    def get_steering_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._merged.steering_mode or "one-at-a-time"

    def set_steering_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings.steering_mode = mode
        self._save_global()

    def get_follow_up_mode(self) -> Literal["all", "one-at-a-time"]:
        return self._merged.follow_up_mode or "one-at-a-time"

    def set_follow_up_mode(self, mode: Literal["all", "one-at-a-time"]) -> None:
        self._global_settings.follow_up_mode = mode
        self._save_global()

    def get_theme(self) -> str | None:
        theme = self._merged.theme
        if theme and "/" in theme:
            return None
        return theme

    def set_theme(self, theme: str) -> None:
        self._global_settings.theme = theme
        self._save_global()

    def get_compaction_enabled(self) -> bool:
        return (
            self._merged.compaction.enabled
            if self._merged.compaction and self._merged.compaction.enabled is not None
            else True
        )

    def set_compaction_enabled(self, enabled: bool) -> None:
        if not self._global_settings.compaction:
            self._global_settings.compaction = CompactionSettings()
        self._global_settings.compaction.enabled = enabled
        self._save_global()

    def get_compaction_reserve_tokens(self) -> int:
        if (
            self._merged.compaction
            and self._merged.compaction.reserve_tokens is not None
        ):
            return self._merged.compaction.reserve_tokens
        return 16384

    def get_compaction_keep_recent_tokens(self) -> int:
        if (
            self._merged.compaction
            and self._merged.compaction.keep_recent_tokens is not None
        ):
            return self._merged.compaction.keep_recent_tokens
        return 20000

    def get_compaction_settings(self) -> dict[str, bool | int]:
        return {
            "enabled": self.get_compaction_enabled(),
            "reserve_tokens": self.get_compaction_reserve_tokens(),
            "keep_recent_tokens": self.get_compaction_keep_recent_tokens(),
        }

    def get_branch_summary_settings(self) -> dict[str, bool | int]:
        bs = self._merged.branch_summary
        return {
            "reserve_tokens": bs.reserve_tokens
            if bs and bs.reserve_tokens is not None
            else 16384,
            "skip_prompt": bs.skip_prompt
            if bs and bs.skip_prompt is not None
            else False,
        }

    def get_retry_enabled(self) -> bool:
        if self._merged.retry and self._merged.retry.enabled is not None:
            return self._merged.retry.enabled
        return True

    def set_retry_enabled(self, enabled: bool) -> None:
        if not self._global_settings.retry:
            self._global_settings.retry = RetrySettings()
        self._global_settings.retry.enabled = enabled
        self._save_global()

    def get_retry_settings(self) -> dict[str, bool | int]:
        r = self._merged.retry
        return {
            "enabled": self.get_retry_enabled(),
            "max_retries": r.max_retries if r and r.max_retries is not None else 3,
            "base_delay_ms": r.base_delay_ms
            if r and r.base_delay_ms is not None
            else 2000,
        }

    def get_provider_retry_settings(self) -> ProviderRetrySettings:
        """获取 provider 重试设置（对应 TS ``getProviderRetrySettings``）。"""
        r = self._merged.retry
        if r and r.provider is not None:
            return r.provider
        return ProviderRetrySettings()

    def get_block_images(self) -> bool:
        if self._merged.images and self._merged.images.block_images is not None:
            return self._merged.images.block_images
        return False

    def set_block_images(self, blocked: bool) -> None:
        if not self._global_settings.images:
            self._global_settings.images = ImageSettings()
        self._global_settings.images.block_images = blocked
        self._save_global()

    def get_image_auto_resize(self) -> bool:
        if self._merged.images and self._merged.images.auto_resize is not None:
            return self._merged.images.auto_resize
        return True

    def set_image_auto_resize(self, enabled: bool) -> None:
        if not self._global_settings.images:
            self._global_settings.images = ImageSettings()
        self._global_settings.images.auto_resize = enabled
        self._save_global()

    def get_show_images(self) -> bool:
        if self._merged.terminal and self._merged.terminal.show_images is not None:
            return self._merged.terminal.show_images
        return True

    def set_show_images(self, show: bool) -> None:
        if not self._global_settings.terminal:
            self._global_settings.terminal = TerminalSettings()
        self._global_settings.terminal.show_images = show
        self._save_global()

    def get_ui_mode(self) -> UiMode:
        return (
            self._merged.ui_mode if self._merged.ui_mode == "fullscreen" else "regular"
        )

    def set_ui_mode(self, mode: UiMode) -> None:
        self._global_settings.ui_mode = mode
        self._save_global()

    def get_hide_thinking_block(self) -> bool:
        return self._merged.hide_thinking_block or False

    def set_hide_thinking_block(self, hide: bool) -> None:
        self._global_settings.hide_thinking_block = hide
        self._save_global()

    def get_show_cache_miss_notices(self) -> bool:
        return self._merged.show_cache_miss_notices or False

    def set_show_cache_miss_notices(self, show: bool) -> None:
        self._global_settings.show_cache_miss_notices = show
        self._save_global()

    def get_quiet_startup(self) -> bool:
        return self._merged.quiet_startup or False

    def set_quiet_startup(self, quiet: bool) -> None:
        self._global_settings.quiet_startup = quiet
        self._save_global()

    def get_default_project_trust(self) -> DefaultProjectTrust:
        val = self._global_settings.default_project_trust
        return val if val in ("always", "never") else "ask"

    def set_default_project_trust(self, trust: DefaultProjectTrust) -> None:
        self._global_settings.default_project_trust = trust
        self._save_global()

    def get_enable_install_telemetry(self) -> bool:
        return (
            self._merged.enable_install_telemetry
            if self._merged.enable_install_telemetry is not None
            else True
        )

    def get_enable_analytics(self) -> bool:
        return self._merged.enable_analytics or False

    def get_tracking_id(self) -> str | None:
        return self._merged.tracking_id

    def set_enable_analytics(self, enabled: bool) -> None:
        import uuid

        self._global_settings.enable_analytics = enabled
        if enabled and not self._global_settings.tracking_id:
            self._global_settings.tracking_id = str(uuid.uuid4())
        self._save_global()

    def get_thinking_budgets(self) -> ThinkingBudgetsSettings | None:
        return self._merged.thinking_budgets

    def get_enabled_models(self) -> list[str] | None:
        return self._merged.enabled_models

    def set_enabled_models(self, patterns: list[str] | None) -> None:
        self._global_settings.enabled_models = patterns
        self._save_global()

    def get_double_escape_action(self) -> Literal["fork", "tree", "none"]:
        return self._merged.double_escape_action or "tree"

    def set_double_escape_action(self, action: Literal["fork", "tree", "none"]) -> None:
        self._global_settings.double_escape_action = action
        self._save_global()

    def get_tree_filter_mode(
        self,
    ) -> Literal["default", "no-tools", "user-only", "labeled-only", "all"]:
        mode = self._merged.tree_filter_mode
        valid = ("default", "no-tools", "user-only", "labeled-only", "all")
        return mode if mode in valid else "default"

    def set_tree_filter_mode(
        self, mode: Literal["default", "no-tools", "user-only", "labeled-only", "all"]
    ) -> None:
        self._global_settings.tree_filter_mode = mode
        self._save_global()

    def get_code_block_indent(self) -> str:
        if (
            self._merged.markdown
            and self._merged.markdown.code_block_indent is not None
        ):
            return self._merged.markdown.code_block_indent
        return "  "

    def get_mermaid_rendering_mode(self) -> MermaidRenderingMode:
        mode = self._merged.markdown.mermaid if self._merged.markdown else None
        return mode if mode in ("off", "final") else "streaming"

    def set_mermaid_rendering_mode(self, mode: MermaidRenderingMode) -> None:
        if not self._global_settings.markdown:
            self._global_settings.markdown = MarkdownSettings()
        self._global_settings.markdown.mermaid = mode
        self._save_global()

    def get_warnings(self) -> WarningSettings:
        return (
            self._merged.warnings.model_copy(deep=True)
            if self._merged.warnings
            else WarningSettings()
        )

    def set_warnings(self, warnings: WarningSettings) -> None:
        self._global_settings.warnings = warnings.model_copy(deep=True)
        self._save_global()

    def get_session_dir(self) -> str | None:
        return self._merged.session_dir

    def get_http_proxy(self) -> str | None:
        return self._merged.http_proxy

    def is_project_trusted(self) -> bool:
        """检查当前项目是否受信任。"""
        trust = self.get_default_project_trust()
        if trust == "always":
            return True
        if trust == "never":
            return False
        # "ask" 模式默认视为不信任
        return False

    def set_project_trusted(self, trusted: bool) -> None:
        """设置项目信任状态（运行时标志，不持久化）。"""
        self._project_trusted = trusted

    def reload(self) -> None:
        """重新加载设置。"""
        self._load()


__all__ = [
    "BranchSummarySettings",
    "CompactionSettings",
    "DefaultProjectTrust",
    "ImageSettings",
    "MarkdownSettings",
    "MermaidRenderingMode",
    "PackageSource",
    "ProviderRetrySettings",
    "RetrySettings",
    "Settings",
    "SettingsManager",
    "TerminalSettings",
    "ThinkingBudgetsSettings",
    "TransportSetting",
    "UiMode",
    "WarningSettings",
]
