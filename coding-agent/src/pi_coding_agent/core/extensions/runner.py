"""扩展运行器（对应 TS ``core/extensions/runner.ts``）。

执行扩展并管理其生命周期。ExtensionRunner 类负责将事件派发到扩展处理器，
以及管理扩展的注册、快捷键、命令等。
"""

from __future__ import annotations

import copy
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, Literal, cast

from .types import ExtensionActions, ExtensionContextActions

if TYPE_CHECKING:
    from pi_agent.types import AgentMessage
    from pi_ai.types import ImageContent, ProviderHeaders

    from ..diagnostics import ResourceDiagnostic
    from ..model_registry import ModelRegistry
    from ..session_manager import SessionManager
    from ..system_prompt import BuildSystemPromptOptions
    from .types import (
        BeforeAgentStartEvent,
        BeforeProviderHeadersEvent,
        BeforeProviderRequestEvent,
        CompactOptions,
        ContextEvent,
        ContextUsage,
        EntryRenderer,
        Extension,
        ExtensionActions,
        ExtensionCommandContextActions,
        ExtensionContextActions,
        ExtensionError,
        ExtensionEvent,
        ExtensionFlag,
        ExtensionMode,
        ExtensionRuntime,
        ExtensionShortcut,
        ExtensionUIContext,
        InputEvent,
        LoadExtensionsResult,
        MarkdownTransformer,
        MessageEndEvent,
        MessageRenderer,
        ProjectTrustContext,
        ProjectTrustEvent,
        RegisteredCommand,
        RegisteredTool,
        ResolvedCommand,
        ResourcesDiscoverEvent,
        SessionShutdownEvent,
        ToolCallEvent,
        ToolCallEventResult,
        ToolResultEvent,
        ToolResultEventResult,
        UserBashEvent,
        UserBashEventResult,
    )

# ---------------------------------------------------------------------------
# 保留快捷键（与 keybindings.json 的规范快捷键 ID 冲突）
# ---------------------------------------------------------------------------

RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS: tuple[str, ...] = (
    "app.interrupt",
    "app.clear",
    "app.exit",
    "app.suspend",
    "app.thinking.cycle",
    "app.model.cycleForward",
    "app.model.cycleBackward",
    "app.model.select",
    "app.tools.expand",
    "app.thinking.toggle",
    "app.editor.external",
    "app.message.copy",
    "app.message.followUp",
    "tui.input.submit",
    "tui.select.confirm",
    "tui.select.cancel",
    "tui.input.copy",
    "tui.editor.deleteToLineEnd",
)

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

_BuiltInKeybindings = dict[str, dict[str, Any]]


def _build_builtin_keybindings(
    resolved_keybindings: dict[str, Any],
) -> _BuiltInKeybindings:
    """构建内置快捷键映射。"""
    builtin_keybindings: _BuiltInKeybindings = {}
    for keybinding, keys in resolved_keybindings.items():
        if keys is None:
            continue
        key_list = keys if isinstance(keys, list) else [keys]
        restrict_override = keybinding in RESERVED_KEYBINDINGS_FOR_EXTENSION_CONFLICTS
        for key in key_list:
            normalized_key = key.lower()
            existing = builtin_keybindings.get(normalized_key)
            if existing and existing.get("restrictOverride") and not restrict_override:
                continue
            builtin_keybindings[normalized_key] = {
                "keybinding": keybinding,
                "restrictOverride": restrict_override,
            }
    return builtin_keybindings


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


async def emit_session_shutdown_event(
    extension_runner: ExtensionRunner,
    event: SessionShutdownEvent,
) -> bool:
    """向扩展发送会话关闭事件。

    Args:
        extension_runner: ExtensionRunner 实例。
        event: 会话关闭事件。

    Returns:
        事件是否已发送（有处理器时返回 True）。
    """
    if extension_runner.has_handlers("session_shutdown"):
        await extension_runner.emit(event)
        return True
    return False


async def emit_project_trust_event(
    extensions_result: LoadExtensionsResult,
    event: ProjectTrustEvent,
    ctx: ProjectTrustContext,
) -> dict[str, Any]:
    """向扩展发送项目信任事件。

    Args:
        extensions_result: 扩展加载结果。
        event: 项目信任事件。
        ctx: 项目信任上下文。

    Returns:
        包含 result 和 errors 的字典。
    """
    errors: list[ExtensionError] = []
    for ext in extensions_result.extensions:
        handlers = ext.handlers.get("project_trust")
        if not handlers:
            continue

        for handler in handlers:
            try:
                handler_result = await handler(event, ctx)
                if handler_result.trusted == "undecided":
                    continue
                return {"result": handler_result, "errors": errors}
            except Exception as err:
                errors.append(
                    ExtensionError(
                        extension_path=ext.path,
                        event=event.type,
                        error=str(err),
                        stack=getattr(err, "__traceback__", None),
                    )
                )
    return {"errors": errors}


# ---------------------------------------------------------------------------
# 无操作 UI 上下文
# ---------------------------------------------------------------------------

_no_op_ui_context: ExtensionUIContext | None = None


def _get_no_op_ui_context() -> ExtensionUIContext:
    """获取无操作 UI 上下文单例。"""
    global _no_op_ui_context  # noqa: F824
    if _no_op_ui_context is None:
        from .types import ExtensionUIContext

        async def _no_op_async(*args: Any, **kwargs: Any) -> None:
            return None

        _no_op_ui_context = ExtensionUIContext(
            select=_no_op_async,
            confirm=_no_op_async,  # type: ignore[arg-type]
            input=_no_op_async,
            notify=lambda *args: None,
            on_terminal_input=lambda *args: lambda: None,
            set_status=lambda *args: None,
            set_working_message=lambda *args: None,
            set_working_visible=lambda *args: None,
            set_working_indicator=lambda *args: None,
            set_hidden_thinking_label=lambda *args: None,
            set_widget=lambda *args: None,
            set_footer=lambda *args: None,
            set_header=lambda *args: None,
            set_title=lambda *args: None,
            custom=_no_op_async,
            paste_to_editor=lambda *args: None,
            set_editor_text=lambda *args: None,
            get_editor_text=lambda: "",
            editor=_no_op_async,
            add_autocomplete_provider=lambda *args: None,
            set_editor_component=lambda *args: None,
            get_editor_component=lambda: None,
            theme=None,
            get_all_themes=list,
            get_theme=lambda *args: None,
            set_theme=lambda *args: {"success": False, "error": "UI not available"},
            get_tools_expanded=lambda: False,
            set_tools_expanded=lambda *args: None,
        )
    return _no_op_ui_context


# ---------------------------------------------------------------------------
# ExtensionRunner 上下文实现
# ---------------------------------------------------------------------------


class _ExtensionContextImpl:
    """ExtensionContext 的具体实现（非 Pydantic）。

    使用惰性属性，每次访问时通过闭包获取最新值。
    """

    def __init__(self, runner: ExtensionRunner) -> None:
        self._runner = runner
        self._get_system_prompt: Callable[[], str] = lambda: ""

    @property
    def ui(self) -> ExtensionUIContext:
        self._runner._assert_active()
        return self._runner._ui_context

    @property
    def mode(self) -> ExtensionMode:
        self._runner._assert_active()
        return self._runner._mode

    @property
    def has_ui(self) -> bool:
        self._runner._assert_active()
        return self._runner.has_ui()

    @property
    def cwd(self) -> str:
        self._runner._assert_active()
        return self._runner._cwd

    @property
    def session_manager(self) -> SessionManager:
        self._runner._assert_active()
        return self._runner._session_manager

    @property
    def model_registry(self) -> ModelRegistry:
        self._runner._assert_active()
        return self._runner._model_registry

    @property
    def model(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_model()

    @property
    def scoped_models(self) -> list[Any]:
        self._runner._assert_active()
        return self._runner._get_scoped_models()

    @property
    def thinking_level(self) -> Any:
        self._runner._assert_active()
        return self._runner._runtime.get_thinking_level()

    @property
    def signal(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_signal_fn()

    def is_idle(self) -> bool:
        self._runner._assert_active()
        return self._runner._is_idle_fn()

    def is_project_trusted(self) -> bool:
        self._runner._assert_active()
        return self._runner._is_project_trusted_fn()

    def abort(self) -> None:
        self._runner._assert_active()
        self._runner._abort_fn()

    def has_pending_messages(self) -> bool:
        self._runner._assert_active()
        return self._runner._has_pending_messages_fn()

    def shutdown(self) -> None:
        self._runner._assert_active()
        self._runner._shutdown_handler()

    def get_context_usage(self) -> ContextUsage | None:
        self._runner._assert_active()
        return self._runner._get_context_usage_fn()

    def compact(self, options: CompactOptions | None = None) -> None:
        self._runner._assert_active()
        self._runner._compact_fn(options)

    def get_system_prompt(self) -> str:
        self._runner._assert_active()
        return self._get_system_prompt()


class _ExtensionCommandContextImpl(_ExtensionContextImpl):
    """ExtensionCommandContext 的具体实现。"""

    def __init__(self, runner: ExtensionRunner) -> None:
        super().__init__(runner)

    def get_system_prompt_options(self) -> Any:
        self._runner._assert_active()
        return self._runner._get_system_prompt_options_fn()

    async def wait_for_idle(self) -> None:
        self._runner._assert_active()
        return await self._runner._wait_for_idle_fn()

    async def new_session(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        self._runner._assert_active()
        return await self._runner._new_session_handler(*args, **kwargs)

    async def fork(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        self._runner._assert_active()
        return await self._runner._fork_handler(*args, **kwargs)

    async def navigate_tree(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        self._runner._assert_active()
        return await self._runner._navigate_tree_handler(*args, **kwargs)

    async def switch_session(self, *args: Any, **kwargs: Any) -> dict[str, bool]:
        self._runner._assert_active()
        return await self._runner._switch_session_handler(*args, **kwargs)

    async def reload(self) -> None:
        self._runner._assert_active()
        return await self._runner._reload_handler()


# ---------------------------------------------------------------------------
# ExtensionRunner
# ---------------------------------------------------------------------------


class ExtensionRunner:
    """扩展运行器——执行扩展并管理其生命周期。"""

    _extensions: list[Extension]
    _runtime: Any
    _ui_context: ExtensionUIContext
    _mode: ExtensionMode
    _cwd: str
    _session_manager: SessionManager
    _model_registry: ModelRegistry
    _error_listeners: set[Callable[[ExtensionError], None]]
    _get_model: Callable[[], Any]
    _get_scoped_models: Callable[[], list[Any]]
    _is_idle_fn: Callable[[], bool]
    _is_project_trusted_fn: Callable[[], bool]
    _get_signal_fn: Callable[[], Any]
    _wait_for_idle_fn: Callable[[], Awaitable[None]]
    _abort_fn: Callable[[], None]
    _has_pending_messages_fn: Callable[[], bool]
    _get_context_usage_fn: Callable[[], ContextUsage | None]
    _compact_fn: Callable[[CompactOptions | None], None]
    _get_system_prompt_fn: Callable[[], str]
    _get_system_prompt_options_fn: Callable[[], Any]
    _new_session_handler: Callable[..., Awaitable[dict[str, bool]]]
    _fork_handler: Callable[..., Awaitable[dict[str, bool]]]
    _navigate_tree_handler: Callable[..., Awaitable[dict[str, bool]]]
    _switch_session_handler: Callable[..., Awaitable[dict[str, bool]]]
    _reload_handler: Callable[[], Awaitable[None]]
    _shutdown_handler: Callable[[], None]
    _shortcut_diagnostics: list[ResourceDiagnostic]
    _command_diagnostics: list[ResourceDiagnostic]
    _stale_message: str | None

    def __init__(
        self,
        extensions: list[Extension],
        runtime: ExtensionRuntime,
        cwd: str,
        session_manager: SessionManager,
        model_registry: ModelRegistry,
    ) -> None:
        self._extensions = extensions
        self._runtime = runtime
        self._ui_context = _get_no_op_ui_context()
        self._mode = "print"
        self._cwd = cwd
        self._session_manager = session_manager
        self._model_registry = model_registry
        self._error_listeners = set()
        self._get_model = lambda: None
        self._get_scoped_models = list
        self._is_idle_fn = lambda: True
        self._is_project_trusted_fn = lambda: True
        self._get_signal_fn = lambda: None
        self._wait_for_idle_fn = lambda: None  # type: ignore[return-value,assignment]
        self._abort_fn = lambda: None
        self._has_pending_messages_fn = lambda: False
        self._get_context_usage_fn = lambda: None
        self._compact_fn = lambda *args: None
        self._get_system_prompt_fn = lambda: ""
        self._get_system_prompt_options_fn = lambda: _build_system_prompt_options(
            cwd=self._cwd
        )
        self._new_session_handler = lambda *args, **kwargs: _cancelled_false()
        self._fork_handler = lambda *args, **kwargs: _cancelled_false()
        self._navigate_tree_handler = lambda *args, **kwargs: _cancelled_false()
        self._switch_session_handler = lambda *args, **kwargs: _cancelled_false()
        self._reload_handler = lambda: None  # type: ignore[return-value,assignment]
        self._shutdown_handler = lambda: None
        self._shortcut_diagnostics = []
        self._command_diagnostics = []
        self._stale_message = None

    # ------------------------------------------------------------------
    # 核心绑定
    # ------------------------------------------------------------------

    def bind_core(
        self,
        actions: ExtensionActions | dict[str, Any],
        context_actions: ExtensionContextActions | dict[str, Any],
        provider_actions: dict[str, Any] | None = None,
    ) -> None:
        """绑定核心操作到运行时。

        Args:
            actions: 扩展操作（支持对象或字典）。
            context_actions: 上下文操作（支持对象或字典）。
            provider_actions: 提供者操作（可选）。
        """
        # 统一为对象访问（兼容 dict 传参）
        if isinstance(actions, dict):
            actions = ExtensionActions(**actions)
        if isinstance(context_actions, dict):
            context_actions = ExtensionContextActions(**context_actions)

        # 复制操作到共享运行时
        self._runtime.send_message = actions.send_message
        self._runtime.send_user_message = actions.send_user_message
        self._runtime.append_entry = actions.append_entry
        self._runtime.set_session_name = actions.set_session_name
        self._runtime.get_session_name = actions.get_session_name
        self._runtime.set_label = actions.set_label
        self._runtime.get_active_tools = actions.get_active_tools
        self._runtime.get_all_tools = actions.get_all_tools
        self._runtime.set_active_tools = actions.set_active_tools
        self._runtime.refresh_tools = actions.refresh_tools
        self._runtime.get_commands = actions.get_commands
        self._runtime.set_model = actions.set_model
        self._runtime.get_thinking_level = actions.get_thinking_level
        self._runtime.set_thinking_level = actions.set_thinking_level

        # 上下文操作
        self._get_model = context_actions.get_model
        self._get_scoped_models = context_actions.get_scoped_models
        self._is_idle_fn = context_actions.is_idle
        self._is_project_trusted_fn = context_actions.is_project_trusted
        self._get_signal_fn = context_actions.get_signal
        self._abort_fn = context_actions.abort
        self._has_pending_messages_fn = context_actions.has_pending_messages
        self._shutdown_handler = context_actions.shutdown
        self._get_context_usage_fn = context_actions.get_context_usage
        self._compact_fn = context_actions.compact
        self._get_system_prompt_fn = context_actions.get_system_prompt
        self._get_system_prompt_options_fn = (
            context_actions.get_system_prompt_options
            if context_actions.get_system_prompt_options is not None
            else (lambda: _build_system_prompt_options(cwd=self._cwd))
        )

        # 刷新加载期间排队的提供者注册
        self._flush_pending_providers(provider_actions)

    def _flush_pending_providers(self, provider_actions: dict[str, Any] | None) -> None:
        """刷新加载期间排队的提供者注册。"""

        register_provider_fn = (provider_actions or {}).get("register_provider")
        register_native_provider_fn = (provider_actions or {}).get(
            "register_native_provider"
        )
        unregister_provider_fn = (provider_actions or {}).get("unregister_provider")

        for entry in self._runtime.pending_provider_registrations:
            try:
                if register_provider_fn:
                    register_provider_fn(entry["name"], entry["config"])
                else:
                    self._model_registry.register_provider(
                        entry["name"], entry["config"]
                    )
            except Exception as err:
                self.emit_error(
                    ExtensionError(
                        extension_path=entry.get("extension_path", "<unknown>"),
                        event="register_provider",
                        error=str(err),
                    )
                )
        self._runtime.pending_provider_registrations.clear()

        for entry in self._runtime.pending_native_provider_registrations:
            try:
                provider = entry["provider"]
                if register_native_provider_fn:
                    register_native_provider_fn(provider)
                else:
                    self._model_registry.register_provider(provider)
            except Exception as err:
                self.emit_error(
                    ExtensionError(
                        extension_path=entry.get("extension_path", "<unknown>"),
                        event="register_provider",
                        error=str(err),
                    )
                )
        self._runtime.pending_native_provider_registrations.clear()

        # 从此以后，提供者注册/注销立即生效
        def _register_provider(name: str, config: Any) -> None:
            if register_provider_fn:
                register_provider_fn(name, config)
                return
            self._model_registry.register_provider(name, config)

        def _register_native_provider(provider: Any) -> None:
            if register_native_provider_fn:
                register_native_provider_fn(provider)
                return
            self._model_registry.register_provider(provider)

        def _unregister_provider_fn(name: str) -> None:
            if unregister_provider_fn:
                unregister_provider_fn(name)
                return
            self._model_registry.unregister_provider(name)

        self._runtime.register_provider = _register_provider
        self._runtime.register_native_provider = _register_native_provider
        self._runtime.unregister_provider = _unregister_provider_fn

    def bind_command_context(
        self, actions: ExtensionCommandContextActions | None = None
    ) -> None:
        """绑定命令上下文操作。

        Args:
            actions: 命令上下文操作（可选）。
        """
        if actions is not None:
            self._wait_for_idle_fn = actions.wait_for_idle
            self._new_session_handler = actions.new_session
            self._fork_handler = actions.fork
            self._navigate_tree_handler = actions.navigate_tree
            self._switch_session_handler = actions.switch_session
            self._reload_handler = actions.reload
            return

        self._wait_for_idle_fn = lambda: None  # type: ignore[return-value,assignment]
        self._new_session_handler = lambda *args, **kwargs: _cancelled_false()
        self._fork_handler = lambda *args, **kwargs: _cancelled_false()
        self._navigate_tree_handler = lambda *args, **kwargs: _cancelled_false()
        self._switch_session_handler = lambda *args, **kwargs: _cancelled_false()
        self._reload_handler = lambda: None  # type: ignore[return-value,assignment]

    # ------------------------------------------------------------------
    # UI 上下文
    # ------------------------------------------------------------------

    def set_ui_context(
        self,
        ui_context: ExtensionUIContext | None = None,
        mode: ExtensionMode = "print",
    ) -> None:
        """设置 UI 上下文。

        Args:
            ui_context: UI 上下文（可选）。
            mode: 运行模式（默认 ``"print"``）。
        """
        self._ui_context = (
            ui_context if ui_context is not None else _get_no_op_ui_context()
        )
        self._mode = mode

    def get_ui_context(self) -> ExtensionUIContext:
        """获取当前 UI 上下文。"""
        return self._ui_context

    def has_ui(self) -> bool:
        """是否有 UI。"""
        return self._ui_context is not _get_no_op_ui_context()

    # ------------------------------------------------------------------
    # 扩展信息
    # ------------------------------------------------------------------

    def get_extension_paths(self) -> list[str]:
        """获取所有扩展路径。"""
        return [e.path for e in self._extensions]

    def get_all_registered_tools(self) -> list[RegisteredTool]:
        """获取所有已注册工具（同名的第一个注册胜出）。"""

        tools_by_name: dict[str, RegisteredTool] = {}
        for ext in self._extensions:
            for tool_name, tool in ext.tools.items():
                if tool_name not in tools_by_name:
                    tools_by_name[tool_name] = tool
        return list(tools_by_name.values())

    def get_tool_definition(self, tool_name: str) -> Any | None:
        """按名称获取工具定义。"""
        for ext in self._extensions:
            tool = ext.tools.get(tool_name)
            if tool is not None:
                return tool.definition
        return None

    def get_flags(self) -> dict[str, ExtensionFlag]:
        """获取所有标志。"""
        all_flags: dict[str, ExtensionFlag] = {}
        for ext in self._extensions:
            for name, flag in ext.flags.items():
                if name not in all_flags:
                    all_flags[name] = flag
        return all_flags

    def set_flag_value(self, name: str, value: bool | str) -> None:
        """设置标志值。"""
        self._runtime.flag_values[name] = value

    def get_flag_values(self) -> dict[str, bool | str]:
        """获取所有标志值。"""
        return dict(self._runtime.flag_values)

    def get_shortcuts(
        self, resolved_keybindings: dict[str, Any]
    ) -> dict[str, ExtensionShortcut]:
        """获取所有快捷键。

        Args:
            resolved_keybindings: 已解析的快捷键配置。

        Returns:
            快捷键映射。
        """
        self._shortcut_diagnostics = []
        builtin_keybindings = _build_builtin_keybindings(resolved_keybindings)
        extension_shortcuts: dict[str, ExtensionShortcut] = {}

        def _add_diagnostic(message: str, extension_path: str) -> None:
            from ..diagnostics import ResourceDiagnostic

            self._shortcut_diagnostics.append(
                ResourceDiagnostic(type="warning", message=message, path=extension_path)
            )
            if not self.has_ui():
                import warnings

                warnings.warn(message)

        for ext in self._extensions:
            for key, shortcut in ext.shortcuts.items():
                normalized_key = key.lower()

                built_in = builtin_keybindings.get(normalized_key)
                if built_in and built_in.get("restrictOverride") is True:
                    _add_diagnostic(
                        f"Extension shortcut '{key}' from {shortcut.extension_path} "
                        f"conflicts with built-in shortcut. Skipping.",
                        shortcut.extension_path,
                    )
                    continue

                if built_in and built_in.get("restrictOverride") is False:
                    _add_diagnostic(
                        f"Extension shortcut conflict: '{key}' is built-in shortcut for "
                        f"{built_in['keybinding']} and {shortcut.extension_path}. "
                        f"Using {shortcut.extension_path}.",
                        shortcut.extension_path,
                    )

                existing = extension_shortcuts.get(normalized_key)
                if existing is not None:
                    _add_diagnostic(
                        f"Extension shortcut conflict: '{key}' registered by both "
                        f"{existing.extension_path} and {shortcut.extension_path}. "
                        f"Using {shortcut.extension_path}.",
                        shortcut.extension_path,
                    )

                extension_shortcuts[normalized_key] = shortcut

        return extension_shortcuts

    def get_shortcut_diagnostics(self) -> list[ResourceDiagnostic]:
        """获取快捷键诊断信息。"""
        return self._shortcut_diagnostics

    # ------------------------------------------------------------------
    # 失效 / 活跃检查
    # ------------------------------------------------------------------

    def invalidate(self, message: str | None = None) -> None:
        """标记运行时为失效状态。

        Args:
            message: 失效消息（可选）。
        """
        if self._stale_message is None:
            msg = (
                message
                or "This extension ctx is stale after session replacement or reload. "
                "Do not use a captured pi or command ctx after ctx.new_session(), ctx.fork(), "
                "ctx.switch_session(), or ctx.reload(). For newSession, fork, and switchSession, "
                "move post-replacement work into withSession and use the ctx passed to withSession. "
                "For reload, do not use the old ctx after await ctx.reload()."
            )
            self._stale_message = msg
            self._runtime.invalidate(msg)

    def _assert_active(self) -> None:
        """断言运行时仍活跃。"""
        if self._stale_message is not None:
            raise RuntimeError(self._stale_message)

    # ------------------------------------------------------------------
    # 错误处理
    # ------------------------------------------------------------------

    def on_error(
        self, listener: Callable[[ExtensionError], None]
    ) -> Callable[[], None]:
        """注册错误监听器。

        Args:
            listener: 错误监听器。

        Returns:
            取消订阅函数。
        """
        self._error_listeners.add(listener)
        return lambda: self._error_listeners.discard(listener)

    def emit_error(self, error: ExtensionError) -> None:
        """发送错误事件到所有监听器。

        Args:
            error: 扩展错误。
        """
        for listener in self._error_listeners:
            listener(error)

    # ------------------------------------------------------------------
    # 处理器查询
    # ------------------------------------------------------------------

    def has_handlers(self, event_type: str) -> bool:
        """检查是否有任何扩展注册了指定事件类型的处理器。

        Args:
            event_type: 事件类型。

        Returns:
            是否有处理器。
        """
        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            if handlers:
                return True
        return False

    def get_message_renderer(self, custom_type: str) -> MessageRenderer | None:
        """获取消息渲染器。

        Args:
            custom_type: 自定义类型。

        Returns:
            消息渲染器，或 None。
        """
        for ext in self._extensions:
            renderer = ext.message_renderers.get(custom_type)
            if renderer is not None:
                return renderer
        return None

    def get_markdown_transformers(self) -> list[MarkdownTransformer]:
        """获取所有 Markdown 转换器。"""
        result: list[MarkdownTransformer] = []
        for ext in self._extensions:
            if ext.markdown_transformer is not None:
                result.append(ext.markdown_transformer)
        return result

    def get_entry_renderer(self, custom_type: str) -> EntryRenderer | None:
        """获取条目渲染器。

        Args:
            custom_type: 自定义类型。

        Returns:
            条目渲染器，或 None。
        """
        for ext in self._extensions:
            if ext.entry_renderers is not None:
                renderer = ext.entry_renderers.get(custom_type)
                if renderer is not None:
                    return renderer
        return None

    # ------------------------------------------------------------------
    # 命令解析
    # ------------------------------------------------------------------

    def _resolve_registered_commands(self) -> list[ResolvedCommand]:
        """解析已注册命令，处理重复名称。"""
        from .types import ResolvedCommand

        commands: list[RegisteredCommand] = []
        counts: dict[str, int] = {}

        for ext in self._extensions:
            for command in ext.commands.values():
                commands.append(command)
                counts[command.name] = counts.get(command.name, 0) + 1

        seen: dict[str, int] = {}
        taken_invocation_names: set[str] = set()
        resolved: list[ResolvedCommand] = []

        for command in commands:
            occurrence = seen.get(command.name, 0) + 1
            seen[command.name] = occurrence

            invocation_name = command.name
            total = counts.get(command.name, 0)
            if total is not None and total > 1:
                invocation_name = f"{command.name}:{occurrence}"

            while invocation_name in taken_invocation_names:
                occurrence += 1
                invocation_name = f"{command.name}:{occurrence}"

            taken_invocation_names.add(invocation_name)
            resolved.append(
                ResolvedCommand(
                    name=command.name,
                    source_info=command.source_info,
                    description=command.description,
                    get_argument_completions=command.get_argument_completions,
                    handler=command.handler,
                    invocation_name=invocation_name,
                )
            )

        return resolved

    def get_model_registry(self) -> ModelRegistry:
        """获取模型注册表。"""
        return self._model_registry

    def get_registered_commands(self) -> list[ResolvedCommand]:
        """获取所有已注册命令。"""
        self._command_diagnostics = []
        return self._resolve_registered_commands()

    def get_command_diagnostics(self) -> list[ResourceDiagnostic]:
        """获取命令诊断信息。"""
        return self._command_diagnostics

    def get_command(self, name: str) -> ResolvedCommand | None:
        """按名称获取命令。

        Args:
            name: 命令调用名。

        Returns:
            已解析的命令，或 None。
        """
        for cmd in self._resolve_registered_commands():
            if cmd.invocation_name == name:
                return cmd
        return None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        """请求优雅关闭。"""
        self._shutdown_handler()

    def get_active_tools(self) -> list[str]:
        """获取活跃工具列表。"""
        self._assert_active()
        return self._runtime.get_active_tools()  # type: ignore[no-any-return]

    # ------------------------------------------------------------------
    # 上下文创建
    # ------------------------------------------------------------------

    def create_context(self) -> _ExtensionContextImpl:
        """创建 _ExtensionContextImpl 实例。

        上下文值在调用时解析，因此 bind_core/bindUI 的更改会反映出来。
        """
        return _ExtensionContextImpl(self)

    def create_command_context(self) -> _ExtensionCommandContextImpl:
        """创建 _ExtensionCommandContextImpl 实例。"""
        return _ExtensionCommandContextImpl(self)

    # ------------------------------------------------------------------
    # 事件发送
    # ------------------------------------------------------------------

    async def emit(self, event: ExtensionEvent) -> Any:
        """发送事件到所有扩展处理器。

        Args:
            event: 扩展事件。

        Returns:
            事件结果（session_before_* 事件的取消结果，或 None）。
        """
        ctx = self.create_context()
        result: Any = None

        # 兼容 dict 和 Pydantic model 两种事件格式
        event_type = event.get("type", "") if isinstance(event, dict) else event.type

        for ext in self._extensions:
            handlers = ext.handlers.get(event_type, [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(event, ctx)

                    if handler_result is not None and event_type in (
                        "session_before_switch",
                        "session_before_fork",
                        "session_before_compact",
                        "session_before_tree",
                    ):
                        result = handler_result
                        if getattr(result, "cancel", False):
                            return result
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event=event_type,
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return result

    async def emit_message_end(self, event: MessageEndEvent) -> AgentMessage | None:
        """发送消息结束事件。

        Args:
            event: 消息结束事件。

        Returns:
            修改后的消息，或 None（未修改时）。
        """
        ctx = self.create_context()
        current_message = event.message
        modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("message_end", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    current_event = MessageEndEvent(
                        type="message_end", message=current_message
                    )
                    handler_result = await handler(current_event, ctx)
                    if handler_result is None or handler_result.message is None:
                        continue

                    if handler_result.message.role != current_message.role:
                        self.emit_error(
                            ExtensionError(
                                extension_path=ext.path,
                                event="message_end",
                                error="message_end handlers must return a message with the same role",
                            )
                        )
                        continue

                    current_message = handler_result.message
                    modified = True
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="message_end",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return current_message if modified else None

    async def emit_tool_result(
        self, event: ToolResultEvent
    ) -> ToolResultEventResult | None:
        """发送工具结果事件。

        Args:
            event: 工具结果事件。

        Returns:
            修改后的结果，或 None。
        """
        from .types import ToolResultEventResult

        ctx = self.create_context()
        current = copy.deepcopy(event)
        modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_result", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(current, ctx)
                    if handler_result is None:
                        continue

                    if handler_result.content is not None:
                        current.content = handler_result.content
                        modified = True
                    if handler_result.details is not None:
                        current.details = handler_result.details
                        modified = True
                    if handler_result.is_error is not None:
                        current.is_error = handler_result.is_error
                        modified = True
                    if handler_result.usage is not None:
                        current.usage = handler_result.usage
                        modified = True
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="tool_result",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        if not modified:
            return None

        return ToolResultEventResult(
            content=current.content,
            details=current.details,
            is_error=current.is_error,
            usage=current.usage,
        )

    async def emit_tool_call(self, event: ToolCallEvent) -> ToolCallEventResult | None:
        """发送工具调用事件。

        Args:
            event: 工具调用事件。

        Returns:
            工具调用结果，或 None。
        """
        ctx = self.create_context()
        result: ToolCallEventResult | None = None

        for ext in self._extensions:
            handlers = ext.handlers.get("tool_call", [])
            if not handlers:
                continue

            for handler in handlers:
                handler_result = await handler(event, ctx)
                if handler_result is not None:
                    result = handler_result
                    if getattr(result, "block", False):
                        return result

        return result

    async def emit_user_bash(self, event: UserBashEvent) -> UserBashEventResult | None:
        """发送用户 Bash 事件。

        Args:
            event: 用户 Bash 事件。

        Returns:
            Bash 事件结果，或 None。
        """
        ctx = self.create_context()

        for ext in self._extensions:
            handlers = ext.handlers.get("user_bash", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    handler_result = await handler(event, ctx)
                    if handler_result is not None:
                        return handler_result  # type: ignore[no-any-return]
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="user_bash",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return None

    async def emit_context(self, messages: list[AgentMessage]) -> list[AgentMessage]:
        """发送上下文事件。

        Args:
            messages: 当前消息列表。

        Returns:
            修改后的消息列表。
        """
        from copy import deepcopy

        ctx = self.create_context()
        current_messages = deepcopy(messages)

        for ext in self._extensions:
            handlers = ext.handlers.get("context", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = ContextEvent(type="context", messages=current_messages)
                    handler_result = await handler(event, ctx)
                    if (
                        handler_result is not None
                        and handler_result.messages is not None
                    ):
                        current_messages = handler_result.messages
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="context",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return current_messages

    async def emit_before_provider_request(self, payload: Any) -> Any:
        """发送提供者请求前事件。

        Args:
            payload: 请求负载。

        Returns:
            修改后的负载。
        """
        ctx = self.create_context()
        current_payload = payload

        for ext in self._extensions:
            handlers = ext.handlers.get("before_provider_request", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = BeforeProviderRequestEvent(
                        type="before_provider_request", payload=current_payload
                    )
                    handler_result = await handler(event, ctx)
                    if handler_result is not None:
                        current_payload = handler_result
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_provider_request",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return current_payload

    async def emit_before_provider_headers(
        self, headers: ProviderHeaders
    ) -> ProviderHeaders:
        """发送提供者请求头前事件。

        Args:
            headers: 请求头。

        Returns:
            修改后的请求头。
        """
        ctx = self.create_context()

        for ext in self._extensions:
            handlers = ext.handlers.get("before_provider_headers", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = BeforeProviderHeadersEvent(
                        type="before_provider_headers", headers=headers
                    )
                    await handler(event, ctx)
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_provider_headers",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return headers

    async def emit_before_agent_start(
        self,
        prompt: str,
        images: list[ImageContent] | None,
        system_prompt: str,
        system_prompt_options: BuildSystemPromptOptions,
    ) -> dict[str, Any] | None:
        """发送 agent 启动前事件。

        Args:
            prompt: 用户提示词。
            images: 图片列表。
            system_prompt: 系统提示词。
            system_prompt_options: 系统提示词构建选项。

        Returns:
            合并结果，或 None。
        """
        current_system_prompt = system_prompt
        ctx = self.create_context()
        ctx._get_system_prompt = lambda: current_system_prompt
        messages: list[Any] = []
        system_prompt_modified = False

        for ext in self._extensions:
            handlers = ext.handlers.get("before_agent_start", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = BeforeAgentStartEvent(
                        type="before_agent_start",
                        prompt=prompt,
                        images=images,
                        system_prompt=current_system_prompt,
                        system_prompt_options=system_prompt_options,
                    )
                    handler_result = await handler(event, ctx)
                    if handler_result is not None:
                        if handler_result.message is not None:
                            messages.append(handler_result.message)
                        if handler_result.system_prompt is not None:
                            current_system_prompt = handler_result.system_prompt
                            system_prompt_modified = True
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="before_agent_start",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        if messages or system_prompt_modified:
            return {
                "messages": messages if messages else None,
                "systemPrompt": current_system_prompt
                if system_prompt_modified
                else None,
            }

        return None

    async def emit_resources_discover(
        self,
        cwd: str,
        reason: str,
    ) -> dict[str, list[dict[str, str]]]:
        """发送资源发现事件。

        Args:
            cwd: 当前工作目录。
            reason: 发现原因（"startup" 或 "reload"）。

        Returns:
            发现的资源路径。
        """
        ctx = self.create_context()
        skill_paths: list[dict[str, str]] = []
        prompt_paths: list[dict[str, str]] = []
        theme_paths: list[dict[str, str]] = []

        for ext in self._extensions:
            handlers = ext.handlers.get("resources_discover", [])
            if not handlers:
                continue

            for handler in handlers:
                try:
                    event = ResourcesDiscoverEvent(
                        type="resources_discover",
                        cwd=cwd,
                        reason=cast(Literal["startup", "reload"], reason),
                    )
                    handler_result = await handler(event, ctx)
                    if handler_result is not None:
                        if handler_result.skill_paths:
                            skill_paths.extend(
                                {"path": p, "extensionPath": ext.path}
                                for p in handler_result.skill_paths
                            )
                        if handler_result.prompt_paths:
                            prompt_paths.extend(
                                {"path": p, "extensionPath": ext.path}
                                for p in handler_result.prompt_paths
                            )
                        if handler_result.theme_paths:
                            theme_paths.extend(
                                {"path": p, "extensionPath": ext.path}
                                for p in handler_result.theme_paths
                            )
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="resources_discover",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        return {
            "skillPaths": skill_paths,
            "promptPaths": prompt_paths,
            "themePaths": theme_paths,
        }

    async def emit_input(
        self,
        text: str,
        images: list[ImageContent] | None,
        source: str,
        streaming_behavior: str | None = None,
    ) -> Any:
        """发送输入事件。转换链，"handled" 短路。

        Args:
            text: 输入文本。
            images: 图片列表。
            source: 输入来源。
            streaming_behavior: 流式行为（可选）。

        Returns:
            输入事件结果字典。
        """
        ctx = self.create_context()
        current_text = text
        current_images = images

        for ext in self._extensions:
            for handler in ext.handlers.get("input", []):
                try:
                    event = InputEvent(
                        type="input",
                        text=current_text,
                        images=current_images,
                        source=cast(Literal["interactive", "rpc", "extension"], source),
                        streaming_behavior=cast(
                            Literal["steer", "followUp"] | None, streaming_behavior
                        ),
                    )
                    result = await handler(event, ctx)
                    if result is not None:
                        if result.action == "handled":
                            return result
                        if result.action == "transform":
                            current_text = result.text
                            if result.images is not None:
                                current_images = result.images
                except Exception as err:
                    self.emit_error(
                        ExtensionError(
                            extension_path=ext.path,
                            event="input",
                            error=str(err),
                            stack=getattr(err, "__traceback__", None),
                        )
                    )

        if current_text != text or current_images is not images:
            return {
                "action": "transform",
                "text": current_text,
                "images": current_images,
            }

        return {"action": "continue"}


# ---------------------------------------------------------------------------
# 模块级辅助函数
# ---------------------------------------------------------------------------


def _build_system_prompt_options(cwd: str) -> Any:
    """构建系统提示词选项。"""
    from ..system_prompt import BuildSystemPromptOptions

    return BuildSystemPromptOptions(cwd=cwd)


async def _cancelled_false() -> dict[str, bool]:
    """返回 ``{"cancelled": False}``。"""
    return {"cancelled": False}
