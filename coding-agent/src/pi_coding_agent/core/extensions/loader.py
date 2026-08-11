"""扩展加载器（对应 TS ``core/extensions/loader.ts``）。

处理从文件/目录加载扩展、在 agent 目录中发现扩展、创建 ExtensionRuntime。
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from ..event_bus import EventBus
from ..exec import ExecOptions, exec_command
from ..pi_manifest import read_pi_manifest
from ..source_info import create_synthetic_source_info
from ..timings import time

if TYPE_CHECKING:
    from collections.abc import Callable

    from .types import (
        Extension,
        ExtensionAPI,
        ExtensionRuntime,
        HandlerFn,
        MarkdownTransformer,
        ToolDefinition,
    )

# ---------------------------------------------------------------------------
# 缓存
# ---------------------------------------------------------------------------

_extension_cache: dict[str, Callable[..., Any]] = {}
_extension_cache_cwd: str | None = None
_extension_cache_generation = 0


def clear_extension_cache() -> None:
    """清空扩展模块缓存。"""
    global _extension_cache_cwd, _extension_cache_generation  # noqa: F824
    _extension_cache.clear()
    _extension_cache_cwd = None
    _extension_cache_generation += 1


def _use_extension_cache_cwd(cwd: str) -> dict[str, Any]:
    """管理缓存工作目录，在 cwd 变更时清空缓存。"""
    global _extension_cache_cwd  # noqa: F824
    resolved_cwd = str(Path(cwd).resolve())
    if _extension_cache_cwd is not None and _extension_cache_cwd != resolved_cwd:
        clear_extension_cache()
    _extension_cache_cwd = resolved_cwd
    return {"cwd": resolved_cwd, "generation": _extension_cache_generation}


# ---------------------------------------------------------------------------
# 运行时创建
# ---------------------------------------------------------------------------


class _ExtensionRuntimeImpl:
    """扩展运行时的具体实现。

    使用 plain class 而非 Pydantic BaseModel，因为需要在运行时动态替换方法。
    """

    def __init__(self) -> None:
        def _not_initialized() -> Any:
            msg = "Extension runtime not initialized. Action methods cannot be called during extension loading."
            raise RuntimeError(msg)

        self.send_message: Callable[..., Any] = _not_initialized
        self.send_user_message: Callable[..., Any] = _not_initialized
        self.append_entry: Callable[..., Any] = _not_initialized
        self.set_session_name: Callable[..., Any] = _not_initialized
        self.get_session_name: Callable[..., Any] = _not_initialized
        self.set_label: Callable[..., Any] = _not_initialized
        self.get_active_tools: Callable[..., Any] = _not_initialized
        self.get_all_tools: Callable[..., Any] = _not_initialized
        self.set_active_tools: Callable[..., Any] = _not_initialized
        self.refresh_tools: Callable[..., Any] = lambda: None
        self.get_commands: Callable[..., Any] = _not_initialized
        self.set_model: Callable[..., Any] = lambda: _not_initialized()
        self.get_thinking_level: Callable[..., Any] = _not_initialized
        self.set_thinking_level: Callable[..., Any] = _not_initialized

        self.flag_values: dict[str, bool | str] = {}
        self.pending_provider_registrations: list[dict[str, Any]] = []
        self.pending_native_provider_registrations: list[dict[str, Any]] = []
        self._stale_message: str | None = None

        def _assert_active() -> None:
            if self._stale_message is not None:
                raise RuntimeError(self._stale_message)

        def _invalidate(message: str | None = None) -> None:
            if self._stale_message is None:
                self._stale_message = (
                    message
                    or "This extension ctx is stale after session replacement or reload. "
                    "Do not use a captured pi or command ctx after ctx.new_session(), ctx.fork(), "
                    "ctx.switch_session(), or ctx.reload(). For newSession, fork, and switchSession, "
                    "move post-replacement work into withSession and use the ctx passed to withSession. "
                    "For reload, do not use the old ctx after await ctx.reload()."
                )

        self.assert_active = _assert_active
        self.invalidate = _invalidate

        def _register_provider(
            name: str, config: Any, extension_path: str = "<unknown>"
        ) -> None:
            self.pending_provider_registrations.append(
                {"name": name, "config": config, "extension_path": extension_path}
            )

        def _register_native_provider(
            provider: Any, extension_path: str = "<unknown>"
        ) -> None:
            self.pending_native_provider_registrations.append(
                {"provider": provider, "extension_path": extension_path}
            )

        def _unregister_provider(name: str) -> None:
            self.pending_provider_registrations = [
                r for r in self.pending_provider_registrations if r["name"] != name
            ]
            self.pending_native_provider_registrations = [
                r
                for r in self.pending_native_provider_registrations
                if r.get("provider", {}).get("id") != name
            ]

        self.register_provider = _register_provider
        self.register_native_provider = _register_native_provider
        self.unregister_provider = _unregister_provider


def create_extension_runtime() -> ExtensionRuntime:
    """创建带有抛出存根的运行时。Runner.bind_core() 会替换为真实实现。

    Returns:
        新的 ExtensionRuntime 实例。
    """
    return _ExtensionRuntimeImpl()  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Extension API 创建
# ---------------------------------------------------------------------------


def _create_extension_api(
    extension: Extension,
    runtime: _ExtensionRuntimeImpl,
    cwd: str,
    event_bus: EventBus,
) -> ExtensionAPI:
    """为扩展创建 ExtensionAPI 实例。

    注册方法写入扩展对象。操作方法委托给共享运行时。
    """
    from .types import ExtensionAPI, RegisteredTool

    class _ExtensionAPIImpl(ExtensionAPI):
        """扩展 API 实现。"""

        def on(self, event: str, handler: HandlerFn) -> None:
            runtime.assert_active()
            handlers = extension.handlers.get(event, [])
            handlers.append(handler)
            extension.handlers[event] = handlers

        def register_tool(self, tool: ToolDefinition) -> None:
            runtime.assert_active()
            extension.tools[tool.name] = RegisteredTool(
                definition=tool, source_info=extension.source_info
            )
            runtime.refresh_tools()

        def register_command(self, name: str, options: dict[str, Any]) -> None:
            runtime.assert_active()
            from .types import RegisteredCommand

            extension.commands[name] = RegisteredCommand(
                name=name,
                source_info=extension.source_info,
                **{
                    k: v
                    for k, v in options.items()
                    if k in ("description", "get_argument_completions", "handler")
                },
            )

        def register_shortcut(self, shortcut: str, options: dict[str, Any]) -> None:
            runtime.assert_active()
            from .types import ExtensionShortcut

            extension.shortcuts[shortcut] = ExtensionShortcut(
                shortcut=shortcut,
                extension_path=extension.path,
                **{k: v for k, v in options.items() if k in ("description", "handler")},
            )

        def register_flag(self, name: str, options: dict[str, Any]) -> None:
            runtime.assert_active()
            from .types import ExtensionFlag

            flag_type: str = options.get("type", "boolean")
            default_val = options.get("default")
            extension.flags[name] = ExtensionFlag(
                name=name,
                extension_path=extension.path,
                description=options.get("description"),
                type=cast("Any", flag_type),
                default=default_val,
            )
            if default_val is not None and name not in runtime.flag_values:
                runtime.flag_values[name] = default_val

        def register_message_renderer(self, custom_type: str, renderer: Any) -> None:
            runtime.assert_active()
            extension.message_renderers[custom_type] = renderer

        def register_markdown_transformer(
            self, transformer: MarkdownTransformer
        ) -> None:
            runtime.assert_active()
            extension.markdown_transformer = transformer

        def register_entry_renderer(self, custom_type: str, renderer: Any) -> None:
            runtime.assert_active()
            if extension.entry_renderers is None:
                extension.entry_renderers = {}
            extension.entry_renderers[custom_type] = renderer

        def get_flag(self, name: str) -> bool | str | None:
            runtime.assert_active()
            if name not in extension.flags:
                return None
            return runtime.flag_values.get(name)

        def send_message(
            self, message: dict[str, Any], options: dict[str, Any] | None = None
        ) -> None:
            runtime.assert_active()
            runtime.send_message(message, options)

        def send_user_message(
            self, content: str | list[Any], options: dict[str, Any] | None = None
        ) -> None:
            runtime.assert_active()
            runtime.send_user_message(content, options)

        def append_entry(self, custom_type: str, data: Any = None) -> None:
            runtime.assert_active()
            runtime.append_entry(custom_type, data)

        def set_session_name(self, name: str) -> None:
            runtime.assert_active()
            runtime.set_session_name(name)

        def get_session_name(self) -> str | None:
            runtime.assert_active()
            return runtime.get_session_name()  # type: ignore[no-any-return]

        def set_label(self, entry_id: str, label: str | None) -> None:
            runtime.assert_active()
            runtime.set_label(entry_id, label)

        async def exec(
            self, command: str, args: list[str], options: ExecOptions | None = None
        ) -> Any:
            runtime.assert_active()
            resolved_cwd = options.cwd if options and options.cwd else cwd
            return await exec_command(command, args, resolved_cwd, options)

        def get_active_tools(self) -> list[str]:
            runtime.assert_active()
            return runtime.get_active_tools()  # type: ignore[no-any-return]

        def get_all_tools(self) -> list[Any]:
            runtime.assert_active()
            return runtime.get_all_tools()  # type: ignore[no-any-return]

        def set_active_tools(self, tool_names: list[str]) -> None:
            runtime.assert_active()
            runtime.set_active_tools(tool_names)

        def get_commands(self) -> list[Any]:
            runtime.assert_active()
            return runtime.get_commands()  # type: ignore[no-any-return]

        async def set_model(self, model: Any) -> bool:
            runtime.assert_active()
            return await runtime.set_model(model)  # type: ignore[no-any-return]

        def get_thinking_level(self) -> Any:
            runtime.assert_active()
            return runtime.get_thinking_level()

        def set_thinking_level(self, level: Any) -> None:
            runtime.assert_active()
            runtime.set_thinking_level(level)

        def register_provider(self, name_or_provider: Any, config: Any = None) -> None:
            runtime.assert_active()
            if isinstance(name_or_provider, str):
                if config is None:
                    raise ValueError(
                        "Provider config is required when registering by name"
                    )
                runtime.register_provider(name_or_provider, config, extension.path)
            else:
                runtime.register_native_provider(name_or_provider, extension.path)

        def unregister_provider(self, name: str) -> None:
            runtime.assert_active()
            runtime.unregister_provider(name)

        @property
        def events(self) -> EventBus:
            return event_bus

    return _ExtensionAPIImpl()


# ---------------------------------------------------------------------------
# 扩展对象创建
# ---------------------------------------------------------------------------


def _create_extension(extension_path: str, resolved_path: str) -> Extension:
    """创建带有空集合的 Extension 对象。

    Args:
        extension_path: 原始扩展路径。
        resolved_path: 解析后的绝对路径。

    Returns:
        新的 Extension 实例。
    """
    from .types import Extension

    if extension_path.startswith("<") and extension_path.endswith(">"):
        source = (extension_path[1:-1].split(":")[0]) or "temporary"
    else:
        source = "local"

    base_dir: str | None = None
    if not extension_path.startswith("<"):
        base_dir = str(Path(resolved_path).parent)

    return Extension(
        path=extension_path,
        resolved_path=resolved_path,
        source_info=create_synthetic_source_info(
            extension_path, {"source": source, "base_dir": base_dir}
        ),
        handlers={},
        tools={},
        message_renderers={},
        entry_renderers=None,
        commands={},
        flags={},
        shortcuts={},
    )


# ---------------------------------------------------------------------------
# 扩展模块加载
# ---------------------------------------------------------------------------


async def _load_extension_module(
    extension_path: str, cache_token: dict[str, Any] | None = None
) -> Any:
    """动态加载扩展 Python 模块并返回工厂函数。

    Args:
        extension_path: 模块文件路径。
        cache_token: 缓存令牌（含 cwd 和 generation）。

    Returns:
        扩展工厂函数，或 None。
    """
    is_current = (
        cache_token is not None
        and _extension_cache_cwd == cache_token.get("cwd")
        and _extension_cache_generation == cache_token.get("generation", 0)
    )

    if is_current and extension_path in _extension_cache:
        return _extension_cache[extension_path]

    path = Path(extension_path)
    if not path.exists():
        return None

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return None

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    factory: Any = getattr(module, "default", None)
    if factory is None:
        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if callable(attr) and not attr_name.startswith("_"):
                factory = attr
                break

    if factory is None or not callable(factory):
        return None

    if is_current:
        _extension_cache[extension_path] = factory

    return factory


# ---------------------------------------------------------------------------
# 单个扩展加载
# ---------------------------------------------------------------------------


async def _load_extension(
    extension_path: str,
    cwd: str,
    event_bus: EventBus,
    runtime: _ExtensionRuntimeImpl,
    cache_token: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """加载单个扩展。

    Args:
        extension_path: 扩展路径。
        cwd: 当前工作目录。
        event_bus: 事件总线。
        runtime: 扩展运行时。
        cache_token: 缓存令牌。

    Returns:
        ``{"extension": Extension | None, "error": str | None}``。
    """
    resolved_path = str(Path(extension_path).resolve())

    try:
        factory = await _load_extension_module(resolved_path, cache_token)
        time(f"{extension_path} module import", "extensions")

        if factory is None:
            return {
                "extension": None,
                "error": f"Extension does not export a valid factory function: {extension_path}",
            }

        extension = _create_extension(extension_path, resolved_path)
        api = _create_extension_api(extension, runtime, cwd, event_bus)
        result = factory(api)
        if result is not None:
            await result
        time(f"{extension_path} factory", "extensions")

        return {"extension": extension, "error": None}
    except Exception as err:
        message = str(err)
        return {"extension": None, "error": f"Failed to load extension: {message}"}


# ---------------------------------------------------------------------------
# 内联工厂加载
# ---------------------------------------------------------------------------


async def load_extension_from_factory(
    factory: Any,
    cwd: str,
    event_bus: EventBus,
    runtime: ExtensionRuntime,
    extension_path: str = "<inline>",
) -> Extension:
    """从内联工厂函数创建扩展。

    Args:
        factory: 扩展工厂函数。
        cwd: 当前工作目录。
        event_bus: 事件总线。
        runtime: 扩展运行时。
        extension_path: 扩展路径标签（默认 ``"<inline>"``）。

    Returns:
        加载的扩展。
    """

    extension = _create_extension(extension_path, extension_path)
    resolved_cwd = str(Path(cwd).resolve())
    impl: _ExtensionRuntimeImpl = runtime  # type: ignore[assignment]
    api = _create_extension_api(extension, impl, resolved_cwd, event_bus)
    result = factory(api)
    if result is not None:
        await result
    time(f"{extension_path} factory", "extensions")
    return extension


# ---------------------------------------------------------------------------
# 批量扩展加载
# ---------------------------------------------------------------------------


async def _load_extensions_internal(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
    use_cache: bool = False,
) -> dict[str, Any]:
    """从路径列表加载扩展。

    Args:
        paths: 扩展路径列表。
        cwd: 当前工作目录。
        event_bus: 事件总线（可选）。
        runtime: 扩展运行时（可选）。
        use_cache: 是否使用缓存。

    Returns:
        ``{"extensions": [...], "errors": [...], "runtime": ...}``。
    """

    extensions: list[Extension] = []
    errors: list[dict[str, str]] = []
    cache_token = _use_extension_cache_cwd(cwd) if use_cache else None
    resolved_cwd = cache_token["cwd"] if cache_token else str(Path(cwd).resolve())
    resolved_event_bus = event_bus if event_bus is not None else EventBus()
    resolved_runtime = runtime if runtime is not None else create_extension_runtime()
    impl: _ExtensionRuntimeImpl = resolved_runtime  # type: ignore[assignment]

    for ext_path in paths:
        result = await _load_extension(
            ext_path, resolved_cwd, resolved_event_bus, impl, cache_token
        )

        error = result.get("error")
        if error:
            errors.append({"path": ext_path, "error": error})
            continue

        extension = result.get("extension")
        if extension is not None:
            extensions.append(extension)

    return {
        "extensions": extensions,
        "errors": errors,
        "runtime": resolved_runtime,
    }


async def load_extensions(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
) -> dict[str, Any]:
    """加载扩展（无缓存）。

    Args:
        paths: 扩展路径列表。
        cwd: 当前工作目录。
        event_bus: 事件总线（可选）。
        runtime: 扩展运行时（可选）。

    Returns:
        加载结果字典。
    """
    return await _load_extensions_internal(
        paths, cwd, event_bus, runtime, use_cache=False
    )


async def load_extensions_cached(
    paths: list[str],
    cwd: str,
    event_bus: EventBus | None = None,
    runtime: ExtensionRuntime | None = None,
) -> dict[str, Any]:
    """加载扩展（使用缓存）。

    Args:
        paths: 扩展路径列表。
        cwd: 当前工作目录。
        event_bus: 事件总线（可选）。
        runtime: 扩展运行时（可选）。

    Returns:
        加载结果字典。
    """
    return await _load_extensions_internal(
        paths, cwd, event_bus, runtime, use_cache=True
    )


# ---------------------------------------------------------------------------
# 扩展发现
# ---------------------------------------------------------------------------


def _is_extension_file(name: str) -> bool:
    """判断文件名是否为扩展文件。

    Args:
        name: 文件名。

    Returns:
        是否为扩展文件。
    """
    return name.endswith(".py") and not name.startswith("__")


def _resolve_extension_entries(dir_path: str) -> list[str] | None:
    """从目录解析扩展入口点。

    检查：
    1. package.json 中的 "pi.extensions" 字段 -> 返回声明路径
    2. __init__.py -> 返回目录本身（作为包）

    Args:
        dir_path: 目录路径。

    Returns:
        解析后的路径列表，或 None。
    """
    dir_path_obj = Path(dir_path)

    # 1. 检查 package.json 中的 pi 字段
    package_json_path = dir_path_obj / "package.json"
    if package_json_path.exists():
        manifest = read_pi_manifest(str(package_json_path))
        if manifest and manifest.extensions:
            entries: list[str] = []
            for ext_path in manifest.extensions:
                resolved_ext_path = str((dir_path_obj / ext_path).resolve())
                if Path(resolved_ext_path).exists():
                    entries.append(resolved_ext_path)
            if entries:
                return entries

    # 2. 检查 __init__.py
    init_py = dir_path_obj / "__init__.py"
    if init_py.exists():
        return [str(init_py)]

    return None


def _discover_extensions_in_dir(dir_path: str) -> list[str]:
    """在目录中发现扩展。

    发现规则：
    1. 直接文件：目录中的 ``*.py`` 文件（非 ``__init__.py``）→ 加载
    2. 子目录：子目录包含 ``__init__.py`` → 加载
    3. 子目录：子目录包含 package.json 且含 pi 字段 → 加载其声明

    不递归超过一级。复杂包必须使用 package.json 清单。

    Args:
        dir_path: 目录路径。

    Returns:
        发现的扩展路径列表。
    """
    dir_path_obj = Path(dir_path)
    if not dir_path_obj.exists():
        return []

    discovered: list[str] = []

    try:
        for entry in dir_path_obj.iterdir():
            entry_path = str(entry)

            # 1. 直接 Python 文件
            if entry.is_file() and _is_extension_file(entry.name):
                discovered.append(entry_path)
                continue

            # 2 & 3. 子目录
            if entry.is_dir():
                entries = _resolve_extension_entries(entry_path)
                if entries:
                    discovered.extend(entries)
    except OSError:
        return []

    return discovered


# ---------------------------------------------------------------------------
# 发现并加载扩展
# ---------------------------------------------------------------------------


async def discover_and_load_extensions(
    configured_paths: list[str],
    cwd: str,
    agent_dir: str | None = None,
    event_bus: EventBus | None = None,
) -> dict[str, Any]:
    """从标准位置发现并加载扩展。

    Args:
        configured_paths: 显式配置的扩展路径。
        cwd: 当前工作目录。
        agent_dir: agent 配置目录（可选）。
        event_bus: 事件总线（可选）。

    Returns:
        加载结果字典。
    """
    from ..config import (  # type: ignore[import-not-found]
        CONFIG_DIR_NAME,
        get_agent_dir,
    )

    resolved_cwd = str(Path(cwd).resolve())
    resolved_agent_dir = str(
        Path(agent_dir if agent_dir else str(get_agent_dir())).resolve()
    )
    all_paths: list[str] = []
    seen: set[str] = set()

    def _add_paths(paths: list[str]) -> None:
        for p in paths:
            resolved = str(Path(p).resolve())
            if resolved not in seen:
                seen.add(resolved)
                all_paths.append(p)

    # 1. 项目本地扩展：cwd/.pi/extensions/
    local_ext_dir = str(Path(resolved_cwd) / CONFIG_DIR_NAME / "extensions")
    _add_paths(_discover_extensions_in_dir(local_ext_dir))

    # 2. 全局扩展：agent_dir/extensions/
    global_ext_dir = str(Path(resolved_agent_dir) / "extensions")
    _add_paths(_discover_extensions_in_dir(global_ext_dir))

    # 3. 显式配置路径
    for p in configured_paths:
        resolved = str(Path(p).resolve())
        if Path(resolved).exists() and Path(resolved).is_dir():
            entries = _resolve_extension_entries(resolved)
            if entries:
                _add_paths(entries)
                continue
            _add_paths(_discover_extensions_in_dir(resolved))
            continue

        _add_paths([resolved])

    return await load_extensions(all_paths, resolved_cwd, event_bus)
