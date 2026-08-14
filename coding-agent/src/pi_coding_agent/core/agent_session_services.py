"""Agent session services。 

提供创建 cwd 绑定的运行时服务以及从已创建服务构建 AgentSession 的函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from pi_ai.models import ModelsRefreshOptions

from pi_coding_agent.config import get_agent_dir

from .model_runtime import CreateModelRuntimeOptions, ModelRuntime
from .session_manager import SessionManager
from .settings_manager import SettingsManager

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------


class AgentSessionRuntimeDiagnostic:
    """非致命运行时诊断。

    在创建服务或会话时收集，由应用层决定是否显示警告或中止启动。
    """

    def __init__(
        self,
        type_: Literal["info", "warning", "error"],
        message: str,
    ) -> None:
        self.type: Literal["info", "warning", "error"] = type_
        self.message: str = message


class CreateAgentSessionServicesOptions:
    """创建 cwd 绑定运行时服务的输入。

    这些服务在有效会话 cwd 变化时重建。
    CLI 提供的资源路径应在到达此函数前解析为绝对路径，
    以避免后续 cwd 切换重新解释它们。
    """

    def __init__(
        self,
        cwd: str,
        agent_dir: str | None = None,
        settings_manager: SettingsManager | None = None,
        model_runtime: ModelRuntime | None = None,
        model_runtime_signal: Any = None,
        extension_flag_values: dict[str, bool | str] | None = None,
        resource_loader_options: dict[str, Any] | None = None,
        resource_loader_reload_options: Any = None,
    ) -> None:
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.settings_manager = settings_manager
        self.model_runtime = model_runtime
        self.model_runtime_signal = model_runtime_signal
        self.extension_flag_values = extension_flag_values
        self.resource_loader_options = resource_loader_options
        self.resource_loader_reload_options = resource_loader_reload_options


class CreateAgentSessionFromServicesOptions:
    """从已创建服务构建 AgentSession 的输入。

    在服务已存在且 cwd 绑定的模型/工具/会话选项已针对这些服务解析后使用。
    """

    def __init__(
        self,
        services: AgentSessionServices,
        session_manager: SessionManager,
        session_start_event: Any = None,
        model: Any = None,
        thinking_level: Any = None,
        scoped_models: list[dict[str, Any]] | None = None,
        tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        no_tools: str | None = None,
        custom_tools: list[Any] | None = None,
    ) -> None:
        self.services = services
        self.session_manager = session_manager
        self.session_start_event = session_start_event
        self.model = model
        self.thinking_level = thinking_level
        self.scoped_models = scoped_models
        self.tools = tools
        self.exclude_tools = exclude_tools
        self.no_tools = no_tools
        self.custom_tools = custom_tools


class AgentSessionServices:
    """一个有效会话 cwd 的连贯 cwd 绑定运行时服务。

    仅基础设施。AgentSession 本身是单独创建的，
    以便会话选项可以先针对这些服务进行解析。
    """

    def __init__(
        self,
        cwd: str,
        agent_dir: str,
        model_runtime: ModelRuntime,
        settings_manager: SettingsManager,
        resource_loader: Any,
        diagnostics: list[AgentSessionRuntimeDiagnostic],
    ) -> None:
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.model_runtime = model_runtime
        self.settings_manager = settings_manager
        self.resource_loader = resource_loader
        self.diagnostics = diagnostics


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _resolve_path(path_str: str) -> str:
    """解析路径，处理 ~ 展开和相对路径。"""
    return str(Path(path_str).expanduser().resolve())


def _apply_extension_flag_values(
    resource_loader: Any,
    extension_flag_values: dict[str, bool | str] | None,
) -> list[AgentSessionRuntimeDiagnostic]:
    """应用扩展标志值。"""
    if not extension_flag_values:
        return []

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    registered_flags: dict[str, dict[str, str]] = {}
    for extension in extensions_result.extensions:
        for name, flag in extension.flags:
            registered_flags[name] = {"type": flag.type}

    unknown_flags: list[str] = []
    for name, value in extension_flag_values.items():
        flag = registered_flags.get(name)
        if not flag:
            unknown_flags.append(name)
            continue
        if flag["type"] == "boolean":
            extensions_result.runtime.flag_values[name] = True
            continue
        if isinstance(value, str):
            extensions_result.runtime.flag_values[name] = value
            continue
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type_="error",
                message=f'Extension flag "--{name}" requires a value',
            )
        )

    if unknown_flags:
        suffix = "" if len(unknown_flags) == 1 else "s"
        flag_list = ", ".join(f"--{name}" for name in unknown_flags)
        diagnostics.append(
            AgentSessionRuntimeDiagnostic(
                type_="error",
                message=f"Unknown option{suffix}: {flag_list}",
            )
        )

    return diagnostics


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


async def create_agent_session_services(
    options: CreateAgentSessionServicesOptions,
) -> AgentSessionServices:
    """创建 cwd 绑定的运行时服务。

    返回服务加诊断信息。不创建 AgentSession。
    """
    cwd = _resolve_path(options.cwd)
    agent_dir = (
        _resolve_path(options.agent_dir) if options.agent_dir else str(get_agent_dir())
    )

    model_runtime = options.model_runtime or await ModelRuntime.create(
        CreateModelRuntimeOptions(
            auth_path=str(Path(agent_dir) / "auth.json"),
            models_path=str(Path(agent_dir) / "models.json"),
            signal=options.model_runtime_signal,
        )
    )

    settings_manager = options.settings_manager or SettingsManager.create(
        cwd, agent_dir
    )

    from .resource_loader import DefaultResourceLoader, DefaultResourceLoaderOptions

    resource_loader_options: dict[str, Any] = dict(
        options.resource_loader_options or {}
    )
    resource_loader_options.update(
        {"cwd": cwd, "agent_dir": agent_dir, "settings_manager": settings_manager}
    )
    resource_loader = DefaultResourceLoader(
        DefaultResourceLoaderOptions(**resource_loader_options)
    )
    await resource_loader.reload(options.resource_loader_reload_options)

    diagnostics: list[AgentSessionRuntimeDiagnostic] = []
    extensions_result = resource_loader.get_extensions()
    for registration in extensions_result.runtime.pending_provider_registrations:  # type: ignore[attr-defined]
        try:
            model_runtime.register_provider(registration.name, registration.config)
        except Exception as error:
            message = str(error)
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type_="error",
                    message=f'Extension "{registration.extension_path}" error: {message}',
                )
            )
    extensions_result.runtime.pending_provider_registrations = []  # type: ignore[attr-defined]

    for registration in extensions_result.runtime.pending_native_provider_registrations:  # type: ignore[attr-defined]
        try:
            model_runtime.register_native_provider(registration.provider)
        except Exception as error:
            message = str(error)
            diagnostics.append(
                AgentSessionRuntimeDiagnostic(
                    type_="error",
                    message=f'Extension "{registration.extension_path}" error: {message}',
                )
            )
    extensions_result.runtime.pending_native_provider_registrations = []  # type: ignore[attr-defined]

    await model_runtime.refresh(ModelsRefreshOptions(allow_network=False))
    diagnostics.extend(
        _apply_extension_flag_values(resource_loader, options.extension_flag_values)
    )

    return AgentSessionServices(
        cwd=cwd,
        agent_dir=agent_dir,
        model_runtime=model_runtime,
        settings_manager=settings_manager,
        resource_loader=resource_loader,
        diagnostics=diagnostics,
    )


async def create_agent_session_from_services(
    options: CreateAgentSessionFromServicesOptions,
) -> Any:
    """从已创建服务构建 AgentSession。

    将会话创建与服务创建分离，以便调用方在构建会话前
    可以针对目标 cwd 解析模型、思考级别、工具和其他会话输入。
    """
    from .sdk import (
        CreateAgentSessionOptions,
        create_agent_session,
    )

    return await create_agent_session(
        CreateAgentSessionOptions(
            cwd=options.services.cwd,
            agent_dir=options.services.agent_dir,
            model_runtime=options.services.model_runtime,
            settings_manager=options.services.settings_manager,
            resource_loader=options.services.resource_loader,
            session_manager=options.session_manager,
            model=options.model,
            thinking_level=options.thinking_level,
            scoped_models=options.scoped_models,
            tools=options.tools,
            exclude_tools=options.exclude_tools,
            no_tools=cast("Literal['all', 'builtin'] | None", options.no_tools),
            custom_tools=options.custom_tools,
            session_start_event=options.session_start_event,
        )
    )
