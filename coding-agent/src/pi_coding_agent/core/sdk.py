"""SDK 入口：创建 AgentSession（对应 TS ``core/sdk.ts``）。

提供 ``CreateAgentSessionOptions``、``CreateAgentSessionResult`` 和
``create_agent_session`` 函数。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Protocol, cast

from pi_agent.agent import Agent
from pi_agent.agent_lifecycle import AgentOptions
from pi_agent.stream_fn import set_default_stream_fn
from pi_agent.types import (
    AgentMessage,
    AgentState,
    ThinkingLevel,
)
from pi_ai.compat import stream_simple
from pi_ai.models import clamp_thinking_level
from pi_ai.types import Message, Model, SimpleStreamOptions

from ..config import get_agent_dir, get_sessions_dir

# ---------------------------------------------------------------------------
# 尚不存在的模块导入（将在后续任务中创建）
# ---------------------------------------------------------------------------
from .agent_session import AgentSession, AgentSessionConfig
from .auth_guidance import format_no_models_available_message
from .defaults import DEFAULT_THINKING_LEVEL
from .extensions import (
    LoadExtensionsResult,
    SessionStartEvent,
    ToolDefinition,
)
from .messages import convert_to_llm
from .model_resolver import find_initial_model  # type: ignore[attr-defined]
from .model_runtime import ModelRuntime
from .provider_attribution import merge_provider_attribution_headers
from .resource_loader import DefaultResourceLoader, ResourceLoader
from .session_manager import SessionManager
from .settings_manager import SettingsManager
from .timings import time
from .tools import (
    create_bash_tool,
    create_coding_tools,
    create_edit_tool,
    create_find_tool,
    create_grep_tool,
    create_ls_tool,
    create_read_only_tools,
    create_read_tool,
    create_write_tool,
    with_file_mutation_queue,
)

# ---------------------------------------------------------------------------
# ExtensionRunner Protocol（对应 TS ``extensions/runner.ts``）
# 完整实现将在后续任务中创建
# ---------------------------------------------------------------------------


class ExtensionRunner(Protocol):
    """扩展运行器（对应 TS ``ExtensionRunner``）。"""

    async def emit_before_provider_headers(
        self, headers: dict[str, str | None]
    ) -> dict[str, str]: ...

    def has_handlers(self, event_type: str) -> bool: ...

    async def emit_context(
        self, messages: list[AgentMessage]
    ) -> list[AgentMessage]: ...


# ---------------------------------------------------------------------------
# 设置默认 streamFn（兼容旧版扩展）
# ---------------------------------------------------------------------------

# pi-agent-core 保持 provider 无关，不导入 pi-ai/compat 自身。
# 此处设置默认 streamFn，确保不提供 streamFn 的旧版扩展仍能正常工作。
set_default_stream_fn(stream_simple)

# ---------------------------------------------------------------------------
# CreateAgentSessionOptions
# ---------------------------------------------------------------------------


class CreateAgentSessionOptions:
    """创建 AgentSession 的选项（对应 TS ``CreateAgentSessionOptions``）。

    Attributes:
        cwd: 项目本地发现的工作目录。默认: ``os.getcwd()``。
        agent_dir: 全局配置目录。默认: ``~/.pi/agent``。
        model_runtime: 规范的模型/认证运行时。
        model: 要使用的模型。
        thinking_level: 思考级别。
        scoped_models: 可用于循环的模型列表。
        no_tools: 可选的默认工具抑制模式。
        tools: 可选的工具名称允许列表。
        exclude_tools: 可选的工具名称禁止列表。
        custom_tools: 要注册的自定义工具。
        resource_loader: 资源加载器。
        session_manager: 会话管理器。
        settings_manager: 设置管理器。
        session_start_event: 扩展运行时启动的会话开始事件元数据。
    """

    def __init__(
        self,
        *,
        cwd: str | None = None,
        agent_dir: str | None = None,
        model_runtime: ModelRuntime | None = None,
        model: Model | None = None,
        thinking_level: ThinkingLevel | None = None,
        scoped_models: list[dict[str, Any]] | None = None,
        no_tools: Literal["all", "builtin"] | None = None,
        tools: list[str] | None = None,
        exclude_tools: list[str] | None = None,
        custom_tools: list[ToolDefinition] | None = None,
        resource_loader: ResourceLoader | None = None,
        session_manager: SessionManager | None = None,
        settings_manager: SettingsManager | None = None,
        session_start_event: SessionStartEvent | None = None,
    ) -> None:
        self.cwd = cwd
        self.agent_dir = agent_dir
        self.model_runtime = model_runtime
        self.model = model
        self.thinking_level = thinking_level
        self.scoped_models = scoped_models
        self.no_tools = no_tools
        self.tools = tools
        self.exclude_tools = exclude_tools
        self.custom_tools = custom_tools
        self.resource_loader = resource_loader
        self.session_manager = session_manager
        self.settings_manager = settings_manager
        self.session_start_event = session_start_event


# ---------------------------------------------------------------------------
# CreateAgentSessionResult
# ---------------------------------------------------------------------------


class CreateAgentSessionResult:
    """``create_agent_session`` 的结果（对应 TS ``CreateAgentSessionResult``）。

    Attributes:
        session: 创建的会话。
        extensions_result: 扩展结果。
        model_fallback_message: 模型回退警告。
    """

    def __init__(
        self,
        *,
        session: AgentSession,
        extensions_result: LoadExtensionsResult,
        model_fallback_message: str | None = None,
    ) -> None:
        self.session = session
        self.extensions_result = extensions_result
        self.model_fallback_message = model_fallback_message


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _get_default_agent_dir() -> str:
    return str(get_agent_dir())


def _resolve_path(path_str: str) -> str:
    """解析路径字符串（展开 ~ 并标准化）。"""
    return str(Path(path_str).expanduser().resolve())


# ---------------------------------------------------------------------------
# create_agent_session
# ---------------------------------------------------------------------------


async def create_agent_session(
    options: CreateAgentSessionOptions | None = None,
) -> CreateAgentSessionResult:
    """使用指定选项创建 AgentSession（对应 TS ``createAgentSession``）。"""
    opts = options or CreateAgentSessionOptions()

    cwd = _resolve_path(
        opts.cwd
        or (opts.session_manager.get_cwd() if opts.session_manager else None)
        or str(Path.cwd())
    )
    agent_dir = (
        _resolve_path(opts.agent_dir) if opts.agent_dir else _get_default_agent_dir()
    )
    resource_loader = opts.resource_loader

    auth_path = str(Path(agent_dir) / "auth.json") if opts.agent_dir else None
    models_path = str(Path(agent_dir) / "models.json") if opts.agent_dir else None
    model_runtime = opts.model_runtime or await ModelRuntime.create(
        type(
            "_CreateModelRuntimeOptions",
            (),
            {
                "auth_path": auth_path,
                "models_path": models_path,
            },
        )(),
    )

    settings_manager = opts.settings_manager or SettingsManager.create(cwd, agent_dir)
    session_manager = opts.session_manager or SessionManager.create(
        cwd, str(get_sessions_dir())
    )

    if not resource_loader:
        resource_loader = DefaultResourceLoader(  # type: ignore[call-arg]
            cwd=cwd, agent_dir=agent_dir, settings_manager=settings_manager
        )
        await resource_loader.reload()
        time("resource_loader.reload")

    # 检查会话是否有现有数据要恢复
    existing_session = session_manager.build_session_context()
    has_existing_session = len(existing_session.messages) > 0
    has_thinking_entry = any(
        entry.type == "thinking_level_change" for entry in session_manager.get_branch()
    )

    model = opts.model
    model_fallback_message: str | None = None

    # 如果会话有数据，尝试从会话恢复模型
    if not model and has_existing_session and existing_session.model:
        restored_model = model_runtime.get_model(
            existing_session.model["provider"], existing_session.model["model_id"]
        )
        if restored_model and model_runtime.has_configured_auth(
            restored_model.provider
        ):
            model = restored_model
        if not model:
            model_fallback_message = (
                f"Could not restore model {existing_session.model['provider']}/"
                f"{existing_session.model['model_id']}"
            )

    # 如果仍然没有模型，使用 find_initial_model
    if not model:
        result = await find_initial_model(
            scoped_models=[],
            is_continuing=has_existing_session,
            default_provider=settings_manager.get_default_provider(),
            default_model_id=settings_manager.get_default_model(),
            default_thinking_level=settings_manager.get_default_thinking_level(),
            model_runtime=model_runtime,
        )
        model = result.model
        if not model:
            model_fallback_message = format_no_models_available_message()
        elif model_fallback_message:
            model_fallback_message += f". Using {model.provider}/{model.model_id}"

    thinking_level = opts.thinking_level

    # 如果会话有数据，从会话恢复思考级别
    if thinking_level is None and has_existing_session:
        thinking_level = (
            cast("ThinkingLevel | None", existing_session.thinking_level)
            if has_thinking_entry
            else (
                settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL
            )
        )

    # 回退到设置默认值
    if thinking_level is None:
        thinking_level = (
            settings_manager.get_default_thinking_level() or DEFAULT_THINKING_LEVEL
        )

    # 夹紧到模型能力
    if not model:
        thinking_level = "off"
    else:
        thinking_level = clamp_thinking_level(model, thinking_level)

    default_active_tool_names: list[str] = ["read", "bash", "edit", "write"]
    allowed_tool_names = (
        opts.tools
        if opts.tools is not None
        else ([] if opts.no_tools == "all" else None)
    )
    excluded_tool_names = opts.exclude_tools
    excluded_tool_name_set = set(excluded_tool_names) if excluded_tool_names else None
    initial_active_tool_names: list[str] = [
        name
        for name in (
            list(opts.tools)
            if opts.tools
            else ([] if opts.no_tools else default_active_tool_names)
        )
        if not (excluded_tool_name_set and name in excluded_tool_name_set)
    ]

    # 创建 convertToLlm 包装器，如果启用 block_images 则过滤图片
    def _convert_to_llm_with_block_images(
        messages: list[AgentMessage],
    ) -> list[Message]:
        converted = convert_to_llm(messages)
        if not settings_manager.get_block_images():
            return converted
        result: list[Message] = []
        for msg in converted:
            if msg.role in ("user", "toolResult"):
                content = msg.content
                if isinstance(content, list):
                    if any(
                        getattr(c, "type", None) == "image"
                        or (isinstance(c, dict) and c.get("type") == "image")
                        for c in content
                    ):
                        filtered: list[Any] = []
                        for c in content:
                            c_type = (
                                getattr(c, "type", None)
                                if not isinstance(c, dict)
                                else c.get("type")
                            )
                            if c_type == "image":
                                filtered.append(
                                    {
                                        "type": "text",
                                        "text": "Image reading is disabled.",
                                    }
                                )
                            else:
                                filtered.append(c)
                        # 去重连续的占位符文本
                        deduped: list[Any] = []
                        for i, c in enumerate(filtered):
                            if (
                                i > 0
                                and isinstance(c, dict)
                                and c.get("type") == "text"
                                and c.get("text") == "Image reading is disabled."
                                and isinstance(filtered[i - 1], dict)
                                and filtered[i - 1].get("type") == "text"
                                and filtered[i - 1].get("text")
                                == "Image reading is disabled."
                            ):
                                continue
                            deduped.append(c)
                        result.append(msg.model_copy(update={"content": deduped}))
                        continue
            result.append(msg)
        return result

    extension_runner_ref: dict[str, ExtensionRunner | None] = {"current": None}

    # 构建 AgentOptions
    agent_initial_state = AgentState(
        system_prompt="",
        model=model,
        thinking_level=thinking_level,
        tools=[],
    )

    agent_options = AgentOptions(
        initial_state=agent_initial_state,
        convert_to_llm=_convert_to_llm_with_block_images,
        stream_fn=_make_stream_fn(
            model_runtime, settings_manager, extension_runner_ref
        ),
        session_id=session_manager.get_session_id(),
        transform_context=_make_transform_context(extension_runner_ref),
        steering_mode=settings_manager.get_steering_mode(),
        follow_up_mode=settings_manager.get_follow_up_mode(),
        transport=settings_manager.get_transport(),
        max_retry_delay_ms=settings_manager.get_provider_retry_settings().max_retry_delay_ms,
    )

    agent = Agent(agent_options)

    # 恢复消息（如果会话有现有数据）
    if has_existing_session:
        agent.state.messages = list(existing_session.messages)
        if not has_thinking_entry:
            session_manager.append_thinking_level_change(thinking_level)
    else:
        if model:
            session_manager.append_model_change(model.provider, model.model_id)
        session_manager.append_thinking_level_change(thinking_level)

    session = AgentSession(
        AgentSessionConfig(
            agent=agent,
            session_manager=session_manager,
            settings_manager=settings_manager,
            cwd=cwd,
            scoped_models=opts.scoped_models,
            resource_loader=resource_loader,
            custom_tools=opts.custom_tools,
            model_runtime=model_runtime,
            initial_active_tool_names=initial_active_tool_names,
            allowed_tool_names=allowed_tool_names,
            excluded_tool_names=excluded_tool_names,
            extension_runner_ref=extension_runner_ref,
            session_start_event=opts.session_start_event,
        )
    )
    extensions_result = resource_loader.get_extensions()

    return CreateAgentSessionResult(
        session=session,
        extensions_result=extensions_result,
        model_fallback_message=model_fallback_message,
    )


# ---------------------------------------------------------------------------
# 内部辅助工厂
# ---------------------------------------------------------------------------


def _make_stream_fn(
    model_runtime: ModelRuntime,
    settings_manager: SettingsManager,
    extension_runner_ref: dict[str, ExtensionRunner | None],
) -> Any:
    """创建 stream_fn 可调用对象。"""

    async def _stream_fn(
        model: Model,
        context: Any,
        options: dict[str, Any] | None = None,
    ) -> Any:
        provider_retry_settings = settings_manager.get_provider_retry_settings()
        http_idle_timeout_ms = settings_manager.get_http_idle_timeout_ms()  # type: ignore[attr-defined]
        effective_timeout_ms = (
            2147483647 if http_idle_timeout_ms == 0 else http_idle_timeout_ms
        )
        timeout_ms = (
            provider_retry_settings.timeout_ms
            if provider_retry_settings.timeout_ms is not None
            else effective_timeout_ms
        )
        if options and options.get("timeout_ms") is not None:
            timeout_ms = options["timeout_ms"]
        websocket_connect_timeout_ms = (
            settings_manager.get_web_socket_connect_timeout_ms()  # type: ignore[attr-defined]
        )
        if options and options.get("websocket_connect_timeout_ms") is not None:
            websocket_connect_timeout_ms = options["websocket_connect_timeout_ms"]

        header_runner = extension_runner_ref.get("current")

        effective_options = dict(options or {})
        effective_options["timeout_ms"] = timeout_ms
        effective_options["websocket_connect_timeout_ms"] = websocket_connect_timeout_ms
        effective_options["max_retries"] = (
            options.get("max_retries") if options else None
        ) or provider_retry_settings.max_retries
        effective_options["max_retry_delay_ms"] = (
            options.get("max_retry_delay_ms") if options else None
        ) or provider_retry_settings.max_retry_delay_ms

        opts_for_transform = dict(options or {})

        async def _transform_headers(
            request_headers: dict[str, str | None] | None,
        ) -> dict[str, str]:
            from pi_ai.models import ModelRecord

            headers = merge_provider_attribution_headers(
                cast(ModelRecord, model),
                settings_manager,
                opts_for_transform.get("session_id"),
                request_headers,
            )
            if header_runner is not None and header_runner.has_handlers(
                "before_provider_headers"
            ):
                result = await header_runner.emit_before_provider_headers(headers or {})
                return result
            return headers or {}  # type: ignore[return-value]

        effective_options["transform_headers"] = _transform_headers

        stream_options = SimpleStreamOptions(**effective_options)
        return model_runtime.stream_simple(model, context, stream_options)

    return _stream_fn


def _make_transform_context(
    extension_runner_ref: dict[str, ExtensionRunner | None],
) -> Any:
    """创建 transform_context 回调。"""

    async def _transform_context(
        messages: list[AgentMessage],
    ) -> list[AgentMessage]:
        runner = extension_runner_ref.get("current")
        if runner is not None:
            return await runner.emit_context(messages)
        return messages

    return _transform_context


# ---------------------------------------------------------------------------
# 再导出
# ---------------------------------------------------------------------------

# 类型再导出（对应 TS 的 export type {...}）
from pi_agent.types import AgentTool as Tool

from .agent_session_runtime import (
    AgentSessionRuntime,
    CreateAgentSessionRuntimeFactory,
    CreateAgentSessionRuntimeResult,
    SessionImportFileNotFoundError,
)
from .extensions import (
    ExtensionAPI,
    ExtensionCommandContext,
    ExtensionContext,
    ExtensionFactory,
    ExtensionHandler,
    InlineExtension,
    SlashCommandInfo,
)
from .prompt_templates import PromptTemplate
from .skills import Skill
from .slash_commands import SlashCommandSource

__all__ = [
    "CreateAgentSessionOptions",
    "CreateAgentSessionResult",
    "create_agent_session",
    # 工具工厂
    "create_bash_tool",
    "create_coding_tools",
    "create_edit_tool",
    "create_find_tool",
    "create_grep_tool",
    "create_ls_tool",
    "create_read_only_tools",
    "create_read_tool",
    "create_write_tool",
    "with_file_mutation_queue",
    # 类型再导出
    "AgentSessionRuntime",
    "CreateAgentSessionRuntimeFactory",
    "CreateAgentSessionRuntimeResult",
    "SessionImportFileNotFoundError",
    "ExtensionAPI",
    "ExtensionCommandContext",
    "ExtensionContext",
    "ExtensionFactory",
    "ExtensionHandler",
    "InlineExtension",
    "SlashCommandInfo",
    "SlashCommandSource",
    "PromptTemplate",
    "Skill",
    "Tool",
]
