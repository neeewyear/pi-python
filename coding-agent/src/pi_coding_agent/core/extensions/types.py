"""扩展系统类型。

扩展是 Python 模块，可以：
- 订阅 agent 生命周期事件
- 注册 LLM 可调用的工具
- 注册命令、快捷键和 CLI 标志
- 通过 UI 原语与用户交互
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Literal, Protocol, TypeAlias, runtime_checkable

from pi_agent.types import (
    AgentMessage,
    AgentToolResult,
    AgentToolUpdateCallback,
    CustomMessage,
    ThinkingLevel,
    ToolExecutionMode,
)
from pi_ai.models import Provider, RefreshModelsContext
from pi_ai.types import (
    Api,
    AssistantMessageEvent,
    Context,
    ImageContent,
    Model,
    ProviderHeaders,
    SimpleStreamOptions,
    TextContent,
    ToolResultMessage,
    Usage,
)
from pi_ai.utils.event_stream import AssistantMessageEventStream
from pydantic import BaseModel, ConfigDict, Field

from ..exec import ExecOptions, ExecResult
from ..model_registry import ModelRegistry
from ..model_resolver import ScopedModel
from ..session_manager import (
    BranchSummaryEntry,
    CompactionEntry,
    CustomEntry,
    SessionEntry,
)
from ..system_prompt import BuildSystemPromptOptions

# ============================================================================
# Re-exports
# ============================================================================

# ExecOptions, ExecResult, BuildSystemPromptOptions are re-exported via imports above.
# AgentToolResult, AgentToolUpdateCallback, ToolExecutionMode from pi_agent.types.

# ============================================================================
# 常量
# ============================================================================

# ---------------------------------------------------------------------------
# 占位 Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Component(Protocol):
    """TUI 组件。"""

    def dispose(self) -> None: ...


@runtime_checkable
class TUI(Protocol):
    """TUI 实例。"""


@runtime_checkable
class EditorTheme(Protocol):
    """编辑器主题。"""


@runtime_checkable
class EditorComponent(Protocol):
    """编辑器组件。"""  


@runtime_checkable
class AutocompleteProvider(Protocol):
    """自动补全提供者。"""


class AutocompleteItem(BaseModel):
    """自动补全条目。"""

    label: str
    insert_text: str | None = None
    detail: str | None = None


KeyId: TypeAlias = str
"""快捷键标识符。"""


@runtime_checkable
class OverlayHandle(Protocol):
    """浮层句柄。"""


class OverlayOptions(BaseModel):
    """浮层定位/尺寸选项。"""

    width: int | None = None
    height: int | None = None
    x: int | None = None
    y: int | None = None


@runtime_checkable
class Theme(Protocol):
    """主题对象。"""


@runtime_checkable
class ReadonlyFooterDataProvider(Protocol):
    """只读页脚数据提供者。"""  


@runtime_checkable
class AppKeybinding(Protocol):
    """应用级快捷键。"""


@runtime_checkable
class KeybindingsManager(Protocol):
    """快捷键管理器。"""


@runtime_checkable
class ReadonlySessionManager(Protocol):
    """只读会话管理器。"""


@runtime_checkable
class SlashCommandInfo(Protocol):
    """斜杠命令信息。"""


# ============================================================================
# UI Context
# ============================================================================


class ExtensionUIDialogOptions(BaseModel):
    """扩展 UI 对话框选项。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    signal: asyncio.Event | None = None
    """可用于编程关闭对话框的取消信号。"""
    timeout: int | None = None
    """超时毫秒数。对话框超时自动关闭，带实时倒计时显示。"""


WidgetPlacement: TypeAlias = Literal["aboveEditor", "belowEditor"]
"""扩展 widget 放置位置。"""


class ExtensionWidgetOptions(BaseModel):
    """扩展 widget 选项。"""

    placement: WidgetPlacement = "aboveEditor"
    """widget 渲染位置。默认 ``aboveEditor``。"""


TerminalInputHandler: TypeAlias = Callable[[str], dict[str, bool | str] | None]
"""原始终端输入监听器。"""


class WorkingIndicatorOptions(BaseModel):
    """工作指示器配置。"""

    frames: list[str] | None = None
    """动画帧。空数组隐藏指示器。自定义帧原样渲染。"""
    interval_ms: int | None = None
    """动画帧间隔毫秒数。"""


AutocompleteProviderFactory: TypeAlias = Callable[
    [AutocompleteProvider], AutocompleteProvider
]
"""自动补全提供者工厂。"""

EditorFactory: TypeAlias = Callable[
    [TUI, EditorTheme, KeybindingsManager], EditorComponent
]
"""编辑器工厂。"""


class ExtensionUIContext(BaseModel):
    """扩展 UI 上下文。

    每个模式（interactive、RPC、print）提供自己的实现。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 方法字段使用 Callable 类型
    select: Callable[
        [str, list[str], ExtensionUIDialogOptions | None],
        Awaitable[str | None],
    ]
    """显示选择器并返回用户选择。"""
    confirm: Callable[[str, str, ExtensionUIDialogOptions | None], Awaitable[bool]]
    """显示确认对话框。"""
    input: Callable[
        [str, str | None, ExtensionUIDialogOptions | None],
        Awaitable[str | None],
    ]
    """显示文本输入对话框。"""
    notify: Callable[[str, str | None], None]
    """向用户显示通知。"""
    on_terminal_input: Callable[[TerminalInputHandler], Callable[[], None]]
    """监听原始终端输入（仅 interactive 模式）。返回取消订阅函数。"""
    set_status: Callable[[str, str | None], None]
    """在页脚/状态栏设置状态文本。"""
    set_working_message: Callable[[str | None], None]
    """设置流式传输期间的工作/加载消息。"""
    set_working_visible: Callable[[bool], None]
    """显示或隐藏内置交互式工作加载行。"""
    set_working_indicator: Callable[[WorkingIndicatorOptions | None], None]
    """配置流式传输期间的工作指示器。"""
    set_hidden_thinking_label: Callable[[str | None], None]
    """设置隐藏思考块的标签。"""
    set_widget: Callable[[str, Any | None, ExtensionWidgetOptions | None], None]
    """在编辑器上方或下方设置 widget。"""
    set_footer: Callable[
        [
            Callable[
                [TUI, Theme, ReadonlyFooterDataProvider],
                Component,
            ]
            | None,
        ],
        None,
    ]
    """设置自定义页脚组件。"""
    set_header: Callable[
        [Callable[[TUI, Theme], Component] | None],
        None,
    ]
    """设置自定义头部组件。"""
    set_title: Callable[[str], None]
    """设置终端窗口/标签页标题。"""
    custom: Callable[
        [
            Callable[
                [TUI, Theme, KeybindingsManager, Callable[[Any], None]],
                Any,
            ],
            dict[str, Any] | None,
        ],
        Awaitable[Any],
    ]
    """显示自定义组件并获取键盘焦点。"""
    paste_to_editor: Callable[[str], None]
    """将文本粘贴到编辑器中。"""
    set_editor_text: Callable[[str], None]
    """设置核心输入编辑器的文本。"""
    get_editor_text: Callable[[], str]
    """获取核心输入编辑器的当前文本。"""
    editor: Callable[[str, str | None], Awaitable[str | None]]
    """显示多行编辑器进行文本编辑。"""
    add_autocomplete_provider: Callable[[AutocompleteProviderFactory], None]
    """在内置提供者之上堆叠额外的自动补全行为。"""
    set_editor_component: Callable[[EditorFactory | None], None]
    """通过工厂函数设置自定义编辑器组件。"""
    get_editor_component: Callable[[], EditorFactory | None]
    """获取当前配置的自定义编辑器工厂。"""
    theme: Theme
    """获取当前主题。"""
    get_all_themes: Callable[[], list[dict[str, str | None]]]
    """获取所有可用主题及其名称和文件路径。"""
    get_theme: Callable[[str], Theme | None]
    """按名称加载主题。"""
    set_theme: Callable[[str | Theme], dict[str, bool | str]]
    """按名称或 Theme 对象设置当前主题。"""
    get_tools_expanded: Callable[[], bool]
    """获取当前工具输出展开状态。"""
    set_tools_expanded: Callable[[bool], None]
    """设置工具输出展开状态。"""


# ============================================================================
# Extension Context
# ============================================================================


class ContextUsage(BaseModel):
    """上下文使用量。"""

    tokens: int | None = None
    """估计的上下文 token 数，未知时为 None。"""
    context_window: int
    """上下文窗口大小。"""
    percent: int | None = None
    """上下文使用百分比，token 未知时为 None。"""


class CompactOptions(BaseModel):
    """压缩选项。"""

    custom_instructions: str | None = None
    on_complete: Callable[[Any], None] | None = None
    on_error: Callable[[Exception], None] | None = None


ExtensionMode: TypeAlias = Literal["tui", "rpc", "json", "print"]
"""扩展运行模式。"""


class ExtensionContext(BaseModel):
    """传递给扩展事件处理器的上下文。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    ui: ExtensionUIContext
    """UI 方法。"""
    mode: ExtensionMode
    """当前运行模式。"""
    has_ui: bool
    """是否具有对话框能力的 UI（TUI 和 RPC 模式为 True）。"""
    cwd: str
    """当前工作目录。"""
    session_manager: ReadonlySessionManager
    """会话管理器（只读）。"""
    model_registry: ModelRegistry
    """模型注册表。"""
    model: Model | None
    """当前模型（可能为 None）。"""
    scoped_models: list[ScopedModel] | None = None
    """限定到当前会话的模型列表。"""
    thinking_level: ThinkingLevel | None = None
    """当前思考级别。"""
    is_idle: Callable[[], bool]
    """agent 是否空闲（未流式传输）。"""
    is_project_trusted: Callable[[], bool]
    """项目本地信任是否激活。"""
    signal: asyncio.Event | None = None
    """当前取消信号，agent 未流式传输时为 None。"""
    abort: Callable[[], None]
    """中止当前 agent 操作。"""
    has_pending_messages: Callable[[], bool]
    """是否有队列消息等待处理。"""
    shutdown: Callable[[], None]
    """优雅关闭 pi 并退出。"""
    get_context_usage: Callable[[], ContextUsage | None]
    """获取当前模型下的上下文使用情况。"""
    compact: Callable[[CompactOptions | None], None]
    """触发压缩，不等待完成。"""
    get_system_prompt: Callable[[], str]
    """获取当前生效的系统提示词。"""


class ExtensionCommandContext(ExtensionContext):
    """命令处理器的扩展上下文。"""

    get_system_prompt_options: Callable[[], BuildSystemPromptOptions]
    """获取当前基础系统提示词构建选项。"""
    wait_for_idle: Callable[[], Awaitable[None]]
    """等待 agent 完成流式传输。"""
    new_session: Callable[
        [Any | None],
        Awaitable[dict[str, bool]],
    ]
    """启动新会话。"""
    fork: Callable[
        [str, Any | None],
        Awaitable[dict[str, bool]],
    ]
    """从指定条目 fork 出新会话。"""
    navigate_tree: Callable[
        [str, Any | None],
        Awaitable[dict[str, bool]],
    ]
    """导航到会话树中的不同节点。"""
    switch_session: Callable[
        [str, Any | None],
        Awaitable[dict[str, bool]],
    ]
    """切换到不同的会话文件。"""
    reload: Callable[[], Awaitable[None]]
    """重新加载扩展、技能、提示词、主题和上下文文件。"""


class ReplacedSessionContext(ExtensionCommandContext):
    """替换会话后的命令上下文。"""

    send_message: Callable[
        [
            dict[str, Any],
            dict[str, Any] | None,
        ],
        Awaitable[None],
    ]
    """发送自定义消息到会话。"""
    send_user_message: Callable[
        [
            str | list[TextContent | ImageContent],
            dict[str, Any] | None,
        ],
        Awaitable[None],
    ]
    """发送用户消息。"""


# ============================================================================
# Tool Types
# ============================================================================


class ToolRenderResultOptions(BaseModel):
    """工具结果渲染选项。"""

    expanded: bool
    """结果视图是否展开。"""
    is_partial: bool
    """是否为部分/流式结果。"""


class ToolRenderContext(BaseModel):
    """传递给工具渲染器的上下文。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    args: Any
    """当前工具调用参数。"""
    tool_call_id: str
    """此工具执行的唯一 ID。"""
    invalidate: Callable[[], None]
    """使此工具执行组件失效以进行重绘。"""
    last_component: Component | None = None
    """此前为此渲染槽返回的组件。"""
    state: Any
    """此工具行的共享渲染器状态。"""
    cwd: str
    """此工具执行的工作目录。"""
    execution_started: bool
    """工具执行是否已开始。"""
    args_complete: bool
    """工具调用参数是否完整。"""
    is_partial: bool
    """工具结果是否为部分/流式。"""
    expanded: bool
    """结果视图是否展开。"""
    show_images: bool
    """TUI 中是否当前显示内联图片。"""
    is_error: bool
    """当前结果是否为错误。"""


class ToolDefinition(BaseModel):
    """工具定义。

    注意：TS 版本使用 TypeBox 的 ``TSchema``/``Static`` 进行参数类型推断。
    Python 版本使用 ``dict[str, object]`` 作为参数 schema。
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    """工具名称（用于 LLM 工具调用）。"""
    label: str
    """人类可读的 UI 标签。"""
    description: str
    """LLM 描述。"""
    prompt_snippet: str | None = None
    """可选的单行代码片段。"""
    prompt_guidelines: list[str] | None = None
    """可选的自定义准则。"""
    parameters: dict[str, object]
    """参数 schema。"""
    constrained_sampling: Any | None = None
    """可选的提供者端受限采样配置。"""
    render_shell: Literal["default", "self"] | None = None
    """控制渲染外壳。"""
    prepare_arguments: Callable[[dict[str, object]], dict[str, object]] | None = None
    """可选的参数兼容性转换。"""
    execution_mode: ToolExecutionMode | None = None
    """每工具执行模式覆盖。"""
    execute: Callable[
        [
            str,
            dict[str, object],
            asyncio.Event | None,
            AgentToolUpdateCallback | None,
            ExtensionContext,
        ],
        Awaitable[AgentToolResult],
    ]
    """执行工具。"""
    render_call: Callable[..., Component] | None = None
    """自定义工具调用渲染。"""
    render_result: Callable[..., Component] | None = None
    """自定义工具结果渲染。"""


AnyToolDefinition: TypeAlias = ToolDefinition
"""任意工具定义。"""


def define_tool(tool: ToolDefinition) -> ToolDefinition:
    """定义工具，保留参数推断。"""
    return tool


# ============================================================================
# Startup/Resource Events
# ============================================================================


class ProjectTrustEvent(BaseModel):
    """项目信任事件。"""

    type: Literal["project_trust"] = "project_trust"
    cwd: str


ProjectTrustEventDecision: TypeAlias = Literal["yes", "no", "undecided"]
"""项目信任决策。"""


class ProjectTrustEventResult(BaseModel):
    """项目信任事件结果。"""

    trusted: ProjectTrustEventDecision
    remember: bool | None = None


class ProjectTrustContext(BaseModel):
    """项目信任上下文。"""

    cwd: str
    mode: ExtensionMode
    has_ui: bool
    ui: ExtensionUIContext | None = None


ProjectTrustHandler: TypeAlias = Callable[
    [ProjectTrustEvent, ProjectTrustContext],
    Awaitable[ProjectTrustEventResult] | ProjectTrustEventResult,
]
"""项目信任处理器。"""


class ResourcesDiscoverEvent(BaseModel):
    """资源发现事件。"""

    type: Literal["resources_discover"] = "resources_discover"
    cwd: str
    reason: Literal["startup", "reload"]


class ResourcesDiscoverResult(BaseModel):
    """资源发现事件结果。"""

    skill_paths: list[str] | None = None
    prompt_paths: list[str] | None = None
    theme_paths: list[str] | None = None


# ============================================================================
# Session Events
# ============================================================================


class SessionStartEvent(BaseModel):
    """会话启动事件。"""

    type: Literal["session_start"] = "session_start"
    reason: Literal["startup", "reload", "new", "resume", "fork"]
    previous_session_file: str | None = None


class SessionInfoChangedEvent(BaseModel):
    """会话信息变更事件。"""

    type: Literal["session_info_changed"] = "session_info_changed"
    name: str | None = None


class SessionBeforeSwitchEvent(BaseModel):
    """会话切换前事件。"""

    type: Literal["session_before_switch"] = "session_before_switch"
    reason: Literal["new", "resume"]
    target_session_file: str | None = None


class SessionBeforeForkEvent(BaseModel):
    """会话 fork 前事件。"""

    type: Literal["session_before_fork"] = "session_before_fork"
    entry_id: str
    position: Literal["before", "at"]


class SessionBeforeCompactEvent(BaseModel):
    """会话压缩前事件。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["session_before_compact"] = "session_before_compact"
    preparation: Any
    """CompactionPreparation。"""
    branch_entries: list[SessionEntry]
    custom_instructions: str | None = None
    reason: Literal["manual", "threshold", "overflow"]
    will_retry: bool
    signal: asyncio.Event | None = None


class SessionCompactEvent(BaseModel):
    """会话压缩事件。"""

    type: Literal["session_compact"] = "session_compact"
    compaction_entry: CompactionEntry[object]
    from_extension: bool
    reason: Literal["manual", "threshold", "overflow"]
    will_retry: bool


class SessionShutdownEvent(BaseModel):
    """会话关闭事件。"""

    type: Literal["session_shutdown"] = "session_shutdown"
    reason: Literal["quit", "reload", "new", "resume", "fork"]
    target_session_file: str | None = None


class TreePreparation(BaseModel):
    """树导航准备数据。"""  

    target_id: str
    old_leaf_id: str | None = None
    common_ancestor_id: str | None = None
    entries_to_summarize: list[SessionEntry]
    user_wants_summary: bool
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


class SessionBeforeTreeEvent(BaseModel):
    """树导航前事件。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["session_before_tree"] = "session_before_tree"
    preparation: TreePreparation
    signal: asyncio.Event | None = None


class SessionTreeEvent(BaseModel):
    """树导航后事件。"""

    type: Literal["session_tree"] = "session_tree"
    new_leaf_id: str | None = None
    old_leaf_id: str | None = None
    summary_entry: BranchSummaryEntry[object] | None = None
    from_extension: bool | None = None


SessionEvent: TypeAlias = (
    SessionStartEvent
    | SessionInfoChangedEvent
    | SessionBeforeSwitchEvent
    | SessionBeforeForkEvent
    | SessionBeforeCompactEvent
    | SessionCompactEvent
    | SessionShutdownEvent
    | SessionBeforeTreeEvent
    | SessionTreeEvent
)
"""会话事件联合。"""


# ============================================================================
# Agent Events
# ============================================================================


class ContextEvent(BaseModel):
    """上下文事件。"""

    type: Literal["context"] = "context"
    messages: list[AgentMessage]


class BeforeProviderRequestEvent(BaseModel):
    """提供者请求前事件。"""

    type: Literal["before_provider_request"] = "before_provider_request"
    payload: Any


class BeforeProviderHeadersEvent(BaseModel):
    """提供者请求头前事件。"""

    type: Literal["before_provider_headers"] = "before_provider_headers"
    headers: ProviderHeaders


class AfterProviderResponseEvent(BaseModel):
    """提供者响应后事件。"""

    type: Literal["after_provider_response"] = "after_provider_response"
    status: int
    headers: dict[str, str]


class BeforeAgentStartEvent(BaseModel):
    """Agent 启动前事件。"""

    type: Literal["before_agent_start"] = "before_agent_start"
    prompt: str
    images: list[ImageContent] | None = None
    system_prompt: str
    system_prompt_options: BuildSystemPromptOptions


class AgentStartEvent(BaseModel):
    """Agent 启动事件。"""

    type: Literal["agent_start"] = "agent_start"


class AgentEndEvent(BaseModel):
    """Agent 结束事件。"""

    type: Literal["agent_end"] = "agent_end"
    messages: list[AgentMessage]


class AgentSettledEvent(BaseModel):
    """Agent 稳定事件。"""

    type: Literal["agent_settled"] = "agent_settled"


class TurnStartEvent(BaseModel):
    """回合开始事件。"""

    type: Literal["turn_start"] = "turn_start"
    turn_index: int
    timestamp: int


class TurnEndEvent(BaseModel):
    """回合结束事件。"""

    type: Literal["turn_end"] = "turn_end"
    turn_index: int
    message: AgentMessage
    tool_results: list[ToolResultMessage]


class MessageStartEvent(BaseModel):
    """消息开始事件。"""

    type: Literal["message_start"] = "message_start"
    message: AgentMessage


class MessageUpdateEvent(BaseModel):
    """消息更新事件。"""

    type: Literal["message_update"] = "message_update"
    message: AgentMessage
    assistant_message_event: AssistantMessageEvent


class MessageEndEvent(BaseModel):
    """消息结束事件。"""    

    type: Literal["message_end"] = "message_end"
    message: AgentMessage


class ToolExecutionStartEvent(BaseModel):
    """工具执行开始事件。"""

    type: Literal["tool_execution_start"] = "tool_execution_start"
    tool_call_id: str
    tool_name: str
    args: Any


class ToolExecutionUpdateEvent(BaseModel):
    """工具执行更新事件。"""

    type: Literal["tool_execution_update"] = "tool_execution_update"
    tool_call_id: str
    tool_name: str
    args: Any
    partial_result: Any


class ToolExecutionEndEvent(BaseModel):
    """工具执行结束事件。"""

    type: Literal["tool_execution_end"] = "tool_execution_end"
    tool_call_id: str
    tool_name: str
    result: Any
    is_error: bool


# ============================================================================
# Model Events
# ============================================================================


ModelSelectSource: TypeAlias = Literal["set", "cycle", "restore"]
"""模型选择来源。"""


class ModelSelectEvent(BaseModel):
    """模型选择事件。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    type: Literal["model_select"] = "model_select"
    model: Model
    previous_model: Model | None = None
    source: ModelSelectSource


class ThinkingLevelSelectEvent(BaseModel):
    """思考级别选择事件。"""    

    type: Literal["thinking_level_select"] = "thinking_level_select"
    level: ThinkingLevel
    previous_level: ThinkingLevel


# ============================================================================
# User Bash Events
# ============================================================================


class UserBashEvent(BaseModel):
    """用户 Bash 事件。"""

    type: Literal["user_bash"] = "user_bash"
    command: str
    exclude_from_context: bool
    cwd: str


# ============================================================================
# Input Events
# ============================================================================


InputSource: TypeAlias = Literal["interactive", "rpc", "extension"]
"""输入来源。"""


class InputEvent(BaseModel):
    """输入事件。"""

    type: Literal["input"] = "input"
    text: str
    images: list[ImageContent] | None = None
    source: InputSource
    streaming_behavior: Literal["steer", "followUp"] | None = None


class InputContinueResult(BaseModel):
    """继续处理输入事件结果。"""

    action: Literal["continue"] = "continue"


class InputTransformResult(BaseModel):
    """转换输入事件结果。"""

    action: Literal["transform"] = "transform"
    text: str
    images: list[ImageContent] | None = None


class InputHandledResult(BaseModel):
    """已处理输入事件结果。"""

    action: Literal["handled"] = "handled"


InputEventResult: TypeAlias = Annotated[
    InputContinueResult | InputTransformResult | InputHandledResult,
    Field(discriminator="action"),
]
"""输入事件结果。"""


# ============================================================================
# Tool Events
# ============================================================================


class ToolCallEventBase(BaseModel):
    """工具调用事件基类。"""

    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str


class BashToolCallEvent(ToolCallEventBase):
    """Bash 工具调用事件。"""

    tool_name: Literal["bash"] = "bash"
    input: Any


class ReadToolCallEvent(ToolCallEventBase):
    """Read 工具调用事件。"""

    tool_name: Literal["read"] = "read"
    input: Any


class EditToolCallEvent(ToolCallEventBase):
    """Edit 工具调用事件。"""

    tool_name: Literal["edit"] = "edit"
    input: Any


class WriteToolCallEvent(ToolCallEventBase):
    """Write 工具调用事件。"""

    tool_name: Literal["write"] = "write"
    input: Any


class GrepToolCallEvent(ToolCallEventBase):
    """Grep 工具调用事件。"""

    tool_name: Literal["grep"] = "grep"
    input: Any


class FindToolCallEvent(ToolCallEventBase):
    """Find 工具调用事件。"""   

    tool_name: Literal["find"] = "find"
    input: Any


class LsToolCallEvent(ToolCallEventBase):
    """Ls 工具调用事件。"""

    tool_name: Literal["ls"] = "ls"
    input: Any


class CustomToolCallEvent(ToolCallEventBase):
    """自定义工具调用事件。"""

    tool_name: str
    input: dict[str, Any]


ToolCallEvent: TypeAlias = (
    BashToolCallEvent
    | ReadToolCallEvent
    | EditToolCallEvent
    | WriteToolCallEvent
    | GrepToolCallEvent
    | FindToolCallEvent
    | LsToolCallEvent
    | CustomToolCallEvent
)
"""工具调用事件联合。"""


class ToolResultEventBase(BaseModel):
    """工具结果事件基类。"""

    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    input: dict[str, Any]
    content: list[TextContent | ImageContent]
    is_error: bool
    usage: Usage | None = None


class BashToolResultEvent(ToolResultEventBase):
    """Bash 工具结果事件。"""

    tool_name: Literal["bash"] = "bash"
    details: Any = None


class ReadToolResultEvent(ToolResultEventBase):
    """Read 工具结果事件。"""

    tool_name: Literal["read"] = "read"
    details: Any = None


class EditToolResultEvent(ToolResultEventBase):
    """Edit 工具结果事件。"""   

    tool_name: Literal["edit"] = "edit"
    details: Any = None


class WriteToolResultEvent(ToolResultEventBase):
    """Write 工具结果事件。"""

    tool_name: Literal["write"] = "write"
    details: None = None


class GrepToolResultEvent(ToolResultEventBase):
    """Grep 工具结果事件。"""

    tool_name: Literal["grep"] = "grep"
    details: Any = None


class FindToolResultEvent(ToolResultEventBase):
    """Find 工具结果事件。"""

    tool_name: Literal["find"] = "find"
    details: Any = None


class LsToolResultEvent(ToolResultEventBase):
    """Ls 工具结果事件。"""

    tool_name: Literal["ls"] = "ls"
    details: Any = None


class CustomToolResultEvent(ToolResultEventBase):
    """自定义工具结果事件。"""

    tool_name: str
    details: Any = None


ToolResultEvent: TypeAlias = (
    BashToolResultEvent
    | ReadToolResultEvent
    | EditToolResultEvent
    | WriteToolResultEvent
    | GrepToolResultEvent
    | FindToolResultEvent
    | LsToolResultEvent
    | CustomToolResultEvent
)
"""工具结果事件联合。"""


def is_bash_tool_result(e: ToolResultEvent) -> bool:
    """Bash 工具结果类型守卫。"""
    return e.tool_name == "bash"


def is_read_tool_result(e: ToolResultEvent) -> bool:
    """Read 工具结果类型守卫。"""
    return e.tool_name == "read"


def is_edit_tool_result(e: ToolResultEvent) -> bool:
    """Edit 工具结果类型守卫。"""
    return e.tool_name == "edit"


def is_write_tool_result(e: ToolResultEvent) -> bool:
    """Write 工具结果类型守卫。"""
    return e.tool_name == "write"


def is_grep_tool_result(e: ToolResultEvent) -> bool:
    """Grep 工具结果类型守卫。"""
    return e.tool_name == "grep"


def is_find_tool_result(e: ToolResultEvent) -> bool:
    """Find 工具结果类型守卫。"""
    return e.tool_name == "find"


def is_ls_tool_result(e: ToolResultEvent) -> bool:
    """Ls 工具结果类型守卫。"""
    return e.tool_name == "ls"


def is_tool_call_event_type(tool_name: str, event: ToolCallEvent) -> bool:
    """工具调用事件类型守卫。"""
    return event.tool_name == tool_name


# ============================================================================
# Extension Event Union
# ============================================================================

ExtensionEvent: TypeAlias = (
    ProjectTrustEvent
    | ResourcesDiscoverEvent
    | SessionEvent
    | ContextEvent
    | BeforeProviderRequestEvent
    | BeforeProviderHeadersEvent
    | AfterProviderResponseEvent
    | BeforeAgentStartEvent
    | AgentStartEvent
    | AgentEndEvent
    | AgentSettledEvent
    | TurnStartEvent
    | TurnEndEvent
    | MessageStartEvent
    | MessageUpdateEvent
    | MessageEndEvent
    | ToolExecutionStartEvent
    | ToolExecutionUpdateEvent
    | ToolExecutionEndEvent
    | ModelSelectEvent
    | ThinkingLevelSelectEvent
    | UserBashEvent
    | InputEvent
    | ToolCallEvent
    | ToolResultEvent
)
"""扩展事件联合。"""


# ============================================================================
# Event Results
# ============================================================================


class ContextEventResult(BaseModel):
    """上下文事件结果。"""

    messages: list[AgentMessage] | None = None


BeforeProviderRequestEventResult: TypeAlias = Any
"""提供者请求前事件结果。"""


class ToolCallEventResult(BaseModel):
    """工具调用事件结果。"""

    block: bool | None = None
    reason: str | None = None


class UserBashEventResult(BaseModel):
    """用户 Bash 事件结果。"""

    operations: Any | None = None
    """BashOperations。"""
    result: Any | None = None
    """BashResult。"""


class ToolResultEventResult(BaseModel):
    """工具结果事件结果。"""

    content: list[TextContent | ImageContent] | None = None
    details: Any = None
    is_error: bool | None = None
    usage: Usage | None = None


class MessageEndEventResult(BaseModel):
    """消息结束事件结果。"""

    message: AgentMessage | None = None


class BeforeAgentStartEventResult(BaseModel):
    """Agent 启动前事件结果。"""

    message: dict[str, Any] | None = None
    system_prompt: str | None = None


class SessionBeforeSwitchResult(BaseModel):
    """会话切换前结果。"""

    cancel: bool | None = None


class SessionBeforeForkResult(BaseModel):
    """会话 fork 前结果。"""

    cancel: bool | None = None
    skip_conversation_restore: bool | None = None


class SessionBeforeCompactResult(BaseModel):
    """会话压缩前结果。"""

    cancel: bool | None = None
    compaction: Any | None = None
    """CompactionResult。"""


class SessionBeforeTreeResult(BaseModel):
    """树导航前结果。"""

    cancel: bool | None = None
    summary: dict[str, Any] | None = None
    custom_instructions: str | None = None
    replace_instructions: bool | None = None
    label: str | None = None


# ============================================================================
# Message and Entry Rendering
# ============================================================================


class MessageRenderOptions(BaseModel):
    """消息渲染选项。"""

    expanded: bool
    output_pad: int


class MarkdownTransformContext(BaseModel):
    """Markdown 转换上下文。"""

    message_type: Literal["user", "assistant", "assistant-thinking"]
    is_streaming: bool
    available_width: int


MarkdownTransformer: TypeAlias = Callable[[str, MarkdownTransformContext], str]
"""Markdown 转换器。"""


class EntryRenderOptions(BaseModel):
    """条目渲染选项。"""

    expanded: bool


MessageRenderer: TypeAlias = Callable[
    [CustomMessage, MessageRenderOptions, Theme], Component | None
]
"""消息渲染器。"""

EntryRenderer: TypeAlias = Callable[
    [CustomEntry[object], EntryRenderOptions, Theme], Component | None
]
"""条目渲染器。"""


# ============================================================================
# Command Registration
# ============================================================================


class RegisteredCommand(BaseModel):
    """已注册命令。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    source_info: Any
    """SourceInfo。"""
    description: str | None = None
    get_argument_completions: Callable[[str], Awaitable[Any] | Any] | None = None
    handler: Callable[[str, ExtensionCommandContext], Awaitable[None]]


class ResolvedCommand(RegisteredCommand):
    """已解析命令。"""  

    invocation_name: str


# ============================================================================
# Extension API
# ============================================================================


ExtensionHandler: TypeAlias = Callable[[Any, ExtensionContext], Awaitable[Any] | Any]
"""扩展事件处理器。"""


@runtime_checkable
class ExtensionAPI(Protocol):
    """扩展 API 接口。

    扩展工厂函数接收此接口的实例。
    """

    # Event Subscription
    def on(self, event: str, handler: ExtensionHandler) -> None: ...

    # Tool Registration
    def register_tool(self, tool: ToolDefinition) -> None: ...

    # Command, Shortcut, Flag Registration
    def register_command(self, name: str, options: dict[str, Any]) -> None: ...

    def register_shortcut(
        self,
        shortcut: KeyId,
        options: dict[str, Any],
    ) -> None: ...

    def register_flag(
        self,
        name: str,
        options: dict[str, Any],
    ) -> None: ...

    def get_flag(self, name: str) -> bool | str | None: ...

    # Message Rendering
    def register_message_renderer(
        self, custom_type: str, renderer: MessageRenderer
    ) -> None: ...

    def register_markdown_transformer(
        self, transformer: MarkdownTransformer
    ) -> None: ...

    def register_entry_renderer(
        self, custom_type: str, renderer: EntryRenderer
    ) -> None: ...

    # Actions
    def send_message(
        self,
        message: dict[str, Any],
        options: dict[str, Any] | None = None,
    ) -> None: ...

    def send_user_message(
        self,
        content: str | list[TextContent | ImageContent],
        options: dict[str, Any] | None = None,
    ) -> None: ...

    def append_entry(self, custom_type: str, data: Any = None) -> None: ...

    # Session Metadata
    def set_session_name(self, name: str) -> None: ...

    def get_session_name(self) -> str | None: ...

    def set_label(self, entry_id: str, label: str | None) -> None: ...

    def exec(
        self,
        command: str,
        args: list[str],
        options: ExecOptions | None = None,
    ) -> Awaitable[ExecResult]: ...

    def get_active_tools(self) -> list[str]: ...

    def get_all_tools(self) -> list[Any]: ...

    def set_active_tools(self, tool_names: list[str]) -> None: ...

    def get_commands(self) -> list[Any]: ...

    # Model and Thinking Level
    def set_model(self, model: Model) -> Awaitable[bool]: ...

    def get_thinking_level(self) -> ThinkingLevel: ...

    def set_thinking_level(self, level: ThinkingLevel) -> None: ...

    # Provider Registration
    def register_provider(
        self, name_or_provider: str | Provider, config: Any = None
    ) -> None: ...

    def unregister_provider(self, name: str) -> None: ...

    # Event Bus
    @property
    def events(self) -> Any: ...


# ============================================================================
# Provider Registration Types
# ============================================================================


class ProviderConfig(BaseModel):
    """Provider 配置。"""   

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    api: Api | None = None
    stream_simple: (
        Callable[
            [Model, Context, SimpleStreamOptions | None],
            AssistantMessageEventStream,
        ]
        | None
    ) = None
    headers: dict[str, str] | None = None
    auth_header: bool | None = None
    models: list[Any] | None = None
    """ProviderModelConfig 列表。"""
    refresh_models: Callable[[RefreshModelsContext], Awaitable[list[Any]]] | None = None
    oauth: dict[str, Any] | None = None


class ProviderModelConfig(BaseModel):
    """Provider 模型配置。"""   

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str
    name: str
    api: Api | None = None
    base_url: str | None = None
    reasoning: bool
    thinking_level_map: Any = None
    input: list[Literal["text", "image"]]
    cost: Any
    """ModelCost。"""
    context_window: int
    max_tokens: int
    headers: dict[str, str] | None = None
    compat: Any = None


ExtensionFactory: TypeAlias = Callable[[ExtensionAPI], Awaitable[None] | None]
"""扩展工厂函数。"""


class InlineExtension(BaseModel):
    """内联扩展。"""

    name: str | None = None
    factory: ExtensionFactory | None = None
    hidden: bool | None = None


# ============================================================================
# Loaded Extension Types
# ============================================================================


class RegisteredTool(BaseModel):
    """已注册工具。"""

    definition: ToolDefinition
    source_info: Any
    """SourceInfo。"""


class ExtensionFlag(BaseModel):
    """扩展标志。"""

    name: str
    description: str | None = None
    type: Literal["boolean", "string"]
    default: bool | str | None = None
    extension_path: str


class ExtensionShortcut(BaseModel):
    """扩展快捷键。"""

    shortcut: KeyId
    description: str | None = None
    handler: Callable[[ExtensionContext], Awaitable[None] | None]
    extension_path: str


HandlerFn: TypeAlias = Callable[..., Awaitable[Any]]
"""处理器函数。"""

SendMessageHandler: TypeAlias = Callable[[dict[str, Any], dict[str, Any] | None], None]
"""发送消息处理器。"""

SendUserMessageHandler: TypeAlias = Callable[
    [str | list[TextContent | ImageContent], dict[str, Any] | None],
    None,
]
"""发送用户消息处理器。"""

AppendEntryHandler: TypeAlias = Callable[[str, Any], None]
"""追加条目处理器。"""

SetSessionNameHandler: TypeAlias = Callable[[str], None]
"""设置会话名称处理器。"""

GetSessionNameHandler: TypeAlias = Callable[[], str | None]
"""获取会话名称处理器。"""  

GetActiveToolsHandler: TypeAlias = Callable[[], list[str]]
"""获取活跃工具处理器。"""


class ToolInfo(BaseModel):
    """工具信息。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, object]
    prompt_guidelines: list[str] | None = None
    source_info: Any
    """SourceInfo。"""


GetAllToolsHandler: TypeAlias = Callable[[], list[ToolInfo]]
"""获取所有工具处理器。"""

GetCommandsHandler: TypeAlias = Callable[[], list[Any]]
"""获取命令处理器。"""

SetActiveToolsHandler: TypeAlias = Callable[[list[str]], None]
"""设置活跃工具处理器。"""

RefreshToolsHandler: TypeAlias = Callable[[], None]
"""刷新工具处理器。"""

SetModelHandler: TypeAlias = Callable[[Model], Awaitable[bool]]
"""设置模型处理器。"""

GetThinkingLevelHandler: TypeAlias = Callable[[], ThinkingLevel]
"""获取思考级别处理器。"""

SetThinkingLevelHandler: TypeAlias = Callable[[ThinkingLevel], None]
"""设置思考级别处理器。"""

SetLabelHandler: TypeAlias = Callable[[str, str | None], None]
"""设置标签处理器。"""


class ExtensionRuntimeState(BaseModel):
    """扩展运行时状态。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    flag_values: dict[str, bool | str]
    pending_provider_registrations: list[dict[str, Any]]
    pending_native_provider_registrations: list[dict[str, Any]]
    assert_active: Callable[[], None]
    invalidate: Callable[[str | None], None]
    register_provider: Callable[..., None]
    register_native_provider: Callable[..., None]
    unregister_provider: Callable[..., None]


class ExtensionActions(BaseModel):
    """扩展操作。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    send_message: SendMessageHandler
    send_user_message: SendUserMessageHandler
    append_entry: AppendEntryHandler
    set_session_name: SetSessionNameHandler
    get_session_name: GetSessionNameHandler
    set_label: SetLabelHandler
    get_active_tools: GetActiveToolsHandler
    get_all_tools: GetAllToolsHandler
    set_active_tools: SetActiveToolsHandler
    refresh_tools: RefreshToolsHandler
    get_commands: GetCommandsHandler
    set_model: SetModelHandler
    get_thinking_level: GetThinkingLevelHandler
    set_thinking_level: SetThinkingLevelHandler


class ExtensionContextActions(BaseModel):
    """扩展上下文操作。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    get_model: Callable[[], Model | None]
    get_scoped_models: Callable[[], list[Any]]
    is_idle: Callable[[], bool]
    is_project_trusted: Callable[[], bool]
    get_signal: Callable[[], asyncio.Event | None]
    abort: Callable[[], None]
    has_pending_messages: Callable[[], bool]
    shutdown: Callable[[], None]
    get_context_usage: Callable[[], ContextUsage | None]
    compact: Callable[[CompactOptions | None], None]
    get_system_prompt: Callable[[], str]
    get_system_prompt_options: Callable[[], BuildSystemPromptOptions] | None = None


class ExtensionCommandContextActions(BaseModel):
    """扩展命令上下文操作。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    wait_for_idle: Callable[[], Awaitable[None]]
    new_session: Callable[..., Awaitable[dict[str, bool]]]
    fork: Callable[..., Awaitable[dict[str, bool]]]
    navigate_tree: Callable[..., Awaitable[dict[str, bool]]]
    switch_session: Callable[..., Awaitable[dict[str, bool]]]
    reload: Callable[[], Awaitable[None]]


class ExtensionRuntime(BaseModel):
    """完整运行时 = 状态 + 操作。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Extension(BaseModel):
    """已加载扩展。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: str
    resolved_path: str
    hidden: bool | None = None
    source_info: Any
    """SourceInfo。"""
    handlers: dict[str, list[HandlerFn]]
    tools: dict[str, RegisteredTool]
    message_renderers: dict[str, MessageRenderer]
    markdown_transformer: MarkdownTransformer | None = None
    entry_renderers: dict[str, EntryRenderer] | None = None
    commands: dict[str, RegisteredCommand]
    flags: dict[str, ExtensionFlag]
    shortcuts: dict[str, ExtensionShortcut]


class LoadExtensionsResult(BaseModel):
    """扩展加载结果。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    extensions: list[Extension]
    errors: list[dict[str, str]]
    runtime: Any  # _ExtensionRuntimeImpl 实例，非 Pydantic 模型


# ============================================================================
# Extension Error
# ============================================================================


class ExtensionError(BaseModel):
    """扩展错误。"""

    extension_path: str
    event: str
    error: str
    stack: str | None = None


# ============================================================================
# __all__
# ============================================================================

__all__ = [
    # Re-exports
    "ExecOptions",
    "ExecResult",
    "BuildSystemPromptOptions",
    "AgentToolResult",
    "AgentToolUpdateCallback",
    "ToolExecutionMode",
    # Protocols
    "Component",
    "TUI",
    "EditorTheme",
    "EditorComponent",
    "AutocompleteProvider",
    "AutocompleteItem",
    "KeyId",
    "OverlayHandle",
    "OverlayOptions",
    "Theme",
    "ReadonlyFooterDataProvider",
    "AppKeybinding",
    "KeybindingsManager",
    "ReadonlySessionManager",
    "SlashCommandInfo",
    # UI Context
    "ExtensionUIDialogOptions",
    "WidgetPlacement",
    "ExtensionWidgetOptions",
    "TerminalInputHandler",
    "WorkingIndicatorOptions",
    "AutocompleteProviderFactory",
    "EditorFactory",
    "ExtensionUIContext",
    # Extension Context
    "ContextUsage",
    "CompactOptions",
    "ExtensionMode",
    "ExtensionContext",
    "ExtensionCommandContext",
    "ReplacedSessionContext",
    # Tool Types
    "ToolRenderResultOptions",
    "ToolRenderContext",
    "ToolDefinition",
    "AnyToolDefinition",
    "define_tool",
    # Startup/Resource Events
    "ProjectTrustEvent",
    "ProjectTrustEventDecision",
    "ProjectTrustEventResult",
    "ProjectTrustContext",
    "ProjectTrustHandler",
    "ResourcesDiscoverEvent",
    "ResourcesDiscoverResult",
    # Session Events
    "SessionStartEvent",
    "SessionInfoChangedEvent",
    "SessionBeforeSwitchEvent",
    "SessionBeforeForkEvent",
    "SessionBeforeCompactEvent",
    "SessionCompactEvent",
    "SessionShutdownEvent",
    "TreePreparation",
    "SessionBeforeTreeEvent",
    "SessionTreeEvent",
    "SessionEvent",
    # Agent Events
    "ContextEvent",
    "BeforeProviderRequestEvent",
    "BeforeProviderHeadersEvent",
    "AfterProviderResponseEvent",
    "BeforeAgentStartEvent",
    "AgentStartEvent",
    "AgentEndEvent",
    "AgentSettledEvent",
    "TurnStartEvent",
    "TurnEndEvent",
    "MessageStartEvent",
    "MessageUpdateEvent",
    "MessageEndEvent",
    "ToolExecutionStartEvent",
    "ToolExecutionUpdateEvent",
    "ToolExecutionEndEvent",
    # Model Events
    "ModelSelectSource",
    "ModelSelectEvent",
    "ThinkingLevelSelectEvent",
    # User Bash Events
    "UserBashEvent",
    # Input Events
    "InputSource",
    "InputEvent",
    "InputContinueResult",
    "InputTransformResult",
    "InputHandledResult",
    "InputEventResult",
    # Tool Events
    "ToolCallEventBase",
    "BashToolCallEvent",
    "ReadToolCallEvent",
    "EditToolCallEvent",
    "WriteToolCallEvent",
    "GrepToolCallEvent",
    "FindToolCallEvent",
    "LsToolCallEvent",
    "CustomToolCallEvent",
    "ToolCallEvent",
    "ToolResultEventBase",
    "BashToolResultEvent",
    "ReadToolResultEvent",
    "EditToolResultEvent",
    "WriteToolResultEvent",
    "GrepToolResultEvent",
    "FindToolResultEvent",
    "LsToolResultEvent",
    "CustomToolResultEvent",
    "ToolResultEvent",
    "is_bash_tool_result",
    "is_read_tool_result",
    "is_edit_tool_result",
    "is_write_tool_result",
    "is_grep_tool_result",
    "is_find_tool_result",
    "is_ls_tool_result",
    "is_tool_call_event_type",
    # Extension Event Union
    "ExtensionEvent",
    # Event Results
    "ContextEventResult",
    "BeforeProviderRequestEventResult",
    "ToolCallEventResult",
    "UserBashEventResult",
    "ToolResultEventResult",
    "MessageEndEventResult",
    "BeforeAgentStartEventResult",
    "SessionBeforeSwitchResult",
    "SessionBeforeForkResult",
    "SessionBeforeCompactResult",
    "SessionBeforeTreeResult",
    # Message and Entry Rendering
    "MessageRenderOptions",
    "MarkdownTransformContext",
    "MarkdownTransformer",
    "EntryRenderOptions",
    "MessageRenderer",
    "EntryRenderer",
    # Command Registration
    "RegisteredCommand",
    "ResolvedCommand",
    # Extension API
    "ExtensionHandler",
    "ExtensionAPI",
    # Provider Registration Types
    "ProviderConfig",
    "ProviderModelConfig",
    "ExtensionFactory",
    "InlineExtension",
    # Loaded Extension Types
    "RegisteredTool",
    "ExtensionFlag",
    "ExtensionShortcut",
    "HandlerFn",
    "SendMessageHandler",
    "SendUserMessageHandler",
    "AppendEntryHandler",
    "SetSessionNameHandler",
    "GetSessionNameHandler",
    "GetActiveToolsHandler",
    "ToolInfo",
    "GetAllToolsHandler",
    "GetCommandsHandler",
    "SetActiveToolsHandler",
    "RefreshToolsHandler",
    "SetModelHandler",
    "GetThinkingLevelHandler",
    "SetThinkingLevelHandler",
    "SetLabelHandler",
    "ExtensionRuntimeState",
    "ExtensionActions",
    "ExtensionContextActions",
    "ExtensionCommandContextActions",
    "ExtensionRuntime",
    "Extension",
    "LoadExtensionsResult",
    # Extension Error
    "ExtensionError",
]
