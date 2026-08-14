"""AgentHarness 类。

当前实现为 scaffold：所有执行类方法返回 ``HarnessNotImplemented``，
属性读写方法有实际实现。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Awaitable, Callable, TypeAlias

from pydantic import BaseModel

from ..types import (
    AgentMessage,
    AgentTool,
    ImageContent,
    Model,
    QueueMode,
    ThinkingLevel,
    Usage,
)
from .agent_harness_types import (
    AbortResult,
    AgentHarnessOptions,
    ActionInfo,
    CancelQueuedResult,
    CompactionResult,
    CreateLaneResult,
    HarnessClosed,
    HarnessNotImplemented,
    HarnessTool,
    LaneInfo,
    LaneSnapshot,
    LaneQueues,
    NavigateOptions,
    NavigationResult,
    PassiveRegistry,
    PendingWrite,
    QueueResult,
    RecordUsageResult,
    Resources,
    ResumeResult,
    RetryPolicy,
    RunResult,
    SessionSnapshot,
    StreamOptions,
    WatchHandle,
)
from .compaction.compaction import CompactionSettings

if TYPE_CHECKING:
    from pi_session.session import Session

# 注：SessionTree 契约由 Session 实现，因此使用 Session 类型。
# AgentLane 协议从 agent_harness_types 导入。


class AgentHarness:
    """AgentHarness scaffold 实现。  

    实现 ``AgentLane`` 协议，当前所有执行类方法返回
    ``HarnessNotImplemented`` 或 ``HarnessClosed``。
    """

    name: str = "main"
    hooks: PassiveRegistry
    events: PassiveRegistry

    # Session 类型（私有）
    _durable_session: Session
    _model: Model
    _thinking_level: ThinkingLevel
    _active_tool_names: list[str]
    _tools: list[HarnessTool]
    _resources: Resources
    _stream_options: StreamOptions
    _retry_policy: RetryPolicy
    _compaction_settings: CompactionSettings
    _steering_mode: QueueMode
    _follow_up_mode: QueueMode
    _closed: bool = False

    def __init__(self, options: AgentHarnessOptions) -> None:
        # session 字段在 options 中是 object（避免循环导入），运行时确保是 Session
        self._durable_session = options.session  # type: ignore[assignment]
        self._model = options.model
        self._thinking_level = options.thinking_level if options.thinking_level else "off"
        self._active_tool_names = list(
            options.active_tool_names
            if options.active_tool_names
            else ([t.name for t in options.tools] if options.tools else [])
        )
        self._tools = list(options.tools) if options.tools else []
        self._resources = Resources(
            skills=list(options.resources.skills) if options.resources and options.resources.skills else None,
            prompt_templates=(
                list(options.resources.prompt_templates)
                if options.resources and options.resources.prompt_templates
                else None
            ),
        )
        self._stream_options = StreamOptions.model_validate(
            options.stream_options.model_dump() if options.stream_options else {}
        )
        self._retry_policy = options.retry if options.retry else RetryPolicy(
            enabled=False, max_retries=0, base_delay_ms=1000
        )
        self._compaction_settings = options.compaction if options.compaction else CompactionSettings(
            enabled=True, reserve_tokens=16384, keep_recent_tokens=20000
        )
        self._steering_mode = options.steering_mode if options.steering_mode else "one-at-a-time"
        self._follow_up_mode = options.follow_up_mode if options.follow_up_mode else "one-at-a-time"
        self.hooks = PassiveRegistry()
        self.events = PassiveRegistry()

    @staticmethod
    async def create(
        options: AgentHarnessOptions,
        ) -> tuple[AgentHarness, list[object]]:
        """静态工厂方法。"""
        return AgentHarness(options), []

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _unavailable(self, operation: str) -> HarnessNotImplemented | HarnessClosed:
        """返回未实现或已关闭错误。

        注意：TS 版本返回 ``Promise.reject(...)``，Python 侧直接抛出异常。
        """
        if self._closed:
            raise HarnessClosed()
        raise HarnessNotImplemented(operation)

    async def _unavailable_async(self, operation: str) -> object:
        """异步版本的 unavailable。"""
        raise self._unavailable(operation)

    # ------------------------------------------------------------------
    # 运行控制（scaffold）
    # ------------------------------------------------------------------

    async def prompt(self, text: str, images: list[ImageContent] | None = None) -> RunResult:
        raise self._unavailable("prompt")

    async def skill(self, name: str, additional_instructions: str | None = None) -> RunResult:
        raise self._unavailable("skill")

    async def prompt_from_template(self, name: str, args: list[str] | None = None) -> RunResult:
        raise self._unavailable("promptFromTemplate")

    # ------------------------------------------------------------------
    # 操作管理（scaffold）
    # ------------------------------------------------------------------

    async def compact(
        self, options: dict[str, str] | None = None
    ) -> CompactionResult:
        raise self._unavailable("compact")

    async def navigate_tree(
        self, target_id: str | None, options: NavigateOptions | None = None
    ) -> NavigationResult:
        raise self._unavailable("navigateTree")

    async def resume(self) -> ResumeResult:
        raise self._unavailable("resume")

    async def abort(self) -> AbortResult:
        raise self._unavailable("abort")

    # ------------------------------------------------------------------
    # 队列操作（scaffold）
    # ------------------------------------------------------------------

    async def steer(self, text: str, images: list[ImageContent] | None = None) -> QueueResult:
        raise self._unavailable("steer")

    async def follow_up(self, text: str, images: list[ImageContent] | None = None) -> QueueResult:
        raise self._unavailable("followUp")

    async def next_run(self, text: str, images: list[ImageContent] | None = None) -> QueueResult:
        raise self._unavailable("nextRun")

    async def cancel_queued(self, entry_id: str) -> CancelQueuedResult:
        raise self._unavailable("cancelQueued")

    # ------------------------------------------------------------------
    # 用量记录（scaffold）
    # ------------------------------------------------------------------

    async def record_usage(
        self, usage: Usage, options: dict[str, object] | None = None
    ) -> RecordUsageResult:
        raise self._unavailable("recordUsage")

    # ------------------------------------------------------------------
    # 空闲等待
    # ------------------------------------------------------------------

    async def wait_for_idle(self) -> None:
        """等待空闲（当前为空实现）。"""
        return

    async def run_when_idle(self, callback: Callable[[], None | Awaitable[None]]) -> None:
        """空闲时运行回调。"""
        result = callback()
        if hasattr(result, "__await__"):
            await result  # type: ignore[misc]

    # ------------------------------------------------------------------
    # 动作执行
    # ------------------------------------------------------------------

    async def peek_action(self) -> ActionInfo | None:
        """查看待执行动作。"""
        return None

    async def execute_action(self) -> ActionInfo | None:
        """执行动作。"""
        return None

    async def run_to_completion(self) -> None:
        """运行至完成（当前为空实现）。"""
        return

    # ------------------------------------------------------------------
    # 属性访问（有实际实现）
    # ------------------------------------------------------------------

    async def get_leaf_id(self) -> str | None:
        """获取当前叶子 ID。"""
        return await self._durable_session.get_leaf_id()

    async def get_model(self) -> Model:
        """获取当前模型。"""
        return self._model

    async def set_model(self, model: Model) -> None:
        """设置模型。"""
        self._model = model

    async def get_thinking_level(self) -> ThinkingLevel:
        """获取思考级别。"""
        return self._thinking_level

    async def set_thinking_level(self, level: ThinkingLevel) -> None:
        """设置思考级别。"""
        self._thinking_level = level

    async def get_active_tools(self) -> list[str]:
        """获取活跃工具名列表。"""
        return list(self._active_tool_names)

    async def set_active_tools(self, names: list[str]) -> None:
        """设置活跃工具名列表。"""
        self._active_tool_names = list(names)

    @property
    def session(self) -> Session:
        """会话树视图（公开属性）。"""
        return self._durable_session

    # ------------------------------------------------------------------
    # 监听
    # ------------------------------------------------------------------

    async def watch(self) -> WatchHandle[LaneSnapshot]:
        """获取车道快照监听句柄。"""
        leaf_id = await self.get_leaf_id()
        transcript: list[object] = (
            [] if leaf_id is None
            else list(await self._durable_session.find_entries_on_branch(
                {"start": leaf_id, "order": "oldestFirst"}  # type: ignore[arg-type]
            ))
        )
        return WatchHandle[LaneSnapshot](
            snapshot=LaneSnapshot(
                lane=self.name,
                transcript=transcript,
                leaf_id=leaf_id,
                operation=None,
                queues=LaneQueues(),
                pending_writes=[],
                faulted=False,
            ),
        )

    # ------------------------------------------------------------------
    # 车道管理
    # ------------------------------------------------------------------

    async def lane(self, name: str) -> object | None:
        """获取车道。"""
        return self if name == "main" else None

    async def create_lane(self, name: str, at: str | None) -> CreateLaneResult:
        raise self._unavailable("createLane")

    async def lanes(self) -> list[LaneInfo]:
        """获取所有车道信息。"""
        pointers = await self._durable_session.storage.get_lanes()
        return [
            LaneInfo(name=p.lane, leaf_id=p.leaf_id, operation=None)
            for p in pointers
        ]

    # ------------------------------------------------------------------
    # 工具 / 资源管理
    # ------------------------------------------------------------------

    async def get_tools(self) -> list[HarnessTool]:
        """获取工具列表（拷贝）。"""
        return list(self._tools)

    async def set_tools(self, tools: list[HarnessTool], active_names: list[str] | None = None) -> None:
        """设置工具列表。"""
        self._tools = list(tools)
        self._active_tool_names = list(active_names if active_names is not None else [t.name for t in tools])

    async def get_resources(self) -> Resources:
        """获取资源（拷贝）。"""
        return Resources(
            skills=list(self._resources.skills) if self._resources.skills else None,
            prompt_templates=(
                list(self._resources.prompt_templates)
                if self._resources.prompt_templates
                else None
            ),
        )

    async def set_resources(self, resources: Resources) -> None:
        """设置资源。"""
        self._resources = Resources(
            skills=list(resources.skills) if resources.skills else None,
            prompt_templates=(
                list(resources.prompt_templates)
                if resources.prompt_templates
                else None
            ),
        )

    # ------------------------------------------------------------------
    # 配置读写
    # ------------------------------------------------------------------

    async def get_stream_options(self) -> StreamOptions:
        """获取流选项（拷贝）。"""
        return StreamOptions.model_validate(self._stream_options.model_dump())

    async def set_stream_options(self, options: StreamOptions) -> None:
        """设置流选项。"""
        self._stream_options = StreamOptions.model_validate(options.model_dump())

    async def get_retry_policy(self) -> RetryPolicy:
        """获取重试策略（拷贝）。"""
        return RetryPolicy.model_validate(self._retry_policy.model_dump())

    async def set_retry_policy(self, policy: RetryPolicy) -> None:
        """设置重试策略。"""
        self._retry_policy = RetryPolicy.model_validate(policy.model_dump())

    async def get_compaction_settings(self) -> CompactionSettings:
        """获取压缩设置（拷贝）。"""
        return CompactionSettings.model_validate(self._compaction_settings.model_dump())

    async def set_compaction_settings(self, settings: CompactionSettings) -> None:
        """设置压缩设置。"""
        self._compaction_settings = CompactionSettings.model_validate(settings.model_dump())

    async def get_steering_mode(self) -> QueueMode:
        """获取转向模式。"""
        return self._steering_mode

    async def set_steering_mode(self, mode: QueueMode) -> None:
        """设置转向模式。"""
        self._steering_mode = mode

    async def get_follow_up_mode(self) -> QueueMode:
        """获取跟进模式。"""
        return self._follow_up_mode

    async def set_follow_up_mode(self, mode: QueueMode) -> None:
        """设置跟进模式。"""
        self._follow_up_mode = mode

    # ------------------------------------------------------------------
    # 会话监听
    # ------------------------------------------------------------------

    async def watch_session(self) -> WatchHandle[SessionSnapshot]:
        """获取会话快照监听句柄。"""
        return WatchHandle[SessionSnapshot](
            snapshot=SessionSnapshot(
                lanes=[],
                faulted=False,
            ),
        )

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """关闭 harness。"""
        self._closed = True