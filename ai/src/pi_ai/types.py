"""pi-ai 核心类型（对应 ``pi/packages/ai/src/types.ts``）。

命名约定：Python 使用 ``snake_case``，字段名与 TS 的 ``camelCase`` 一一对应。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Annotated, Literal, Protocol, TypeAlias, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

# ---------------------------------------------------------------------------
# 基础标量
# ---------------------------------------------------------------------------

KnownApi: TypeAlias = Literal[
    "openai-completions",
    "mistral-conversations",
    "openai-responses",
    "azure-openai-responses",
    "openai-codex-responses",
    "anthropic-messages",
    "bedrock-converse-stream",
    "google-generative-ai",
    "google-vertex",
    "pi-messages",
]
"""已知 API 类型（对应 TS ``KnownApi``）。"""

Api: TypeAlias = KnownApi | str
"""API 类型（对应 TS ``Api``）。"""

KnownProvider: TypeAlias = Literal[
    "amazon-bedrock",
    "ant-ling",
    "anthropic",
    "google",
    "google-vertex",
    "openai",
    "azure-openai-responses",
    "openai-codex",
    "radius",
    "nvidia",
    "deepseek",
    "github-copilot",
    "xai",
    "groq",
    "cerebras",
    "openrouter",
    "vercel-ai-gateway",
    "zai",
    "zai-coding-cn",
    "mistral",
    "minimax",
    "minimax-cn",
    "moonshotai",
    "moonshotai-cn",
    "huggingface",
    "fireworks",
    "together",
    "baseten",
    "opencode",
    "opencode-go",
    "kimi-coding",
    "cloudflare-workers-ai",
    "cloudflare-ai-gateway",
    "qwen-token-plan",
    "qwen-token-plan-cn",
    "xiaomi",
    "xiaomi-token-plan-cn",
    "xiaomi-token-plan-ams",
    "xiaomi-token-plan-sgp",
]
"""已知 provider 列表（对应 TS ``KnownProvider``）。"""

ProviderId: TypeAlias = KnownProvider | str
"""Provider 标识符（对应 TS ``ProviderId``）。"""

KnownImagesApi: TypeAlias = Literal["openrouter-images"]
"""已知图片生成 API（对应 TS ``KnownImagesApi``）。"""

ImagesApi: TypeAlias = KnownImagesApi | str
"""图片生成 API（对应 TS ``ImagesApi``）。"""

KnownImagesProvider: TypeAlias = Literal["openrouter"]
"""已知图片生成 provider（对应 TS ``KnownImagesProvider``）。"""

ImagesProviderId: TypeAlias = KnownImagesProvider | str
"""图片生成 provider 标识符（对应 TS ``ImagesProviderId``）。"""

ThinkingLevel: TypeAlias = Literal[
    "off", "minimal", "low", "medium", "high", "xhigh", "max"
]
"""思考级别（对应 TS ``ThinkingLevel``）。"""

ModelThinkingLevel: TypeAlias = Literal["off"] | ThinkingLevel
"""模型思考级别（对应 TS ``ModelThinkingLevel``）。"""

ThinkingLevelMap: TypeAlias = dict[ModelThinkingLevel, str | None]
"""思考级别映射（对应 TS ``ThinkingLevelMap``）。"""

ToolExecutionMode: TypeAlias = Literal["sequential", "parallel"]
"""工具执行模式（对应 TS ``ToolExecutionMode``）。"""

StopReason: TypeAlias = Literal[
    "stop",
    "max_tokens",
    "length",
    "tool_use",
    "error",
    "aborted",
    "deferred",
    "pending",
]
"""assistant 消息停止原因（对应 TS ``StopReason``）。"""

CacheRetention: TypeAlias = Literal["none", "short", "long"]
"""缓存保留策略（对应 TS ``CacheRetention``）。"""

Transport: TypeAlias = Literal["sse", "websocket", "websocket-cached", "auto"]
"""LLM 传输层（对应 TS ``Transport``）。"""

SessionAffinityFormat: TypeAlias = Literal["openai", "openai-nosession", "openrouter"]
"""会话亲和性格式（对应 TS ``SessionAffinityFormat``）。"""

FetchFunction: TypeAlias = Callable[..., object]
"""fetch 函数类型（对应 TS ``FetchFunction``）。"""

ProviderEnv: TypeAlias = dict[str, str]
"""Provider 环境变量（对应 TS ``ProviderEnv``）。"""

ProviderHeaders: TypeAlias = dict[str, str | None]
"""Provider 请求头（对应 TS ``ProviderHeaders``）。"""


# ---------------------------------------------------------------------------
# ThinkingBudgets
# ---------------------------------------------------------------------------


class ThinkingBudgets(BaseModel):
    """各思考级别的 token 预算（对应 TS ``ThinkingBudgets``）。"""

    minimal: int | None = None
    low: int | None = None
    medium: int | None = None
    high: int | None = None


# ---------------------------------------------------------------------------
# Usage / Cost
# ---------------------------------------------------------------------------


class Cost(BaseModel):
    """token 成本（对应 TS ``Cost``）。"""

    input: float = 0.0
    output: float = 0.0
    cache_read: float = 0.0
    cache_write: float = 0.0
    total: float = 0.0


class Usage(BaseModel):
    """token 用量（对应 TS ``Usage``）。"""

    input: int = 0
    output: int = 0
    cache_read: int = 0
    cache_write: int = 0
    total_tokens: int = 0
    cost: Cost = Field(default_factory=Cost)


# ---------------------------------------------------------------------------
# 内容块（ContentBlock）
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """纯文本内容块。"""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """图片内容块（base64 data + mime type）。"""

    type: Literal["image"] = "image"
    data: str
    mime_type: str


class ToolCallContent(BaseModel):
    """工具调用内容块（assistant 消息内）。"""

    type: Literal["toolCall"] = "toolCall"
    tool_call_id: str
    name: str
    args: dict[str, object]


class ToolResultContent(BaseModel):
    """工具结果内容块（toolResult 消息内）。"""

    type: Literal["toolResult"] = "toolResult"
    tool_call_id: str
    content: list[TextContent | ImageContent] = Field(default_factory=list)
    is_error: bool = False
    usage: Usage | None = None


ContentBlock: TypeAlias = Annotated[
    TextContent | ImageContent | ToolCallContent | ToolResultContent,
    Field(discriminator="type"),
]
"""内容块判别联合（对应 TS ``ContentBlock``）。"""


class ThinkingBlock(BaseModel):
    """assistant 思考内容块（仅部分模型族支持）。"""

    type: Literal["thinking"] = "thinking"
    text: str
    signature: str | None = None


# ---------------------------------------------------------------------------
# Message（user / assistant / toolResult）
# ---------------------------------------------------------------------------


class UserMessage(BaseModel):
    """用户消息。"""

    role: Literal["user"] = "user"
    content: list[TextContent | ImageContent]
    timestamp: int


class AssistantMessage(BaseModel):
    """assistant 消息。"""

    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]
    api: str
    provider: str
    model: str
    usage: Usage | None = None
    stop_reason: StopReason
    error_message: str | None = None
    thinking: list[ThinkingBlock] | None = None
    deferred: DeferredHandle | None = None
    response_id: str | None = None
    timestamp: int


class ToolResultMessage(BaseModel):
    """工具结果消息。"""

    role: Literal["toolResult"] = "toolResult"
    content: list[TextContent | ImageContent]
    tool_call_id: str
    tool_name: str | None = None
    is_error: bool = False
    details: object | None = None
    usage: Usage | None = None
    added_tool_names: list[str] | None = None
    timestamp: int


Message: TypeAlias = Annotated[
    UserMessage | AssistantMessage | ToolResultMessage,
    Field(discriminator="role"),
]
"""LLM 可理解的标准消息（对应 TS ``Message``）。"""


# ---------------------------------------------------------------------------
# Model / Context / Tool / SimpleStreamOptions
# ---------------------------------------------------------------------------


@runtime_checkable
class Model(Protocol):
    """LLM 模型句柄（对应 TS ``Model``）。

    ``@runtime_checkable``：``Model`` 被用作 Pydantic 字段类型，需要可 isinstance 校验。
    """

    api: str
    provider: str
    model_id: str


class Tool(BaseModel):
    """工具定义基类（对应 TS ``Tool``，仅名称/描述/参数 schema）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    parameters: dict[str, object]


class Context(BaseModel):
    """发送给 LLM 的上下文（对应 TS ``Context``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[Message]
    system_prompt: str = ""
    tools: list[Tool] | None = None
    session_id: str | None = None
    thinking_level: ThinkingLevel | None = None


class SimpleStreamOptions(BaseModel):
    """流式请求选项（对应 TS ``SimpleStreamOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    transport: Transport | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None
    headers: dict[str, str] | None = None
    metadata: dict[str, object] | None = None
    cache_retention: str | None = None
    session_id: str | None = None


# ---------------------------------------------------------------------------
# ProviderRequestOptions / StreamOptions
# ---------------------------------------------------------------------------


class ProviderResponse(BaseModel):
    """Provider HTTP 响应（对应 TS ``ProviderResponse``）。"""

    status: int
    headers: dict[str, str]


class ProviderRequestOptions(BaseModel):
    """Provider 请求公共选项（对应 TS ``ProviderRequestOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    api_key: str | None = None
    fetch: FetchFunction | None = None
    env: ProviderEnv | None = None
    headers: ProviderHeaders | None = None
    timeout_ms: int | None = None
    max_retries: int | None = None
    max_retry_delay_ms: int | None = None


class StreamOptions(ProviderRequestOptions):
    """流式请求选项（完整版，对应 TS ``StreamOptions``）。"""

    temperature: float | None = None
    sampling_params: dict[str, object] | None = None
    max_tokens: int | None = None
    transport: Transport | None = None
    cache_retention: CacheRetention | None = None
    session_id: str | None = None
    websocket_connect_timeout_ms: int | None = None
    metadata: dict[str, object] | None = None


class DeferredFetchOptions(ProviderRequestOptions):
    """Deferred 请求选项（对应 TS ``DeferredFetchOptions``）。"""

    wait: int = 0


DeferredCancelOptions: TypeAlias = ProviderRequestOptions
"""Deferred 取消选项（对应 TS ``DeferredCancelOptions``）。"""


# ---------------------------------------------------------------------------
# ApiOptionsMap
# ---------------------------------------------------------------------------


class ApiOptionsMap(BaseModel):
    """各 API 的 stream 选项类型映射（对应 TS ``ApiOptionsMap``）。"""

    # 具体类型由各 API 格式化器模块定义


ApiStreamOptions: TypeAlias = StreamOptions
"""统一 API stream 选项（对应 TS ``ApiStreamOptions``）。"""


# ---------------------------------------------------------------------------
# ProviderStreams / ProviderImages
# ---------------------------------------------------------------------------


class ProviderStreams(Protocol):
    """API 实现模块的流式传输契约（对应 TS ``ProviderStreams``）。"""

    def stream(
        self,
        model: Model,
        context: Context,
        options: StreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]: ...

    def stream_simple(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]: ...

    def fetch_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredFetchOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]: ...

    async def cancel_deferred(
        self,
        model: Model,
        handle: DeferredHandle,
        options: DeferredCancelOptions | None = None,
    ) -> None: ...


class ProviderImages(Protocol):
    """图片生成 API 实现契约（对应 TS ``ProviderImages``）。"""

    async def generate_images(
        self,
        model: ImagesModel,
        context: ImagesContext,
        options: ImagesOptions | None = None,
    ) -> AssistantImages: ...


# ---------------------------------------------------------------------------
# DeferredHandle
# ---------------------------------------------------------------------------


class DeferredHandle(BaseModel):
    """Deferred 响应句柄（对应 TS ``DeferredHandle``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    deferred_id: str
    polling_url: str | None = None
    status_url: str | None = None
    cancel_url: str | None = None


# ---------------------------------------------------------------------------
# 图片类型
# ---------------------------------------------------------------------------


class ImagesOptions(BaseModel):
    """图片生成选项（对应 TS ``ImagesOptions``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    metadata: dict[str, object] | None = None


class ImagesModel(Protocol):
    """图片生成模型句柄（对应 TS ``ImagesModel``）。"""

    api: str
    provider: str
    model_id: str


class ImagesContext(BaseModel):
    """图片生成上下文（对应 TS ``ImagesContext``）。"""

    prompt: str
    negative_prompt: str | None = None
    size: str | None = None
    n: int = 1


class AssistantImages(BaseModel):
    """图片生成结果（对应 TS ``AssistantImages``）。"""

    data: list[ImageContent]
    usage: Usage | None = None


# ---------------------------------------------------------------------------
# AssistantMessageEvent 流事件
# ---------------------------------------------------------------------------


class AssistantTextDelta(BaseModel):
    """文本增量事件。"""

    type: Literal["text_delta"] = "text_delta"
    delta: str


class AssistantThinkingDelta(BaseModel):
    """思考增量事件。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    delta: str


class AssistantToolCallStart(BaseModel):
    """工具调用开始事件。"""

    type: Literal["tool_call_start"] = "tool_call_start"
    tool_call_id: str
    name: str


class AssistantToolCallUpdate(BaseModel):
    """工具调用参数更新事件。"""

    type: Literal["tool_call_update"] = "tool_call_update"
    tool_call_id: str
    args: dict[str, object]


class AssistantToolCallEnd(BaseModel):
    """工具调用结束事件。"""

    type: Literal["tool_call_end"] = "tool_call_end"
    tool_call_id: str
    content: list[ContentBlock]


class AssistantUsageDelta(BaseModel):
    """用量增量事件。"""

    type: Literal["usage_delta"] = "usage_delta"
    usage: Usage


class AssistantMessageSnapshot(BaseModel):
    """流中完整消息快照（message_start / message_update / message_end）。"""

    type: Literal["message_snapshot"] = "message_snapshot"
    message: AssistantMessage


class AssistantStreamEnd(BaseModel):
    """流结束事件。"""

    type: Literal["stream_end"] = "stream_end"
    reason: str = ""
    message: AssistantMessage | None = None


class AssistantErrorEvent(BaseModel):
    """流错误事件。"""

    type: Literal["error"] = "error"
    reason: str = "error"
    error: AssistantMessage


class AssistantAbortedEvent(BaseModel):
    """流中止事件。"""

    type: Literal["aborted"] = "aborted"
    error: str | None = None


AssistantMessageEvent: TypeAlias = Annotated[
    AssistantTextDelta
    | AssistantThinkingDelta
    | AssistantToolCallStart
    | AssistantToolCallUpdate
    | AssistantToolCallEnd
    | AssistantUsageDelta
    | AssistantMessageSnapshot
    | AssistantStreamEnd
    | AssistantErrorEvent
    | AssistantAbortedEvent,
    Field(discriminator="type"),
]
"""assistant 消息流事件（对应 TS ``AssistantMessageEvent``）。"""


# ---------------------------------------------------------------------------
# StreamFn
# ---------------------------------------------------------------------------


class StreamFn(Protocol):
    """LLM 调用边界（对应 TS ``StreamFn``）。

    契约：
    - 不得为请求/模型/运行时失败抛异常或返回 rejected promise；
    - 必须返回 ``AsyncIterator[AssistantMessageEvent]``；
    - 失败必须编码进流的协议事件与最终 stop_reason="error"/"aborted" 消息。
    """

    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]: ...
