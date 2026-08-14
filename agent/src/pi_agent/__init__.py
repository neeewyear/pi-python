"""Pi Agent 的 Python 重构包。

全部 11 阶段已完成，包含以下模块：

- 基础：``result`` / ``types`` / ``cancellation`` / ``uuid7`` / ``stream_fn``
- agent 循环：``agent_loop`` / ``agent_loop_core`` / ``agent_loop_tools``
- agent 高层封装：``agent`` / ``agent_lifecycle`` / ``agent_queue``
- proxy：``proxy``
- harness：``harness.types`` / ``harness.messages`` / ``harness.skills`` /
  ``harness.system_prompt`` / ``harness.prompt_templates`` /
  ``harness.compaction`` /
  ``harness.agent_harness`` / ``harness.agent_harness_types`` /
  ``harness.env`` / ``harness.tools`` / ``harness.utils``
- session 已独立为 ``pi_session`` 包（见仓库根目录 ``session/``）

共享类型（Model, Message, Context, StreamFn 等）从 ``pi_ai`` 再导出。
"""

from . import (
    agent,
    agent_lifecycle,
    agent_loop,
    agent_loop_core,
    agent_loop_tools,
    agent_queue,
    cancellation,
    harness,
    proxy,
    result,
    stream_fn,
    types,
    uuid7,
)
from .cancellation import CancellationToken
from .result import (
    AgentError,
    Result,
    err,
    get_or_throw,
    get_or_undefined,
    ok,
    to_error,
)
from .stream_fn import set_default_stream_fn
from .uuid7 import uuidv7

# 从 pi_ai 再导出关键类型，确保 pi_agent 用户可直接访问
from pi_ai.types import (
    AssistantMessage,
    AssistantMessageEvent,
    Context,
    Message,
    Model,
    SimpleStreamOptions,
    StreamFn,
    Usage,
)

__all__ = [
    # Agent 模块
    "agent",
    "agent_lifecycle",
    "agent_loop",
    "agent_loop_core",
    "agent_loop_tools",
    "agent_queue",
    "cancellation",
    "harness",
    "proxy",
    "result",
    "stream_fn",
    "types",
    "uuid7",
    # 顶级导出
    "AgentError",
    "AssistantMessage",
    "AssistantMessageEvent",
    "CancellationToken",
    "Context",
    "Message",
    "Model",
    "Result",
    "SimpleStreamOptions",
    "StreamFn",
    "Usage",
    "err",
    "get_or_throw",
    "get_or_undefined",
    "ok",
    "set_default_stream_fn",
    "to_error",
    "uuidv7",
]