#!/usr/bin/env python3
"""Pi Agent 工具调用测试 — 测试 agent_loop 自动调用工具。

任务：读取 ``TypeScript学习指南.md``，归纳总结后写入
``TypeScript 学习指南_总结.md``。

Agent 通过注册的 read / write / edit / bash 工具自主完成任务，
main.py 不做任何硬编码文件写入。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from pathlib import Path

from pi_agent import set_default_stream_fn
from pi_agent.agent_loop import agent_loop
from pi_agent.deepseek_provider import DeepSeekModel, create_deepseek_stream_fn
from pi_agent.harness.env.node import NodeExecutionEnv
from pi_agent.harness.messages import convert_to_llm
from pi_agent.harness.tools import bash, edit, read, write
from pi_agent.harness.tools.tool_context import ExecutionToolContext
from pi_agent.harness.types import AgentHarnessTool
from pi_agent.types import (
    AgentContext,
    AgentEndEvent,
    AgentLoopConfig,
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    AssistantMessage,
    AssistantTextDelta,
    CancellationToken,
    MessageEndEvent,
    MessageUpdateEvent,
    TextContent,
    ToolExecutionEndEvent,
    ToolExecutionStartEvent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# 模型配置
# ---------------------------------------------------------------------------

MODEL_ID = "deepseek-v4-flash"
BASE_URL = "https://api.deepseek.com"
TIMEOUT_MS = 120_000

# 工作区根目录：main.py 所在目录（pi-python）
WORKSPACE_ROOT = Path(__file__).resolve().parent

# INPUT_FILE = WORKSPACE_ROOT / "TypeScript学习指南.md"
# OUTPUT_FILE = WORKSPACE_ROOT / "TypeScript 学习指南_总结.md"
INPUT_FILE = "doc/text_doc/TypeScript学习指南.md"
OUTPUT_FILE = "doc/text_doc/TypeScript 学习指南_总结.md"

# ---------------------------------------------------------------------------
# 主逻辑
# ---------------------------------------------------------------------------


async def main() -> None:

    # 注册 stream_fn
    stream_fn = create_deepseek_stream_fn(
        model_id=MODEL_ID,
        base_url=BASE_URL,
        timeout_ms=TIMEOUT_MS,
    )
    set_default_stream_fn(stream_fn)

    # 执行环境（cwd = 工作区根目录）
    env = NodeExecutionEnv(cwd=str(WORKSPACE_ROOT))
    tools = _build_tools(env)

    # 构建上下文
    context = AgentContext(
        system_prompt=(
            "你是一位资深的 TypeScript 讲师，擅长由浅入深地讲解 TypeScript。"
            "请用中文回答，内容准确、结构清晰，并尽量给出可直接运行的代码示例。\n\n"
            "当需要读写文件或执行命令时，必须使用提供的工具完成，不要臆造文件内容。"
        ),
        messages=[
            UserMessage(
                content=[
                    TextContent(
                        text=(
                            f"请读取 {INPUT_FILE} 文件内容，"
                            f"归纳总结为一个结构清晰的 Markdown 文档，"
                            f"然后使用 write 工具将总结保存到 {OUTPUT_FILE}。"
                        )
                    )
                ],
                timestamp=int(time.time() * 1000),
            ),
        ],
        tools=tools,
    )

    # 构建配置
    config = AgentLoopConfig(
        model=DeepSeekModel(model_id=MODEL_ID),
        convert_to_llm=convert_to_llm,
    )

    # 启动 agent_loop
    stream = agent_loop(
        prompts=[],
        context=context,
        config=config,
        signal=None,
    )

    # 消费事件流
    try:
        # 从事件流（AgentEventStream 内部的 asyncio.Queue）逐个拉取事件，
        # 后台任务每 push 一个事件，这里就消费一个——典型的生产者-消费者模式
        async for event in stream:
            # 分支①：assistant 消息的流式增量（LLM 正在逐字吐文本 / 思考）
            if isinstance(event, MessageUpdateEvent):
                inner = event.assistant_message_event
                # 只处理文本增量，实时打印到终端（不换行，模拟打字机效果）
                if isinstance(inner, AssistantTextDelta):
                    print(inner.delta, end="", flush=True)
            # 分支②：一条 assistant 消息流式结束（本轮 LLM 输出完毕）
            elif isinstance(event, MessageEndEvent):
                msg = event.message
                # 确认是 assistant 消息（非 toolResult），换行结束当前文本流
                if isinstance(msg, AssistantMessage):
                    print(flush=True)
                    # 若模型返回了 token 用量，打印统计便于观察成本
                    if msg.usage:
                        print(
                            f"[Usage] input={msg.usage.input} "
                            f"output={msg.usage.output} "
                            f"total={msg.usage.total_tokens}",
                            flush=True,
                        )
            # 分支③：工具开始执行（agent 决定调用 read/write/bash 等）
            # 打印工具名与入参，便于排查「参数是否正确解析」
            elif isinstance(event, ToolExecutionStartEvent):
                print(
                    f"\n[Tool] 调用 {event.tool_name}(args={event.args}) ...",
                    flush=True,
                )
            # 分支④：工具执行结束，根据 is_error 打印成功 / 失败状态
            # 这里是排查工具调用失败的关键观测点
            elif isinstance(event, ToolExecutionEndEvent):
                status = "OK" if not event.is_error else "ERROR"
                print(f"[Tool] {event.tool_name} → {status}", flush=True)
            # 分支⑤：整个 agent 循环结束（所有 turn 跑完或被中止）
            # 仅换行收尾；最终消息列表可通过 await stream.result() 获取
            elif isinstance(event, AgentEndEvent):
                print(flush=True)
    except Exception as exc:
        print(f"\n[Error] {exc}", flush=True)
        return

    # 校验输出文件是否由 agent 通过工具生成
    output_path = Path(OUTPUT_FILE)
    if output_path.exists():
        size = output_path.stat().st_size
        print(f"\n[Verify] {output_path.name} 已生成（{size} 字节）", flush=True)
    else:
        print(f"\n[Warning] {output_path.name} 未生成", flush=True)

    print("\nDone.", flush=True)


# ---------------------------------------------------------------------------
# 工具适配：AgentHarnessTool → AgentTool
# ---------------------------------------------------------------------------


def _adapt_harness_tool(
    tool: AgentHarnessTool,
    ctx: ExecutionToolContext,
) -> AgentTool:
    """将内置 harness 工具（5 参 execute）适配为 AgentTool（4 参 execute）。

    通过闭包注入固定的 ``ExecutionToolContext``，供 read/write/edit/bash 使用。
    """
    execute = tool.execute
    name = tool.name
    description = tool.description
    parameters = tool.parameters
    label = tool.label

    async def _execute(
        tool_call_id: str,
        args: dict[str, object],
        signal: CancellationToken | None,
        on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        result = await execute(tool_call_id, args, signal, on_update, ctx)
        return result

    return AgentTool(
        name=name,
        description=description,
        parameters=parameters,
        label=label,
        execute=_execute,
    )


def _build_tools(env: NodeExecutionEnv) -> list[AgentTool]:
    """注册内置工具并注入执行上下文。"""
    tool_ctx = ExecutionToolContext(env=env)

    adapters: list[Callable[[], AgentHarnessTool]] = [
        read.create_read_tool,
        write.create_write_tool,
        edit.create_edit_tool,
        bash.create_bash_tool,
    ]

    return [_adapt_harness_tool(adapter(), tool_ctx) for adapter in adapters]


if __name__ == "__main__":
    asyncio.run(main())
