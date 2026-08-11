"""测试 agent_loop 能否正常启动 —— 使用 DeepSeek API。

用法：
    # 确保 ~/.zshrc 中已配置 DEEPSEEK_API_KEY，然后：
    source ~/.zshrc
    cd /Users/alex/code/学习/pi-python/agent
    python test_agent_loop.py
"""

import asyncio
import os
import sys

# 确保项目在 sys.path 中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from pi_agent.agent import Agent
from pi_agent.agent_lifecycle import AgentOptions
from pi_agent.agent_loop import agent_loop
from pi_agent.cancellation import CancellationToken
from pi_agent.deepseek_provider import DeepSeekModel, create_deepseek_stream_fn
from pi_agent.types import (
    AgentContext,
    AgentEvent,
    AgentLoopConfig,
    AgentState,
    TextContent,
    UserMessage,
)

# ---------------------------------------------------------------------------
# 从 ~/.zshrc 读取 API key
# ---------------------------------------------------------------------------


def _load_api_key_from_zshrc() -> str:
    """尝试从 ~/.zshrc 中读取 DEEPSEEK_API_KEY。"""
    zshrc_path = os.path.expanduser("~/.zshrc")
    if not os.path.isfile(zshrc_path):
        return ""
    with open(zshrc_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("export DEEPSEEK_API_KEY="):
                # 提取 = 后面的值，去除引号
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                return value
    return ""


# ---------------------------------------------------------------------------
# 事件监听器
# ---------------------------------------------------------------------------


def create_event_listener() -> callable:
    """创建事件监听器，打印关键事件到控制台。"""

    async def on_event(event: AgentEvent, _signal: CancellationToken) -> None:
        event_type = (
            event.type if hasattr(event, "type") else event.get("type", "unknown")
        )

        if event_type == "agent_start":
            print("[agent_start] Agent 开始处理")
        elif event_type == "turn_start":
            print("[turn_start] 新回合开始")
        elif event_type == "message_start":
            msg = event.message if hasattr(event, "message") else event["message"]
            print(f"[message_start] {msg.role}")
        elif event_type == "message_update":
            msg = event.message if hasattr(event, "message") else event["message"]
            # 只打印文本增量（不打印完整消息）
            delta = (
                event.assistant_message_event
                if hasattr(event, "assistant_message_event")
                else event.get("assistant_message_event")
            )
            if (
                delta is not None
                and hasattr(delta, "type")
                and delta.type == "text_delta"
            ):
                print(delta.delta, end="", flush=True)
        elif event_type == "message_end":
            msg = event.message if hasattr(event, "message") else event["message"]
            print(f"\n[message_end] {msg.role}")
        elif event_type == "tool_execution_start":
            tc_id = (
                event.tool_call_id
                if hasattr(event, "tool_call_id")
                else event["tool_call_id"]
            )
            tool_name = (
                event.tool_name if hasattr(event, "tool_name") else event["tool_name"]
            )
            print(f"[tool_start] {tool_name} (id={tc_id})")
        elif event_type == "tool_execution_end":
            tool_name = (
                event.tool_name if hasattr(event, "tool_name") else event["tool_name"]
            )
            is_error = (
                event.is_error
                if hasattr(event, "is_error")
                else event.get("is_error", False)
            )
            status = "ERROR" if is_error else "OK"
            print(f"[tool_end] {tool_name} -> {status}")
        elif event_type == "turn_end":
            print("[turn_end] 回合结束")
        elif event_type == "agent_end":
            messages = (
                event.messages if hasattr(event, "messages") else event["messages"]
            )
            print(f"[agent_end] Agent 完成，共 {len(messages)} 条新消息")
        else:
            print(f"[{event_type}]")

    return on_event


# ---------------------------------------------------------------------------
# 方式一：使用高层 Agent 封装（推荐）
# ---------------------------------------------------------------------------


async def test_with_agent(api_key: str) -> None:
    """使用高层 Agent 类测试 agent_loop。"""
    print("=" * 60)
    print("方式一：使用 Agent 高层封装")
    print("=" * 60)

    model = DeepSeekModel(model_id="deepseek-v4-flash")
    stream_fn = create_deepseek_stream_fn(api_key=api_key)

    initial_state = AgentState(
        system_prompt="你是一个有用的助手，请用中文回答。",
        model=model,
    )

    options = AgentOptions(
        initial_state=initial_state,
        stream_fn=stream_fn,
    )

    agent = Agent(options)
    unsubscribe = agent.subscribe(create_event_listener())

    try:
        print(
            "\n发送 prompt: '你好，请简单介绍一下你自己，然后用一句话说明 1+1 等于几。'\n"
        )
        await agent.prompt("你好，请简单介绍一下你自己，然后用一句话说明 1+1 等于几。")
        await agent.wait_for_idle()

        print("\n--- Agent 返回的消息 ---")
        for msg in agent.state.messages:
            role = msg.role
            if role == "assistant":
                for block in msg.content:
                    if hasattr(block, "type") and block.type == "text":
                        print(f"[{role}]: {block.text}")
            else:
                print(f"[{role}]: (非文本内容)")

    finally:
        unsubscribe()


# ---------------------------------------------------------------------------
# 方式二：使用低层 agent_loop 直接调用
# ---------------------------------------------------------------------------


async def test_with_agent_loop_direct(api_key: str) -> None:
    """使用低层 agent_loop 函数直接测试。"""
    print("\n" + "=" * 60)
    print("方式二：使用低层 agent_loop 直接调用")
    print("=" * 60)

    model = DeepSeekModel(model_id="deepseek-v4-flash")
    stream_fn = create_deepseek_stream_fn(api_key=api_key)

    prompt = UserMessage(
        content=[TextContent(text="用一句话回答：Python 是什么？")],
        timestamp=0,
    )

    context = AgentContext(
        system_prompt="你是一个有用的助手，请用中文回答。",
        messages=[],
    )

    config = AgentLoopConfig(
        model=model,
        convert_to_llm=lambda msgs: [
            m for m in msgs if m.role in ("user", "assistant", "toolResult")
        ],
    )

    signal = CancellationToken()
    on_event = create_event_listener()

    print("\n发送 prompt: '用一句话回答：Python 是什么？'\n")

    stream = agent_loop([prompt], context, config, signal, stream_fn)

    async for event in stream:
        await on_event(event, signal)

    result = await stream.result()
    print(f"\n--- 最终消息数: {len(result)} ---")
    for msg in result:
        if msg.role == "assistant":
            for block in msg.content:
                if hasattr(block, "type") and block.type == "text":
                    print(f"[assistant]: {block.text}")


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def main() -> None:
    """启动函数：测试 agent_loop 能否正常启动。"""
    # 1. 读取 API key
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = _load_api_key_from_zshrc()
    if not api_key:
        print("错误：未找到 DEEPSEEK_API_KEY")
        print("请确保 ~/.zshrc 中有: export DEEPSEEK_API_KEY='sk-xxx'")
        print("然后运行: source ~/.zshrc && python test_agent_loop.py")
        sys.exit(1)

    print(f"API Key: {api_key[:8]}...{api_key[-4:]}")
    print("Base URL: https://api.deepseek.com")
    print("Model: deepseek-v4-flash\n")

    # 2. 方式一：高层 Agent 封装
    await test_with_agent(api_key)

    # 3. 方式二：低层 agent_loop 直接调用
    await test_with_agent_loop_direct(api_key)

    print("\n✅ 测试完成！agent_loop 正常启动并运行。")


if __name__ == "__main__":
    asyncio.run(main())
