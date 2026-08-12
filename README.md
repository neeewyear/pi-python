# Pi Python

`@earendil-works/pi` TypeScript 单仓库的 Python 重构版。异步优先的多智能体（Multi-Agent）框架，提供 LLM 统一调用层、Agent 引擎、会话数据层和编码助手产品层。

## 架构

```
pi-ai ──▶ pi-agent ──▶ pi-coding-agent
                         ▲
pi-session ──────────────┘
     ▲
pi-session-sqlite ───────┘

pi-tui ──▶ pi-coding-agent（interactive 模式）
```

| 包 | 说明 | 行数 | 状态 |
|---|------|------|------|
| [`pi-ai`](ai/) | LLM API 底座：39 家 provider 工厂、认证/OAuth、模型注册表 | 22,243 | ✅ 已完成 |
| [`pi-agent`](agent/) | 通用 Agent 引擎：Agent 循环、Harness 层、内置工具集、上下文压缩 | 6,134 | ✅ 已完成 |
| [`pi-session`](session/) | 会话数据层：Entry/LaneRecord 模型、SessionStorage 契约、上下文构建 | 1,200 | ✅ 已完成 |
| [`pi-tui`](tui/) | 终端 UI 库：差分渲染引擎、组件系统、编辑器、Markdown 渲染 | - | ✅ 已完成 |
| [`pi-coding-agent`](coding-agent/) | 编码助手产品层：CLI、工具系统、会话管理、扩展框架 | 57,985 | ✅ 已完成 |
| [`pi-session-sqlite`](session-sqlite/) | SQLite 会话后端：写租约 fencing、FTS5 搜索、分支缓存 | 2,215 | ✅ 已完成 |

**总计**：92,377 行 TypeScript 源码 → Python 重构，`mypy --strict` 零错误。

## 包依赖

```bash
# 安装顺序（editable 模式）
pip install -e ai/
pip install -e session/
pip install -e agent/
pip install -e tui/
pip install -e coding-agent/
pip install -e session-sqlite/
```

## 快速开始

### 启动 Agent 循环

```python
import asyncio
from pi_agent import agent_loop, AgentContext, AgentLoopConfig
from pi_agent.harness.env.node import NodeExecutionEnv
from pi_agent.harness.messages import convert_to_llm
from pi_agent.harness.tools import bash, read, write, edit
from pi_agent.harness.tools.tool_context import ExecutionToolContext
from pi_agent.types import (
    AssistantMessage, AssistantTextDelta, MessageEndEvent,
    MessageUpdateEvent, ToolExecutionEndEvent, ToolExecutionStartEvent,
    UserMessage,
)

async def main():
    env = NodeExecutionEnv(cwd="/tmp")
    tool_ctx = ExecutionToolContext(env=env)
    tools = [create_tool(env, tool_ctx) for create_tool in
             [read.create_read_tool, write.create_write_tool,
              edit.create_edit_tool, bash.create_bash_tool]]

    context = AgentContext(
        system_prompt="You are a helpful assistant.",
        messages=[UserMessage(content=[{"type": "text", "text": "Hello!"}])],
        tools=tools,
    )
    config = AgentLoopConfig(model=..., convert_to_llm=convert_to_llm)

    async for event in agent_loop(prompts=[], context=context, config=config):
        if isinstance(event, MessageUpdateEvent):
            if isinstance(event.assistant_message_event, AssistantTextDelta):
                print(event.assistant_message_event.delta, end="", flush=True)

asyncio.run(main())
```

### 使用 SQLite 会话后端

```python
from pi_session_sqlite import (
    SqliteSessionRepository, SqliteSessionRepositoryOptions,
    SqliteSessionRepositoryEnv, SqliteSessionCreateOptions,
    SqliteWriterLeaseOptions, create_aiosqlite_factory,
)

repo = SqliteSessionRepository(SqliteSessionRepositoryOptions(
    env=SqliteSessionRepositoryEnv(),
    sqlite=create_aiosqlite_factory(),
    database_path="/tmp/sessions.db",
    writer_lease=SqliteWriterLeaseOptions(ttl_ms=60_000),
))
session = await repo.create(SqliteSessionCreateOptions(cwd="/tmp"))
meta = await session.get_metadata()
print(f"Session: {meta.id}")
await repo.close()
```

## 开发

```bash
# 激活环境
conda activate pi_env

# 类型检查（全项目 strict）
mypy --strict agent/src/ session/src/ ai/src/ coding-agent/src/ session-sqlite/src/

# 运行测试
pytest agent/tests/ session/tests/ -v
```

### 环境要求

- Python 3.11+
- [conda](https://docs.conda.io/) 环境 `pi_env`
- 依赖：`pydantic>=2.0`、`aiofiles`、`orjson`、`aiosqlite`、`pytest`、`mypy`

## 项目结构

```
pi-python/
├── agent/                  # pi-agent 包
│   ├── src/pi_agent/       #   源码
│   └── tests/              #   测试
├── session/                # pi-session 包
│   ├── src/pi_session/     #   源码
│   └── tests/              #   测试
├── ai/                     # pi-ai 包
│   └── src/pi_ai/          #   源码
├── coding-agent/           # pi-coding-agent 包
│   └── src/pi_coding_agent/#   源码
├── session-sqlite/         # pi-session-sqlite 包
│   └── src/pi_session_sqlite/  # 源码
├── tui/                    # pi-tui 包
│   └── src/pi_tui/         #   源码
├── pi/                     # 原始 TypeScript 单仓库（重构参考）
├── todo/                   # 任务规划文档
├── main.py                 # 端到端测试入口
└── .gitignore
```

## 技术栈

- Python 3.11+
- Pydantic v2（数据模型 + Discriminated Union）
- aiofiles（异步文件 I/O）
- orjson（高性能 JSON）
- aiosqlite（异步 SQLite）
- pytest + pytest-asyncio（测试）
- mypy --strict（类型安全）