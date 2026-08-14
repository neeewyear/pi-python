# Pi Python

异步优先的多智能体（Multi-Agent）框架，提供 LLM 统一调用层（pi-ai）、通用 Agent 引擎（pi-agent）、会话数据层（pi-session / pi-session-sqlite）、终端 UI 库（pi-tui）与编码助手产品层（pi-coding-agent）。

## 项目简介

本项目是一个异步优先的 Python 包集合，各包可独立安装、按需组合：

- **统一 LLM 调用层** — 39 家 Provider 工厂（OpenAI、Anthropic、Google、DeepSeek、Mistral、Groq、OpenRouter、Bedrock、Azure OpenAI 等）、认证/OAuth 系统、模型注册表、统一流式接口
- **通用 Agent 引擎** — 完整 Agent 循环（消息队列、状态机、工具执行流水线）、Harness 层（文件系统、Shell 执行、技能加载、上下文压缩）、内置工具集（Bash、读写、Diff/Patch 编辑、图片）
- **会话数据层** — Entry/LaneRecord 数据模型、SessionStorage 契约、上下文构建；可选 SQLite 后端（写租约 fencing、FTS5 全文搜索、分支缓存）
- **终端 UI 库** — 差分渲染引擎、组件系统（Box、Editor、Markdown、SelectList 等）、Flexbox 布局、LaTeX 渲染
- **编码助手产品层** — 完整 CLI（interactive / print / json / rpc 四种模式）、会话管理、工具系统、扩展框架、模型与认证管理

全部源码通过 `mypy --strict` 类型检查，测试覆盖各包核心行为。

## 架构

```
pi-ai ──▶ pi-agent ──▶ pi-coding-agent
                         ▲
pi-session ──────────────┘
     ▲
pi-session-sqlite ───────┘

pi-tui ──▶ pi-coding-agent（interactive 模式）
```

## 包总览

| 包 | 目录 | 职责 | 状态 |
|---|------|------|------|
| [`pi-ai`](ai/) | `ai/` | LLM API 底座：Provider 工厂、认证/OAuth、模型注册表、流式接口 | ✅ 已完成 |
| [`pi-agent`](agent/) | `agent/` | 通用 Agent 引擎：Agent 循环、Harness 层、内置工具集、上下文压缩 | ✅ 已完成 |
| [`pi-session`](session/) | `session/` | 会话数据层：Entry/LaneRecord 模型、SessionStorage 契约、上下文构建 | ✅ 已完成 |
| [`pi-session-sqlite`](session-sqlite/) | `session-sqlite/` | SQLite 会话后端：写租约 fencing、FTS5 搜索、分支缓存 | ✅ 已完成 |
| [`pi-tui`](tui/) | `tui/` | 终端 UI 库：差分渲染、组件系统、编辑器、Markdown/LaTeX 渲染 | ✅ 已完成 |
| [`pi-coding-agent`](coding-agent/) | `coding-agent/` | 编码助手产品层：CLI、工具系统、会话管理、扩展框架 | ✅ 已完成 |

> 各包详细说明见对应目录下的 `README.md` 与 `AGENTS.md`。

### 安装

依赖顺序安装（editable 模式）：

```bash
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

### 启动编码助手 CLI

```bash
python -m pi_coding_agent                    # interactive 模式（TUI）
python -m pi_coding_agent --print "任务描述"  # 非交互：处理提示词后退出
python -m pi_coding_agent --mode json ...    # JSON 输出模式
python -m pi_coding_agent --mode rpc ...     # RPC 模式
```

`main.py` 为端到端工具调用测试入口：Agent 通过注册的 `read` / `write` / `edit` / `bash` 工具自主完成任务，不做任何硬编码写入。

## 开发指南

```bash
# 激活环境
conda activate pi_env

# 类型检查（全项目 strict）
mypy --strict agent/src/ session/src/ ai/src/ coding-agent/src/ session-sqlite/src/

# 运行测试
pytest agent/tests/ session/tests/ tui/tests/ coding-agent/tests/ -v
```

### 环境要求

- Python 3.11+
- [conda](https://docs.conda.io/) 环境 `pi_env`
- 依赖：`pydantic>=2.0`、`aiofiles`、`orjson`、`aiosqlite`、`pytest`、`mypy`

## 项目结构

```
pi-python/
├── ai/                        # pi-ai 包：LLM API 底座
│   └── src/pi_ai/
├── agent/                     # pi-agent 包：通用 Agent 引擎
│   ├── src/pi_agent/
│   └── tests/
├── session/                   # pi-session 包：会话数据层
│   ├── src/pi_session/
│   └── tests/
├── session-sqlite/            # pi-session-sqlite 包：SQLite 会话后端
│   └── src/pi_session_sqlite/
├── tui/                       # pi-tui 包：终端 UI 库
│   ├── src/pi_tui/
│   └── tests/
├── coding-agent/              # pi-coding-agent 包：编码助手产品层
│   ├── src/pi_coding_agent/
│   ├── scripts/
│   └── tests/
├── doc/                       # 规划文档与学习资料（todo / text_doc / ts_structure.md）
├── tests/                     # 根级测试
├── main.py                    # 端到端工具调用测试入口
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

## 参考与文档索引

- `doc/ts_structure.md` — TypeScript 源码结构梳理
- `doc/todo/` — 各模块任务规划与阶段文档
- `doc/text_doc/` — 学习资料
- 各包 `README.md` / `AGENTS.md` — 包级文档与开发约定
