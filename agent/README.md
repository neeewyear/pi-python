# Pi Agent Python

Pi Agent 模块的 Python 重构，对应 `@earendil-works/pi-agent-core` TypeScript 实现。

## 项目概况

异步优先的多智能体（Multi-Agent）框架核心库，提供：

- **Agent 循环** — 完整的 Agent 生命周期管理（消息队列、状态机、工具执行流水线）
- **Harness 层** — 文件系统、Shell 执行、技能加载、上下文压缩
- **内置工具** — Bash、文件读写、Diff/Patch 编辑、图片处理
- **类型安全** — 全量 Pydantic v2 模型 + mypy 严格模式

> Session 数据层已独立为 [`pi-session`](../session/) 包，本包通过 `pi_session` 依赖它。

## 架构

```
pi_agent/
├── agent.py / agent_lifecycle.py / agent_queue.py   # Agent 高层封装
├── agent_loop.py / agent_loop_core.py / agent_loop_tools.py  # Agent 循环
├── proxy.py / stream_fn.py                          # 流式代理
├── result.py / cancellation.py / uuid7.py / types.py # 基础工具
│
├── harness/
│   ├── types.py                      # FileError / ExecutionError / FileSystem / Shell 协议
│   ├── messages.py / system_prompt.py # 消息转换与系统提示词
│   ├── skills.py / prompt_templates.py # 技能与模板加载
│   ├── compaction/                   # 上下文压缩（Token 估算 / 摘要 / 分支摘要）
│   ├── agent_harness.py              # Agent Harness 脚手架
│   ├── env/
│   │   ├── node_fs.py                # 文件系统实现（aiofiles）
│   │   └── node.py                   # Shell 执行 + NodeExecutionEnv
│   ├── tools/
│   │   ├── bash.py / read.py / write.py / edit.py  # 内置工具
│   │   ├── edit_diff.py              # Diff/Patch 算法
│   │   ├── image.py                  # 图片 MIME 检测 + Base64
│   │   └── file_mutation_queue.py / path_utils.py / tool_context.py  # 工具辅助
│   └── utils/
│       ├── truncate.py               # 文本截断
│       └── shell_output.py           # Shell 输出捕获
│
└── deepseek_provider.py / node.py    # 扩展接入点
```

## 安装

```bash
# 创建 conda 环境
conda create -n pi_env python=3.11
conda activate pi_env

# 安装依赖（先装 pi-session，再装本包；两包互依赖，editable 下无发布问题）
cd session
pip install -e ".[dev]"
cd ../agent
pip install -e ".[dev]"
```

## 快速开始

```python
from pi_agent import CancellationToken, ok, err, uuidv7
from pi_agent.harness.env.node import NodeExecutionEnv

# 创建执行环境
env = NodeExecutionEnv(cwd="/tmp")

# 文件操作
result = await env.read_text_file("/path/to/file.txt")
if result.is_ok():
    print(result.value)

# Shell 执行
result = await env.exec("echo hello")
if result.is_ok():
    print(result.value.stdout)  # "hello\n"
```

```python
from pi_agent.harness.skills import load_skills
from pi_agent.harness.system_prompt import format_skills_for_system_prompt

# 加载技能
skills, diagnostics = await load_skills(env, "/path/to/skills")

# 格式化系统提示词
prompt = format_skills_for_system_prompt(skills)
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_agent/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 单文件测试
conda run -n pi_env python -m pytest tests/test_result.py -v
```

## 技术栈

- Python 3.11+
- Pydantic v2（数据模型）
- aiofiles（异步文件 I/O）
- orjson（高性能 JSON）
- pytest + pytest-asyncio（测试）

## 许可证

Apache 2.0