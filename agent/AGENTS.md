# Pi Agent Python — Agent 开发指南

## 项目介绍

Pi Agent 模块的 Python 重构，对应 `@earendil-works/pi-agent-core` TypeScript 实现。异步优先的多智能体框架核心库，提供 Agent 循环、Harness 层（文件系统 / Shell / 技能 / 压缩）、内置工具集。

> Session 数据层已独立为 `pi-session` 包（仓库根目录 `session/`），本包通过 `pi_session` 依赖它。

## 开发环境

```bash
# 创建并激活环境（已配置）
conda create -n pi_env python=3.11
conda activate pi_env

# 安装（editable 模式 + dev 依赖；先装 pi-session 再装本包，两包互依赖）
cd session
pip install -e ".[dev]"
cd ../agent
pip install -e ".[dev]"
```

## 目录结构

```
agent/
├── src/pi_agent/                  # 源码
│   ├── agent.py                   # Agent 高层封装（~530 行）
│   ├── agent_loop.py              # AgentEventStream 与入口函数
│   ├── agent_loop_core.py         # Agent 循环主逻辑（流式事件）
│   ├── agent_loop_tools.py        # 工具执行流水线
│   ├── agent_lifecycle.py         # Agent 生命周期（状态机）
│   ├── agent_queue.py             # 待处理消息队列
│   ├── proxy.py                   # 流式代理
│   ├── stream_fn.py               # 默认流函数注册表
│   ├── result.py                  # Result 类型（ok/err）
│   ├── cancellation.py            # CancellationToken
│   ├── uuid7.py                   # UUIDv7 生成
│   ├── types.py                   # 核心类型定义
│   └── harness/                   # Harness 层
│       ├── types.py               # 协议与错误类型
│       ├── messages.py            # 消息转换
│       ├── system_prompt.py       # 系统提示词
│       ├── skills.py              # 技能加载
│       ├── prompt_templates.py    # 提示词模板
│       ├── agent_harness.py       # Agent Harness
│       ├── compaction/            # 上下文压缩
│       ├── env/                   # 执行环境
│       │   ├── node_fs.py         # 文件系统（aiofiles）
│       │   └── node.py            # Shell + NodeExecutionEnv
│       ├── tools/                 # 内置工具
│       │   ├── bash.py / read.py / write.py / edit.py
│       │   ├── edit_diff.py       # Diff/Patch 算法
│       │   ├── image.py           # 图片处理
│       │   └── file_mutation_queue.py / path_utils.py
│       └── utils/                 # 工具函数
│           ├── truncate.py        # 文本截断
│           └── shell_output.py    # Shell 输出捕获
├── tests/                         # 测试（pytest + pytest-asyncio）
├── main.py                        # DeepSeek 启动测试入口
├── pyproject.toml                 # 项目配置
└── README.md                      # 使用文档
```

## 核心约定

### 异步优先
- 所有 IO 操作必须是 `async def`，禁止阻塞调用。
- 文件操作使用 `aiofiles`，目录操作使用 `aiofiles.os` 或 `os.scandir`。

### Result 类型
- 文件系统/Shell 方法**永不抛异常**，所有失败编码进返回的 `Result`。
- 使用 `ok(value)` / `err(AgentError)` 构造，`result.is_ok()` / `result.value` / `result.error` 访问。
- 边界检查优先 `get_or_throw`，可空值用 `get_or_undefined`。

### 错误处理
- 文件错误映射为 `FileError`（`not_found` / `permission_denied` 等 8 种码）。
- Shell 错误映射为 `ExecutionError`（`timeout` / `spawn_error` / `callback_error` 等）。
- 底层异常通过 `to_error()` 统一包装。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- Pydantic v2 `BaseModel` 对应 TS Interface，Discriminated Union 处理多态。
- `contextvars` 用于请求级状态。

### 路径
- 一律使用 `pathlib.Path` 或 `os.path`，禁止字符串拼接路径。
- 环境路径通过 `_resolve_path(cwd, path)` 解析（支持 `~`、`file://`、相对路径）。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。
- 对应 TS 实现的方法注释标注 `（对应 TS ``xxx``）` 便于溯源。

## 常用命令

```bash
# 类型检查（全项目 strict）
conda run -n pi_env python -m mypy src/pi_agent/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 运行单个测试文件
conda run -n pi_env python -m pytest tests/test_result.py -v

# 启动端到端测试（DeepSeek API；main.py 位于仓库根目录）
cd ..
conda run -n pi_env python main.py
```

## 工具开发指南

新增工具时遵循以下模式：

1. 在 `harness/tools/` 下新建模块，定义 `create<Xxx>Tool(env, ...) -> Tool` 工厂。
2. 工具 `execute` 闭包返回 `AgentToolResult`，错误用 `err(FileError(...))` 包装。
3. 所有工具注册在 `tools/__init__.py` 的导出中。
4. 为工具补充对应测试（参考 `tests/` 现有模式）。

## 任务完成约定

- 任务完成后更新 `doc/todo.md` 状态。
- 结构性改动需同步更新 `README.md` 与本文档。
