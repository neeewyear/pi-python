# Pi Coding Agent

编码助手产品层 — 对应 `@earendil-works/pi-coding-agent` TypeScript 实现。

## 项目概况

异步优先的 AI 编码助手产品层，提供完整的 CLI 交互界面和 Agent 会话管理：

- **CLI 界面** — 参数解析、模式选择（interactive / print / json / rpc）、配置管理
- **Agent 会话管理** — 会话创建/恢复/分叉、设置管理、运行时服务
- **工具系统** — Bash、文件读写、编辑、搜索（grep/find/ls）、HTML 导出
- **交互模式** — 基于 pi_tui 的终端 UI（流式消息、工具调用展示、斜杠命令、自动补全）
- **扩展框架** — 扩展加载器、运行器、事件系统
- **模型管理** — 模型注册表、解析器、运行时、成本计算
- **认证系统** — 多 provider 认证存储、OAuth 引导
- **速率限制** — HTTP 请求调度、缓存统计

## 依赖关系

```
pi-ai ──▶ pi-agent ──▶ pi-coding-agent
                         ▲
pi-session ──────────────┘
     ▲
pi-session-sqlite ───────┘
```

本包（pi-coding-agent）依赖 `pi-ai`（LLM API 层）、`pi-agent`（Agent 引擎）、`pi-session`（会话数据层）。

## 架构

```
pi_coding_agent/
├── __main__.py / main.py          # 主入口（CLI 参数解析 → 模式分派）
├── config.py                      # 应用配置（APP_NAME, VERSION, 目录路径）
├── migrations.py                  # 会话迁移
│
├── cli/                           # CLI 界面层
│   ├── args.py                    # 参数解析（15+ 参数）
│   ├── config_selector.py         # 配置选择器
│   ├── credential_print.py        # 凭证打印
│   ├── file_processor.py          # 文件处理
│   ├── initial_message.py         # 初始消息
│   ├── list_models.py             # 模型列表
│   ├── project_trust.py           # 项目信任
│   ├── session_picker.py          # 会话选择器
│   └── startup_ui.py              # 启动 UI
│
├── core/                          # 核心逻辑（40+ 模块）
│   ├── agent_session.py           # Agent 会话（主逻辑）
│   ├── agent_session_runtime.py   # 运行时服务
│   ├── agent_session_services.py  # 服务工厂
│   ├── session_manager.py         # 会话管理器
│   ├── settings_manager.py        # 设置管理器
│   ├── model_resolver.py / model_config.py / model_registry.py
│   ├── model_runtime.py / models_store.py
│   ├── provider_composer.py       # Provider 编排
│   ├── sdk.py                     # SDK 封装
│   ├── slash_commands.py          # 斜杠命令系统
│   ├── system_prompt.py           # 系统提示词
│   ├── skills.py                  # 技能管理
│   ├── bash_executor.py           # Bash 执行器
│   ├── event_bus.py               # 事件总线
│   ├── keybindings.py             # 快捷键绑定
│   ├── exec.py / http_dispatcher.py / telemetry.py
│   ├── compaction/                # 上下文压缩
│   ├── export_html/               # HTML 导出（ansi_to_html / templates / tool_renderer）
│   ├── extensions/                # 扩展框架（loader / runner / types / wrapper）
│   └── tools/                     # 工具系统（bash / edit / grep / find / ls / read / write / ...）
│
├── extensions/                    # 扩展实现
│   └── llama/                     # Llama 扩展（client / huggingface / provider / ui）
│
├── modes/                         # 运行模式
│   ├── print_mode.py              # 打印模式（文本/JSON 输出）
│   ├── json_event.py              # JSON 事件
│   ├── interactive/               # 交互式 TUI 模式
│   │   ├── pi_tui_mode.py         # 基于 pi_tui 的交互模式
│   │   └── theme/                 # 主题（syntax_highlight / theme）
│   └── rpc/                       # RPC 模式（jsonl / rpc_client / rpc_mode / rpc_types）
│
├── client/                        # 远程客户端
│   ├── remote_session.py
│   └── transcript.py
│
└── utils/                         # 工具函数
    ├── ansi.py / html.py / json.py / shell.py
    ├── git.py / mime.py / paths.py / sleep.py
    ├── fs_watch.py / frontmatter.py / open_browser.py
    ├── abort.py / deprecation.py / pi_user_agent.py
    └── ...
```

## 安装

```bash
conda activate pi_env

# 安装顺序（先装底层依赖，再装本包）
cd ai && pip install -e ".[dev]" && cd ..
cd session && pip install -e ".[dev]" && cd ..
cd agent && pip install -e ".[dev]" && cd ..
cd coding-agent && pip install -e ".[dev]" && cd ..
```

## 快速开始

```bash
# 启动交互模式（TUI）
conda run -n pi_env python -m pi_coding_agent

# 打印模式（直接对话）
echo "Hello!" | conda run -n pi_env python -m pi_coding_agent --print

# 查看帮助
conda run -n pi_env python -m pi_coding_agent --help

# 查看版本
conda run -n pi_env python -m pi_coding_agent --version

# 列出可用模型
conda run -n pi_env python -m pi_coding_agent --list-models
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_coding_agent/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 启动交互模式
conda run -n pi_env python -m pi_coding_agent

# 启动打印模式（非交互）
echo "Hello!" | conda run -n pi_env python -m pi_coding_agent --print
```

## 技术栈

- Python 3.11+
- Pydantic v2（数据模型）
- aiofiles（异步文件 I/O）
- orjson（高性能 JSON）
- pygments（代码语法高亮）
- pi_tui（差分渲染 TUI 库）
- pytest + pytest-asyncio（测试）

## 许可证

Apache 2.0