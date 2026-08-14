# Pi Coding Agent — 开发指南

## 项目介绍

编码助手产品层，对应 `@earendil-works/pi-coding-agent` TypeScript 实现。提供完整的 CLI 交互界面、Agent 会话管理、工具系统、扩展框架和多种运行模式（interactive / print / json / rpc）。依赖 `pi-ai`、`pi-agent`、`pi-session` 三个底层包。

## 开发环境

```bash
conda activate pi_env

# 安装顺序（先装底层依赖，再装本包）
cd ai && pip install -e ".[dev]" && cd ..
cd session && pip install -e ".[dev]" && cd ..
cd agent && pip install -e ".[dev]" && cd ..
cd coding-agent && pip install -e ".[dev]" && cd ..
```

## 目录结构

```
coding-agent/
├── src/pi_coding_agent/                      # 源码（~57,985 行）
│   ├── __main__.py / main.py                 # 主入口（CLI 参数解析 → 模式分派）
│   ├── config.py                             # 应用配置
│   ├── migrations.py                         # 会话迁移
│   │
│   ├── cli/                                  # CLI 界面层
│   │   ├── args.py                           # 参数解析（15+ 参数）
│   │   ├── config_selector.py                # 配置选择器
│   │   ├── credential_print.py               # 凭证打印
│   │   ├── file_processor.py                 # 文件处理
│   │   ├── initial_message.py                # 初始消息
│   │   ├── list_models.py                    # 模型列表
│   │   ├── project_trust.py                  # 项目信任
│   │   ├── session_picker.py                 # 会话选择器
│   │   └── startup_ui.py                     # 启动 UI
│   │
│   ├── core/                                 # 核心逻辑
│   │   ├── agent_session.py                  # Agent 会话主逻辑
│   │   ├── agent_session_runtime.py          # 运行时服务
│   │   ├── agent_session_services.py         # 服务工厂
│   │   ├── session_manager.py                # 会话管理器
│   │   ├── settings_manager.py               # 设置管理器
│   │   ├── model_resolver.py / model_config.py / model_registry.py
│   │   ├── model_runtime.py / models_store.py
│   │   ├── provider_composer.py              # Provider 编排
│   │   ├── sdk.py                            # SDK 封装
│   │   ├── slash_commands.py                 # 斜杠命令系统
│   │   ├── system_prompt.py                  # 系统提示词
│   │   ├── skills.py                         # 技能管理
│   │   ├── bash_executor.py                  # Bash 执行器
│   │   ├── event_bus.py                      # 事件总线
│   │   ├── keybindings.py                    # 快捷键绑定
│   │   ├── compaction/                       # 上下文压缩
│   │   ├── export_html/                      # HTML 导出
│   │   ├── extensions/                       # 扩展框架
│   │   └── tools/                            # 工具系统
│   │       ├── bash.py / edit.py / read.py / write.py
│   │       ├── grep.py / find.py / ls.py
│   │       ├── edit_diff.py / file_mutation_queue.py
│   │       ├── path_utils.py / truncate.py / render_utils.py
│   │       └── tool_definition_wrapper.py / output_accumulator.py
│   │
│   ├── extensions/                           # 扩展实现
│   │   └── llama/                            # Llama 扩展
│   │
│   ├── modes/                                # 运行模式
│   │   ├── print_mode.py                     # 打印模式（text/json）
│   │   ├── json_event.py                     # JSON 事件
│   │   ├── interactive/                      # 交互式 TUI 模式
│   │   │   ├── pi_tui_mode.py                # 基于 pi_tui 的交互模式
│   │   │   └── theme/                        # 主题
│   │   └── rpc/                              # RPC 模式
│   │
│   ├── client/                               # 远程客户端
│   └── utils/                                # 工具函数
│
├── tests/                                    # 测试
├── pyproject.toml                            # 项目配置
└── README.md                                 # 使用文档
```

## 核心约定

### 异步优先
- 所有 IO 操作必须是 `async def`，禁止阻塞调用。
- 文件操作使用 `aiofiles`，HTTP 使用 `httpx`。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- Pydantic v2 `BaseModel` 对应 TS Interface。
- `py.typed` 已随包发布。

### 模式分派
- 主入口在 `main.py` 中解析 CLI 参数后分派到对应模式。
- 模式选择逻辑：`rpc` > `json` > `print` > `interactive`（自动检测 TTY）。
- 新增模式需在 `modes/` 下创建子模块，并在 `main.py` 中添加分派逻辑。

### 工具系统
- 工具定义在 `core/tools/` 下，每个工具一个模块。
- 工具遵循 `ToolDefinition` 协议，提供 `name`、`description`、`parameters` 和 `execute` 方法。
- 新增工具需在 `core/tools/__init__.py` 中导出。

### 扩展框架
- 扩展通过 `core/extensions/` 框架加载，支持 `runner` 事件钩子。
- 扩展实现放在 `extensions/` 目录下（如 `extensions/llama/`）。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。

## 常用命令

```bash
# 类型检查（strict）
conda run -n pi_env python -m mypy src/pi_coding_agent/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 启动交互模式
conda run -n pi_env python -m pi_coding_agent

# 打印模式
echo "Hello!" | conda run -n pi_env python -m pi_coding_agent --print

# 设置模型
conda run -n pi_env python -m pi_coding_agent --model deepseek/deepseek-v4-flash
```

## 任务完成约定

- 任务完成后更新 `todo/` 下的规划文档状态。
- 结构性改动需同步更新 `README.md` 与本文档。