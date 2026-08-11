# Pi Session — Agent 开发指南

## 项目介绍

Pi Agent 的 Session 数据层独立包（对应 `@earendil-works/pi-agent-core` 的 `harness/session/`）。提供 Entry/LaneRecord 数据模型、SessionStorage/SessionRepo 存储契约、Session 树视图、上下文构建与会话搜索。从 `pi_agent` 包独立，跨包依赖其叶子模块（`pi_agent.types` / `result` / `uuid7` / `harness.messages` / `harness.types`）。

## 开发环境

```bash
conda activate pi_env

# 先装 session（依赖 pi-agent），再刷新 agent
cd session
pip install -e ".[dev]"
cd ../agent
pip install -e ".[dev]"
```

## 目录结构

```
session/
├── src/pi_session/            # 源码
│   ├── __init__.py            # 导出面（对应 TS session/index.ts）
│   ├── types.py               # Entry(7种) / LaneRecord(9种) / 存储契约
│   ├── session.py             # Session 类（树视图，绑定 main lane）
│   ├── memory.py              # InMemorySessionStorage / InMemorySessionRepo
│   ├── context.py             # build_session_context / 消息投影
│   └── search.py              # ScanningSessionSearch
├── tests/                     # 测试（pytest + pytest-asyncio）
├── pyproject.toml             # 项目配置
└── README.md                  # 使用文档
```

## 核心约定

### 异步优先
- 所有 IO 操作必须是 `async def`，禁止阻塞调用。

### Result 类型
- 存储契约方法**永不抛异常**（除 `SessionError` 代表的逻辑错误），所有后端失败编码为 `SessionError(code="storage")`。
- 使用 `pi_agent.result` 的 `ok(value)` / `err(AgentError)` 构造与解包。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- Pydantic v2 `BaseModel` 对应 TS Interface，Discriminated Union 处理多态（`Entry` / `LaneRecord` / `LogItem`）。
- `py.typed` 已随包发布，外部消费方可获得类型信息。

### 依赖纪律
- **只允许**从 `pi_agent` 导入叶子模块：`pi_agent.types` / `pi_agent.result` / `pi_agent.uuid7` / `pi_agent.harness.messages` / `pi_agent.harness.types`。
- **禁止**导入 `pi_agent.harness.compaction` / `pi_agent.harness.agent_harness` 等反向依赖 session 的模块——否则形成跨包循环。
- 新增基础类型时优先考虑是否应下沉到本包。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。
- 对应 TS 实现的方法注释标注 `（对应 TS ``xxx``）` 便于溯源。

## 常用命令

```bash
# 类型检查（strict）
conda run -n pi_env python -m mypy src/pi_session/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 运行单个测试文件
conda run -n pi_env python -m pytest tests/test_session.py -v
```

## 任务完成约定

- 任务完成后更新 `todo/` 下的规划文档状态。
- 结构性改动需同步更新 `README.md` 与本文档。
