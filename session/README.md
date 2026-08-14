# Pi Session

Pi Agent 的 Session 数据层独立包。

## 项目概况

异步优先的会话数据层，从 `pi_agent` 包中独立出来，提供：

- **Entry 数据模型** — 7 种会话树条目（message / model_change / thinking_level_change / active_tools_change / compaction / branch_summary / custom）
- **LaneRecord 记录模型** — 9 种 lane 操作日志（operation_started / step_attempt / tool_started / usage 等）
- **SessionStorage / SessionTree / SessionRepo 契约** — 存储后端抽象（当前实现 `InMemorySessionStorage` / `InMemorySessionRepo`）
- **Session 树视图** — 绑定 main lane 的查询/追加/lane 管理便捷接口
- **上下文构建** — `build_session_context`（压缩边界 / 分支摘要 / 状态推导）
- **会话搜索** — `ScanningSessionSearch` 全量扫描实现

## 依赖关系

```
pi-session ──依赖──▶ pi_agent（叶子模块：types / result / uuid7 / harness.messages / harness.types）
pi_agent ──依赖──▶ pi-session（harness.compaction / harness.agent_harness 消费 session）
```

依赖方向为**跨包叶子依赖**：`pi_session` 只从 `pi_agent` 导入不依赖 session 的叶子模块，两包静态无环。

## 架构

```
session/
├── pyproject.toml            # pi-session 包配置（mypy strict）
└── src/pi_session/
    ├── __init__.py           # 导出面
    ├── types.py              # Entry(7种) / LaneRecord(9种) / 存储契约（551 行）
    ├── session.py            # Session 类（树视图，绑定 main lane）
    ├── memory.py             # InMemorySessionStorage / InMemorySessionRepo
    ├── context.py            # build_session_context / 消息投影
    └── search.py             # ScanningSessionSearch
```

## 安装

```bash
conda activate pi_env

# 先装 session（依赖 pi-agent，需已在环境中），再刷新 agent
cd session
pip install -e ".[dev]"
cd ../agent
pip install -e ".[dev]"
```

## 快速开始

```python
from pi_session import InMemorySessionStorage, Session
from pi_session.types import MessageEntry, NewRecord

# 创建会话
storage = InMemorySessionStorage({"id": "session", "created_at": 1})
session = Session(storage)

# 追加条目
entry = await session.append_entry(
    MessageEntry.model_validate({
        "id": "entry-1",
        "lane": "main",
        "message": {"role": "user", "content": "hello"},
    }),
    "main",
)

# 追加记录
record = await storage.append_record(NewRecord(type="operation_started", ...), "main")

# 查询
entries = await session.find_entries_on_branch({"order": "oldestFirst"})
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_session/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 技术栈

- Python 3.11+
- Pydantic v2（数据模型，Discriminated Union）
- orjson（高性能 JSON）
- pytest + pytest-asyncio（测试）

## 许可证

Apache 2.0