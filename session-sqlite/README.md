# Pi Session SQLite

SQLite 会话后端 — 对应 `@earendil-works/pi-session-sqlite` TypeScript 实现。

## 项目概况

异步优先的 SQLite 会话存储后端，实现 `pi-session` 的 `SessionStorage` / `SessionRepo` 契约：

- **写租约（Writer Lease）** — 基于 fencing 机制的并发写保护，防止多进程同时写入
- **FTS5 全文搜索** — 基于 SQLite FTS5 的会话搜索实现
- **分支缓存** — 分支查询结果缓存，避免重复扫描
- **会话迁移** — 自动迁移/创建表结构
- **类型安全** — 全量 Pydantic v2 模型 + mypy 严格模式

## 依赖关系

```
pi-session-sqlite
  ├──▶ pi-session（会话数据层契约）
  └──▶ pi-agent（叶子模块类型）
```

实现 `pi-session` 的 `SessionStorage` / `SessionSearch` / `SessionRepository` 接口。

## 架构

```
pi_session_sqlite/
├── __init__.py                    # 导出面（对应 TS sqlite/index.ts）
├── _adapter.py                    # aiosqlite 适配器（create_aiosqlite_factory）
├── _migrations.py                 # 自动迁移/建表
├── _repo.py                       # SqliteSessionRepository（主入口）
├── _search.py                     # SqliteSessionSearch（FTS5 搜索）
├── _types.py                      # 内部类型定义（SqliteDatabase / SqliteStatement 等）
├── _branch_cache.py               # 分支查询缓存
│
└── storage/                       # SQLite 存储层
    ├── __init__.py
    ├── sessions.py                # 会话元数据存储
    ├── lanes.py                   # Lane 管理
    ├── entries.py                 # 条目存储
    ├── records.py                 # 记录存储
    ├── branch_entries.py          # 分支条目
    ├── branch_tips.py             # 分支指针
    ├── leases.py                  # 写租约（fencing）
    ├── session_sequences.py       # 会话序列
    ├── session_stats.py           # 会话统计
    └── facts.py                   # 事实存储
```

## 安装

```bash
conda activate pi_env

# 安装顺序（先装依赖，再装本包）
cd ai && pip install -e ".[dev]" && cd ..
cd session && pip install -e ".[dev]" && cd ..
cd agent && pip install -e ".[dev]" && cd ..
cd session-sqlite && pip install -e ".[dev]" && cd ..
```

## 快速开始

```python
from pi_session_sqlite import (
    SqliteSessionRepository, SqliteSessionRepositoryOptions,
    SqliteSessionRepositoryEnv, SqliteSessionCreateOptions,
    SqliteWriterLeaseOptions, create_aiosqlite_factory,
)

# 创建 SQLite 会话仓库
repo = SqliteSessionRepository(SqliteSessionRepositoryOptions(
    env=SqliteSessionRepositoryEnv(),
    sqlite=create_aiosqlite_factory(),
    database_path="/tmp/sessions.db",
    writer_lease=SqliteWriterLeaseOptions(ttl_ms=60_000),
))

# 创建新会话
session = await repo.create(SqliteSessionCreateOptions(cwd="/tmp"))

# 获取会话元数据
meta = await session.get_metadata()
print(f"Session: {meta.id}")

# 搜索会话
from pi_session_sqlite import SqliteSessionSearchOptions, create_sqlite_session_search
search = create_sqlite_session_search(repo)
results = await search.search("hello", SqliteSessionSearchOptions())

# 关闭
await repo.close()
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_session_sqlite/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 技术栈

- Python 3.11+
- aiosqlite（异步 SQLite）
- Pydantic v2（数据模型）
- mypy --strict（类型安全）

## 许可证

Apache 2.0