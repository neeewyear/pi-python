# Pi Session SQLite — 开发指南

## 项目介绍

SQLite 会话后端，对应 `@earendil-works/pi-session-sqlite` TypeScript 实现。实现 `pi-session` 的 `SessionStorage` / `SessionSearch` / `SessionRepository` 契约，提供写租约 fencing 并发保护、FTS5 全文搜索和分支缓存能力。

## 开发环境

```bash
conda activate pi_env

# 安装顺序（先装依赖，再装本包）
cd ai && pip install -e ".[dev]" && cd ..
cd session && pip install -e ".[dev]" && cd ..
cd agent && pip install -e ".[dev]" && cd ..
cd session-sqlite && pip install -e ".[dev]" && cd ..
```

## 目录结构

```
session-sqlite/
├── src/pi_session_sqlite/          # 源码
│   ├── __init__.py                 # 导出面（对应 TS sqlite/index.ts）
│   ├── _adapter.py                 # aiosqlite 适配器
│   ├── _migrations.py              # 自动迁移/建表
│   ├── _repo.py                    # SqliteSessionRepository 主入口
│   ├── _search.py                  # SqliteSessionSearch（FTS5 搜索）
│   ├── _types.py                   # 内部类型定义
│   ├── _branch_cache.py            # 分支查询缓存
│   │
│   └── storage/                    # SQLite 存储层
│       ├── sessions.py / lanes.py / entries.py / records.py
│       ├── branch_entries.py / branch_tips.py
│       ├── leases.py               # 写租约（fencing）
│       ├── session_sequences.py / session_stats.py
│       └── facts.py
├── tests/                          # 测试
├── pyproject.toml                  # 项目配置
└── README.md                       # 使用文档
```

## 核心约定

### 异步优先
- 所有数据库操作使用 `aiosqlite`，必须是 `async def`。
- 连接池通过 `create_aiosqlite_factory()` 工厂创建。

### 写租约（Writer Lease）
- 使用 fencing 令牌机制防止并发写入。
- 每个写操作前检查租约有效性，过期需重新获取。
- 租约 TTL 通过 `SqliteWriterLeaseOptions.ttl_ms` 配置（默认 60s）。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- 注意：`session-sqlite` 使用 `setuptools.backends._legacy:_Backend` 构建后端，不支持 `pydantic.mypy` 插件。

### 数据库迁移
- 迁移自动在首次连接时执行（`_migrations.py`）。
- 迁移版本号存储在 `_migrations` 表中。
- 新增迁移需追加到 `_migrations.py` 的迁移列表。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。
- 对应 TS 实现的方法注释标注 `（对应 TS ``xxx``）` 便于溯源。

## 常用命令

```bash
# 类型检查（strict；注意 pydantic 插件不可用）
conda run -n pi_env python -m mypy src/pi_session_sqlite/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 任务完成约定

- 任务完成后更新 `todo/` 下的规划文档状态。
- 结构性改动需同步更新 `README.md` 与本文档。