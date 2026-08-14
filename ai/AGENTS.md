# Pi AI — 开发指南

## 项目介绍

Unified LLM API layer for pi，对应 `@earendil-works/pi` 的 `packages/ai/` TypeScript 实现。提供 39 家 Provider 的统一调用接口、认证/OAuth 系统、模型注册表和流式 API。作为底层基础设施，pi-ai 不依赖其他 pi 包。

## 开发环境

```bash
conda activate pi_env

# 安装 pi-ai（无其他 pi 包依赖）
cd ai
pip install -e ".[dev]"
```

## 目录结构

```
ai/
├── src/pi_ai/                    # 源码
│   ├── __init__.py               # 导出面
│   ├── types.py                  # 核心类型（Message / Tool / ContentBlock 等）
│   ├── models.py / models_data.py / model_catalog.py
│   ├── models_store.py           # 模型存储
│   ├── stream_fn.py              # 流函数注册表
│   ├── env_api_keys.py           # 环境变量 API 密钥检测
│   │
│   ├── api/                      # API 实现层（16 个子模块）
│   │   ├── lazy.py               # 懒加载流式 API 核心
│   │   ├── openai_responses.py / anthropic_messages.py
│   │   ├── google_generative_ai.py / google_vertex.py
│   │   ├── ... 各 provider 流式 API
│   │   └── transform_messages.py # 消息转换
│   │
│   ├── auth/                     # 认证层
│   │   ├── types.py / resolve.py / credential_store.py
│   │   └── oauth/                # 10 家 OAuth 实现
│   │
│   ├── providers/                # 39 家 Provider 实现
│   │   ├── registry.py
│   │   ├── anthropic.py / openai.py / google.py / ...
│   │   └── images/               # 图片能力注册
│   │
│   └── utils/                    # 工具函数（20+ 模块）
│
├── tests/                        # 测试
├── pyproject.toml                # 项目配置
└── README.md                     # 使用文档
```

## 核心约定

### 异步优先
- 所有 IO 操作必须是 `async def`，禁止阻塞调用。
- 流式响应使用 `async for event in stream:` 模式。

### 类型安全
- 全项目遵守 `mypy --strict`，零 `Any`。
- Pydantic v2 `BaseModel` 对应 TS Interface，Discriminated Union 处理多态（`ContentBlock` / `Message` 等）。
- `py.typed` 已随包发布，外部消费方可获得类型信息。

### Provider 接入规范
- 每个 provider 在 `providers/` 下有一个模块，定义 `createXxxProvider()` 工厂。
- 模型定义放在对应的 `_models.py` 文件中。
- 流式 API 实现放在 `api/` 下，通过 `_lazy.py` 模式实现懒加载。
- 在 `providers/registry.py` 中注册新 provider。

### 认证系统
- API 密钥优先从环境变量读取（`env_api_keys.py`）。
- OAuth 流程通过 `auth/oauth/` 下的设备码流程实现。
- 凭证存储使用 `auth/credential_store.py`。

### 代码风格
- PEP 8：函数/变量 `snake_case`，类 `PascalCase`。
- 公共方法必须有 docstring（中文）。

## 依赖关系

```
pi-ai（底层，无 pi 包依赖）
  ├──▶ pi-agent（消费 pi_ai.types / stream_fn / models）
  └──▶ pi-coding-agent（消费 pi_ai.models / types）
```

除 `pydantic`、`httpx`、`openai` 外无外部依赖。

## 常用命令

```bash
# 类型检查（strict）
conda run -n pi_env python -m mypy src/pi_ai/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v

# 运行单个测试文件
conda run -n pi_env python -m pytest tests/test_types.py -v
```

## 新增 Provider 指南

1. 在 `providers/` 下新建 `xxx.py` 和 `xxx_models.py`，定义 `createXxxProvider()` 工厂。
2. 在 `providers/registry.py` 中注册。
3. 在 `api/` 下新建流式 API 实现（参考 `openai_responses.py` 模式）。
4. 如需 OAuth，在 `auth/oauth/` 下添加设备码流程。
5. 补充模型注册（`models_data.py` 或 `_models.py`）。

## 任务完成约定

- 任务完成后更新 `todo/` 下的规划文档状态。
- 结构性改动需同步更新 `README.md` 与本文档。