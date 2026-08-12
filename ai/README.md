# Pi AI

Unified LLM API layer for pi — 对应 `@earendil-works/pi` 的 `packages/ai/` TypeScript 实现。

## 项目概况

异步优先的 LLM API 统一调用层，提供：

- **39 家 Provider 工厂** — Anthropic、OpenAI、Google、DeepSeek、Mistral、Groq、Fireworks、Together AI、OpenRouter、GitHub Copilot、Amazon Bedrock、Azure OpenAI、Cloudflare Workers AI、HuggingFace、NVIDIA 等
- **认证/OAuth 系统** — 环境变量、凭证存储、OAuth 设备码流程（Anthropic、GitHub Copilot、OpenAI Codex、OpenRouter、Kimi、xAI、Radius）
- **模型注册表** — `ModelCatalog` 模型目录、`ModelsImpl` 模型查询/刷新、模型成本计算
- **流式 API** — 统一 `StreamFn` 接口，支持流式文本/工具调用/思考过程 delta
- **图片生成** — 多 provider 图片生成 API（OpenAI、OpenRouter、AWS Bedrock 等）
- **类型安全** — 全量 Pydantic v2 模型 + mypy 严格模式

## 依赖关系

```
pi-ai（底层）
  │
  ├──▶ pi-agent（消费 pi_ai.types / stream_fn / models）
  └──▶ pi-coding-agent（消费 pi_ai.models / types）
```

作为底层基础设施，pi-ai 不依赖其他 pi 包，被上层模块消费。

## 架构

```
pi_ai/
├── types.py                        # 核心类型（Message / Tool / ContentBlock / Usage 等）
├── models.py / models_data.py      # 模型定义与注册表
├── model_catalog.py                # 模型目录扁平化
├── models_store.py                 # 模型存储（InMemoryModelsStore）
├── env_api_keys.py                 # 环境变量 API 密钥检测
├── stream_fn.py                    # 流函数注册表
│
├── api/                            # API 实现层
│   ├── lazy.py                     # 懒加载流式 API 核心
│   ├── openai_responses.py / _lazy.py / _shared.py  # OpenAI 兼容 API
│   ├── anthropic_messages.py / _lazy.py
│   ├── google_generative_ai.py / _lazy.py
│   ├── google_vertex.py / _lazy.py
│   ├── deepseek_provider.py
│   ├── bedrock_converse_stream.py / _lazy.py
│   ├── azure_openai_responses.py / _lazy.py
│   ├── mistral_conversations.py / _lazy.py
│   ├── openai_codex_responses.py / _lazy.py
│   ├── openai_completions.py / _lazy.py
│   ├── openrouter_images.py / _lazy.py
│   ├── pi_messages.py / _lazy.py
│   ├── transform_messages.py       # 消息转换
│   ├── simple_options.py           # SimpleStreamOptions
│   └── constrained_sampling.py     # 约束采样
│
├── auth/                           # 认证层
│   ├── types.py / context.py / resolve.py / helpers.py
│   ├── credential_store.py
│   └── oauth/                      # OAuth 设备码流程
│       ├── anthropic.py / github_copilot.py / openai_codex.py
│       ├── openrouter.py / kimi_coding.py / xai.py / radius.py
│       ├── device_code.py / pkce.py / oauth_page.py / load.py
│
├── providers/                      # 39 家 Provider 实现
│   ├── registry.py                 # 注册表
│   ├── images/                     # 图片能力注册
│   ├── anthropic.py / openai.py / google.py / deepseek.py
│   ├── mistral.py / groq.py / fireworks.py / together.py
│   ├── openrouter.py / github_copilot.py / xai.py
│   ├── amazon_bedrock.py / azure_openai_responses.py
│   ├── cloudflare_workers_ai.py / cloudflare_stream.py
│   ├── huggingface.py / nvidia.py / cerebras.py
│   ├── baseten.py / faux.py
│   ├── kimi_coding.py / minimax.py / moonshotai.py
│   ├── qwen_token_plan.py / xiaomi.py / zai.py
│   ├── opencode.py / radius.py / vercel_ai_gateway.py
│   └── 各 provider 对应的 _models.py 模型定义
│
├── utils/                          # 工具函数
│   ├── event_stream.py / retry.py / provider_retry.py
│   ├── json_parse.py / hash.py / uuid.py
│   ├── text.py / sanitize_unicode.py / overflow.py
│   ├── headers.py / provider_env.py / error_body.py
│   ├── diagnostics.py / abort.py / deferred_tools.py
│   ├── estimate.py / validation.py / typebox_helpers.py
│   └── node_http_proxy.py
│
├── legacy_api_aliases.py           # 兼容别名
├── images.py                       # 图片 API
├── image_models_generated.py       # 图片模型生成
├── oauth.py / bun_oauth.py         # OAuth 入口
├── compat.py / cli.py              # 兼容与 CLI
├── bedrock_provider.py / deepseek_provider.py  # 扩展
└── session_resources.py            # 会话资源
```

## 安装

```bash
conda activate pi_env

# 安装 pi-ai（无其他 pi 包依赖）
cd ai
pip install -e ".[dev]"
```

## 快速开始

```python
from pi_ai import create_models, ModelsImpl
from pi_ai.types import UserMessage, AssistantMessage

# 创建模型实例
models = create_models()
impl = ModelsImpl(models)

# 查询可用模型
records = await impl.list_models()
for r in records:
    print(f"{r.id} ({r.provider})")

# 流式调用
from pi_ai.types import ProviderRequestOptions

stream = await impl.stream(
    "deepseek/deepseek-v4-flash",
    [UserMessage(content=[{"type": "text", "text": "Hello!"}])],
    ProviderRequestOptions(api_key="..."),
)
async for event in stream:
    print(event)
```

## 开发

```bash
# 类型检查
conda run -n pi_env python -m mypy src/pi_ai/ --strict

# 运行测试
conda run -n pi_env python -m pytest tests/ -v
```

## 技术栈

- Python 3.11+
- Pydantic v2（数据模型，Discriminated Union）
- httpx（HTTP 客户端）
- openai（OpenAI 兼容 API 客户端）
- pytest + pytest-asyncio（测试）

## 许可证

Apache 2.0