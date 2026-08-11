"""模型目录数据（对应 ``models.generated.ts`` + ``image-models.generated.ts``）。

TS 侧由 ``scripts/generate-models.ts`` 自动生成，Python 侧在 Provider 迁移
（15.6 阶段）后使用占位数据。模型数据需从 JSON 数据文件加载后填充。
"""

from __future__ import annotations

from typing import Any

# ---------------------------------------------------------------------------
# 模型目录类型
# ---------------------------------------------------------------------------

# 每个 provider 的模型字典：{model_id: model_record_dict}
ProviderModelsDict = dict[str, dict[str, Any]]

# 完整模型目录：{provider_id: {model_id: model_record_dict}}
ModelsCatalog = dict[str, ProviderModelsDict]

# ---------------------------------------------------------------------------
# 模型目录（对应 TS ``MODELS`` 常量）
# ---------------------------------------------------------------------------

# TODO: 从各 provider 的 .models.py 模块导入实际模型数据
# 当前使用占位空字典，等模型数据 JSON 文件迁移后填充

from .providers.amazon_bedrock_models import AMAZON_BEDROCK_MODELS
from .providers.ant_ling_models import ANT_LING_MODELS
from .providers.anthropic_models import ANTHROPIC_MODELS
from .providers.azure_openai_responses_models import AZURE_OPENAI_RESPONSES_MODELS
from .providers.baseten_models import BASETEN_MODELS
from .providers.cerebras_models import CEREBRAS_MODELS
from .providers.cloudflare_ai_gateway_models import CLOUDFLARE_AI_GATEWAY_MODELS
from .providers.cloudflare_workers_ai_models import CLOUDFLARE_WORKERS_AI_MODELS
from .providers.deepseek_models import DEEPSEEK_MODELS
from .providers.fireworks_models import FIREWORKS_MODELS
from .providers.github_copilot_models import GITHUB_COPILOT_MODELS
from .providers.google_models import GOOGLE_MODELS
from .providers.google_vertex_models import GOOGLE_VERTEX_MODELS
from .providers.groq_models import GROQ_MODELS
from .providers.huggingface_models import HUGGINGFACE_MODELS
from .providers.kimi_coding_models import KIMI_CODING_MODELS
from .providers.minimax_cn_models import MINIMAX_CN_MODELS
from .providers.minimax_models import MINIMAX_MODELS
from .providers.mistral_models import MISTRAL_MODELS
from .providers.moonshotai_cn_models import MOONSHOTAI_CN_MODELS
from .providers.moonshotai_models import MOONSHOTAI_MODELS
from .providers.nvidia_models import NVIDIA_MODELS
from .providers.openai_codex_models import OPENAI_CODEX_MODELS
from .providers.openai_models import OPENAI_MODELS
from .providers.opencode_go_models import OPENCODE_GO_MODELS
from .providers.opencode_models import OPENCODE_MODELS
from .providers.openrouter_models import OPENROUTER_MODELS
from .providers.qwen_token_plan_cn_models import QWEN_TOKEN_PLAN_CN_MODELS
from .providers.qwen_token_plan_models import QWEN_TOKEN_PLAN_MODELS
from .providers.together_models import TOGETHER_MODELS
from .providers.vercel_ai_gateway_models import VERCEL_AI_GATEWAY_MODELS
from .providers.xai_models import XAI_MODELS
from .providers.xiaomi_models import XIAOMI_MODELS
from .providers.xiaomi_token_plan_ams_models import XIAOMI_TOKEN_PLAN_AMS_MODELS
from .providers.xiaomi_token_plan_cn_models import XIAOMI_TOKEN_PLAN_CN_MODELS
from .providers.xiaomi_token_plan_sgp_models import XIAOMI_TOKEN_PLAN_SGP_MODELS
from .providers.zai_coding_cn_models import ZAI_CODING_CN_MODELS
from .providers.zai_models import ZAI_MODELS

MODELS: ModelsCatalog = {
    "amazon-bedrock": AMAZON_BEDROCK_MODELS,
    "ant-ling": ANT_LING_MODELS,
    "anthropic": ANTHROPIC_MODELS,
    "azure-openai-responses": AZURE_OPENAI_RESPONSES_MODELS,
    "baseten": BASETEN_MODELS,
    "cerebras": CEREBRAS_MODELS,
    "cloudflare-ai-gateway": CLOUDFLARE_AI_GATEWAY_MODELS,
    "cloudflare-workers-ai": CLOUDFLARE_WORKERS_AI_MODELS,
    "deepseek": DEEPSEEK_MODELS,
    "fireworks": FIREWORKS_MODELS,
    "github-copilot": GITHUB_COPILOT_MODELS,
    "google": GOOGLE_MODELS,
    "google-vertex": GOOGLE_VERTEX_MODELS,
    "groq": GROQ_MODELS,
    "huggingface": HUGGINGFACE_MODELS,
    "kimi-coding": KIMI_CODING_MODELS,
    "minimax": MINIMAX_MODELS,
    "minimax-cn": MINIMAX_CN_MODELS,
    "mistral": MISTRAL_MODELS,
    "moonshotai": MOONSHOTAI_MODELS,
    "moonshotai-cn": MOONSHOTAI_CN_MODELS,
    "nvidia": NVIDIA_MODELS,
    "openai": OPENAI_MODELS,
    "openai-codex": OPENAI_CODEX_MODELS,
    "opencode": OPENCODE_MODELS,
    "opencode-go": OPENCODE_GO_MODELS,
    "openrouter": OPENROUTER_MODELS,
    "qwen-token-plan": QWEN_TOKEN_PLAN_MODELS,
    "qwen-token-plan-cn": QWEN_TOKEN_PLAN_CN_MODELS,
    "together": TOGETHER_MODELS,
    "vercel-ai-gateway": VERCEL_AI_GATEWAY_MODELS,
    "xai": XAI_MODELS,
    "xiaomi": XIAOMI_MODELS,
    "xiaomi-token-plan-ams": XIAOMI_TOKEN_PLAN_AMS_MODELS,
    "xiaomi-token-plan-cn": XIAOMI_TOKEN_PLAN_CN_MODELS,
    "xiaomi-token-plan-sgp": XIAOMI_TOKEN_PLAN_SGP_MODELS,
    "zai": ZAI_MODELS,
    "zai-coding-cn": ZAI_CODING_CN_MODELS,
}
"""聚合模型目录（对应 TS ``MODELS``）。

键为 provider ID，值为 {model_id: model_record_dict}。
"""

# ---------------------------------------------------------------------------
# 图片模型目录（对应 TS ``IMAGE_MODELS`` 常量）
# ---------------------------------------------------------------------------

IMAGE_MODELS: ModelsCatalog = {}
"""图片生成模型目录（对应 TS ``IMAGE_MODELS``）。

当前仅 ``openrouter`` 一个 provider，约 50 个模型。
"""
