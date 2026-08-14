"""环境变量 API Key 解析。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .types import KnownProvider, ProviderEnv
from .utils.provider_env import get_provider_env_value

ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"

_cached_vertex_adc_credentials_exists: bool | None = None


def _has_vertex_adc_credentials(env: ProviderEnv | None = None) -> bool:
    """检查是否存在 Vertex ADC 凭据。"""
    explicit_credentials_path = env.get("GOOGLE_APPLICATION_CREDENTIALS") if env else None
    if explicit_credentials_path:
        return Path(explicit_credentials_path).exists()

    global _cached_vertex_adc_credentials_exists
    if _cached_vertex_adc_credentials_exists is None:
        gac_path = get_provider_env_value("GOOGLE_APPLICATION_CREDENTIALS", env)
        if gac_path:
            _cached_vertex_adc_credentials_exists = Path(gac_path).exists()
        else:
            # 回退到默认 ADC 路径
            adc_path = Path.home() / ".config" / "gcloud" / "application_default_credentials.json"
            _cached_vertex_adc_credentials_exists = adc_path.exists()
    return _cached_vertex_adc_credentials_exists or False


def _get_api_key_env_vars(provider: str) -> list[str] | None:
    """获取 provider 的 API Key 环境变量名列表。"""
    if provider == "github-copilot":
        return ["COPILOT_GITHUB_TOKEN"]

    if provider == "anthropic":
        return [ANTHROPIC_AUTH_TOKEN_ENV, ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV]

    env_map: dict[str, str] = {
        "ant-ling": "ANT_LING_API_KEY",
        "qwen-token-plan": "QWEN_TOKEN_PLAN_API_KEY",
        "qwen-token-plan-cn": "QWEN_TOKEN_PLAN_CN_API_KEY",
        "openai": "OPENAI_API_KEY",
        "azure-openai-responses": "AZURE_OPENAI_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
        "deepseek": "DEEPSEEK_API_KEY",
        "google": "GEMINI_API_KEY",
        "google-vertex": "GOOGLE_CLOUD_API_KEY",
        "groq": "GROQ_API_KEY",
        "cerebras": "CEREBRAS_API_KEY",
        "xai": "XAI_API_KEY",
        "radius": "RADIUS_API_KEY",
        "openrouter": "OPENROUTER_API_KEY",
        "vercel-ai-gateway": "AI_GATEWAY_API_KEY",
        "zai": "ZAI_API_KEY",
        "zai-coding-cn": "ZAI_CODING_CN_API_KEY",
        "mistral": "MISTRAL_API_KEY",
        "minimax": "MINIMAX_API_KEY",
        "minimax-cn": "MINIMAX_CN_API_KEY",
        "moonshotai": "MOONSHOT_API_KEY",
        "moonshotai-cn": "MOONSHOT_API_KEY",
        "huggingface": "HF_TOKEN",
        "fireworks": "FIREWORKS_API_KEY",
        "together": "TOGETHER_API_KEY",
        "baseten": "BASETEN_API_KEY",
        "opencode": "OPENCODE_API_KEY",
        "opencode-go": "OPENCODE_API_KEY",
        "kimi-coding": "KIMI_API_KEY",
        "cloudflare-workers-ai": "CLOUDFLARE_API_KEY",
        "cloudflare-ai-gateway": "CLOUDFLARE_API_KEY",
        "xiaomi": "XIAOMI_API_KEY",
        "xiaomi-token-plan-cn": "XIAOMI_TOKEN_PLAN_CN_API_KEY",
        "xiaomi-token-plan-ams": "XIAOMI_TOKEN_PLAN_AMS_API_KEY",
        "xiaomi-token-plan-sgp": "XIAOMI_TOKEN_PLAN_SGP_API_KEY",
    }

    env_var = env_map.get(provider)
    return [env_var] if env_var else None


def find_env_keys(provider: str, env: ProviderEnv | None = None) -> list[str] | None:
    """查找可为 provider 提供 API Key 的已配置环境变量。

    仅报告实际的 API Key 变量，排除环境凭证源（如 AWS profiles、IAM 凭证等）。
    """
    env_vars = _get_api_key_env_vars(provider)
    if not env_vars:
        return None

    found = [v for v in env_vars if get_provider_env_value(v, env)]
    return found if found else None


def get_env_api_key(provider: str, env: ProviderEnv | None = None) -> str | None:
    """从已知环境变量获取 provider 的 API Key。

    不会返回需要 OAuth 令牌的 provider 的 API Key。
    """
    env_keys = find_env_keys(provider, env)
    if env_keys:
        api_key_env = env_keys[0] if provider != "anthropic" else next(
            (k for k in env_keys if k != ANTHROPIC_AUTH_TOKEN_ENV), env_keys[0]
        )
        if api_key_env:
            return get_provider_env_value(api_key_env, env)

    # Vertex AI 支持显式 API Key 或 Application Default Credentials
    if provider == "google-vertex":
        has_credentials = _has_vertex_adc_credentials(env)
        has_project = bool(
            get_provider_env_value("GOOGLE_CLOUD_PROJECT", env)
            or get_provider_env_value("GCLOUD_PROJECT", env)
        )
        has_location = bool(get_provider_env_value("GOOGLE_CLOUD_LOCATION", env))
        if has_credentials and has_project and has_location:
            return "<authenticated>"

    if provider == "amazon-bedrock":
        if (
            get_provider_env_value("AWS_PROFILE", env)
            or (
                get_provider_env_value("AWS_ACCESS_KEY_ID", env)
                and get_provider_env_value("AWS_SECRET_ACCESS_KEY", env)
            )
            or get_provider_env_value("AWS_BEARER_TOKEN_BEDROCK", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", env)
            or get_provider_env_value("AWS_CONTAINER_CREDENTIALS_FULL_URI", env)
            or get_provider_env_value("AWS_WEB_IDENTITY_TOKEN_FILE", env)
        ):
            return "<authenticated>"

    return None