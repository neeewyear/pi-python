"""Anthropic Provider（对应 ``anthropic.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.anthropic_messages_lazy import anthropic_messages_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, AuthResult
from ..models import CreateProviderOptions, create_provider
from .anthropic_models import ANTHROPIC_MODELS

# 环境变量常量（对应 ``env-api-keys.ts``）
ANTHROPIC_API_KEY_ENV = "ANTHROPIC_API_KEY"
ANTHROPIC_AUTH_TOKEN_ENV = "ANTHROPIC_AUTH_TOKEN"
ANTHROPIC_OAUTH_TOKEN_ENV = "ANTHROPIC_OAUTH_TOKEN"


class _AnthropicApiKeyAuth:
    """Anthropic API Key 认证实现。"""

    name = "Anthropic API key"

    async def login(self, interaction: Any) -> ApiKeyCredential:
        if interaction.signal is not None:
            interaction.signal.throw_if_cancelled()
        key = await interaction.prompt(
            {
                "type": "secret",
                "message": "Enter Anthropic API key",
                "signal": None,
                "placeholder": None,
                "options": [],
            }
        )
        if interaction.signal is not None:
            interaction.signal.throw_if_cancelled()
        return {"type": "api_key", "key": key}

    async def check(self, input: dict[str, Any]) -> Any:
        return None

    async def resolve(self, input: dict[str, Any]) -> AuthResult | None:
        ctx: Any = input.get("ctx")
        credential: Any = input.get("credential")
        signal: Any = input.get("signal")
        if signal is not None:
            signal.throw_if_cancelled()
        if credential is not None and credential.get("key"):
            return {
                "auth": {"api_key": credential["key"]},
                "env": credential.get("env"),
                "source": "stored credential",
            }
        auth_token = await ctx.env(ANTHROPIC_AUTH_TOKEN_ENV)
        if signal is not None:
            signal.throw_if_cancelled()
        if auth_token:
            return {
                "auth": {"headers": {"Authorization": f"Bearer {auth_token}"}},
                "source": ANTHROPIC_AUTH_TOKEN_ENV,
            }
        for env_var in [ANTHROPIC_OAUTH_TOKEN_ENV, ANTHROPIC_API_KEY_ENV]:
            api_key = await ctx.env(env_var)
            if signal is not None:
                signal.throw_if_cancelled()
            if api_key:
                return {"auth": {"api_key": api_key}, "source": env_var}
        return None


def anthropic_api_key_auth() -> ApiKeyAuth:
    """创建 Anthropic API Key 认证实现。"""
    return _AnthropicApiKeyAuth()


def anthropic_provider() -> Any:
    """创建 Anthropic Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="anthropic",
            name="Anthropic",
            base_url="https://api.anthropic.com",
            models=list(ANTHROPIC_MODELS.values()),
            api=anthropic_messages_api(),
        )
    )
