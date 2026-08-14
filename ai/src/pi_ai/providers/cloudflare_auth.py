"""Cloudflare 认证实现。"""

from __future__ import annotations

from typing import Any

from ..auth.types import ApiKeyAuth, ApiKeyCredential, AuthContext, AuthResult
from ..types import ProviderEnv

CLOUDFLARE_API_KEY = "CLOUDFLARE_API_KEY"
CLOUDFLARE_ACCOUNT_ID = "CLOUDFLARE_ACCOUNT_ID"
CLOUDFLARE_GATEWAY_ID = "CLOUDFLARE_GATEWAY_ID"

CloudflareAuthKind = str  # "workers-ai" | "ai-gateway"


async def _resolve_value(
    name: str,
    ctx: AuthContext,
    credential: ApiKeyCredential | None,
    signal: Any,
) -> str | None:
    """解析单个环境变量值。

    Per-field merge: 优先使用 credential 中的值，回退到环境变量。
    """
    if credential is not None:
        if name == CLOUDFLARE_API_KEY:
            from_credential = credential.get("key")
        else:
            env = credential.get("env")
            from_credential = env.get(name) if env else None
        if from_credential is not None:
            return from_credential
    if signal is not None:
        signal.throw_if_cancelled()
    value = await ctx.env(name)
    if signal is not None:
        signal.throw_if_cancelled()
    return value


async def _resolve_cloudflare_env(
    kind: CloudflareAuthKind,
    ctx: AuthContext,
    credential: ApiKeyCredential | None,
    signal: Any,
) -> dict[str, Any] | None:
    """解析 Cloudflare 环境变量。"""
    api_key = await _resolve_value(CLOUDFLARE_API_KEY, ctx, credential, signal)
    account_id = await _resolve_value(CLOUDFLARE_ACCOUNT_ID, ctx, credential, signal)
    gateway_id = (
        await _resolve_value(CLOUDFLARE_GATEWAY_ID, ctx, credential, signal)
        if kind == "ai-gateway"
        else None
    )

    if not api_key or not account_id or (kind == "ai-gateway" and not gateway_id):
        return None

    env: ProviderEnv = {CLOUDFLARE_ACCOUNT_ID: account_id}
    if gateway_id:
        env[CLOUDFLARE_GATEWAY_ID] = gateway_id

    return {
        "apiKey": api_key,
        "env": env,
        "source": "stored credential" if credential else CLOUDFLARE_API_KEY,
    }


def cloudflare_workers_ai_auth() -> ApiKeyAuth:
    """创建 Cloudflare Workers AI 的 ApiKeyAuth 实现。"""
    auth: ApiKeyAuth = _CloudflareWorkersAIAuth()
    return auth


def cloudflare_ai_gateway_auth() -> ApiKeyAuth:
    """创建 Cloudflare AI Gateway 的 ApiKeyAuth 实现。"""
    auth: ApiKeyAuth = _CloudflareAIGatewayAuth()
    return auth


class _CloudflareWorkersAIAuth:
    """Cloudflare Workers AI ApiKeyAuth 实现。"""

    name = "Cloudflare API key"

    async def login(self, interaction: Any) -> ApiKeyCredential:
        key = await interaction.prompt({
            "type": "secret",
            "message": "Enter Cloudflare API key",
            "signal": None,
            "placeholder": None,
            "options": [],
        })
        account_id = await interaction.prompt({
            "type": "text",
            "message": "Enter Cloudflare account ID",
            "signal": None,
            "placeholder": None,
            "options": [],
        })
        return {"type": "api_key", "key": key, "env": {CLOUDFLARE_ACCOUNT_ID: account_id}}

    async def check(self, input: dict[str, Any]) -> Any:
        return None

    async def resolve(self, input: dict[str, Any]) -> AuthResult | None:
        ctx: Any = input.get("ctx")
        credential: Any = input.get("credential")
        signal: Any = input.get("signal")
        resolved = await _resolve_cloudflare_env("workers-ai", ctx, credential, signal)
        if not resolved:
            return None
        return {
            "auth": {"api_key": resolved["apiKey"]},
            "env": resolved["env"],
            "source": resolved["source"],
        }


class _CloudflareAIGatewayAuth:
    """Cloudflare AI Gateway ApiKeyAuth 实现。"""

    name = "Cloudflare API key"

    async def login(self, interaction: Any) -> ApiKeyCredential:
        key = await interaction.prompt({
            "type": "secret",
            "message": "Enter Cloudflare API key",
            "signal": None,
            "placeholder": None,
            "options": [],
        })
        account_id = await interaction.prompt({
            "type": "text",
            "message": "Enter Cloudflare account ID",
            "signal": None,
            "placeholder": None,
            "options": [],
        })
        gateway_id = await interaction.prompt({
            "type": "text",
            "message": "Enter Cloudflare AI Gateway ID",
            "signal": None,
            "placeholder": None,
            "options": [],
        })
        return {
            "type": "api_key",
            "key": key,
            "env": {
                CLOUDFLARE_ACCOUNT_ID: account_id,
                CLOUDFLARE_GATEWAY_ID: gateway_id,
            },
        }

    async def check(self, input: dict[str, Any]) -> Any:
        return None

    async def resolve(self, input: dict[str, Any]) -> AuthResult | None:
        ctx: Any = input.get("ctx")
        credential: Any = input.get("credential")
        signal: Any = input.get("signal")
        resolved = await _resolve_cloudflare_env("ai-gateway", ctx, credential, signal)
        if not resolved:
            return None
        return {
            "auth": {
                "headers": {
                    "cf-aig-authorization": f"Bearer {resolved['apiKey']}",
                    "Authorization": None,
                    "x-api-key": None,
                },
            },
            "env": resolved["env"],
            "source": resolved["source"],
        }