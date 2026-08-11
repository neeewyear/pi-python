"""OpenRouter OAuth PKCE 流程（对应 ``oauth/openrouter.ts``）。

OpenRouter 使用 PKCE 授权码流程，将授权码交换为永久的、用户控制的 API 密钥，
而非过期的 access/refresh token 对。回调通过手动粘贴授权码完成（无本地回环服务器）。
"""

from __future__ import annotations

from typing import Any
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .pkce import generate_pkce

AUTHORIZE_URL = "https://openrouter.ai/auth"
TOKEN_URL = "https://openrouter.ai/api/v1/auth/keys"
LOGIN_TIMEOUT_MS = 5 * 60 * 1000
TOKEN_EXCHANGE_TIMEOUT_MS = 30_000


def _parse_authorization_input(input: str) -> str | None:
    """Parse the authorization code from user input."""
    value = input.strip()
    if not value:
        return None

    try:
        parsed = urlparse(value)
        if parsed.scheme and parsed.netloc:
            qs = parse_qs(parsed.query)
            code_list = qs.get("code")
            if (
                code_list
                and isinstance(code_list, list)
                and len(code_list) > 0
                and isinstance(code_list[0], str)
            ):
                return code_list[0]
    except Exception:
        pass

    if "code=" in value:
        qs = parse_qs(value)
        code_list = qs.get("code")
        if (
            code_list
            and isinstance(code_list, list)
            and len(code_list) > 0
            and isinstance(code_list[0], str)
        ):
            return code_list[0]

    return value


def _error_detail(body: dict[str, Any]) -> str | None:
    """Extract error detail from response body."""
    error_description = body.get("error_description")
    if isinstance(error_description, str):
        return error_description
    message = body.get("message")
    if isinstance(message, str):
        return message
    error = body.get("error")
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        msg = error.get("message")
        if isinstance(msg, str):
            return msg
    return None


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    signal: Any,
) -> OAuthCredential:
    """Exchange authorization code for an API key."""
    if signal and getattr(signal, "aborted", False):
        raise RuntimeError("Login cancelled")

    body = {
        "code": code,
        "code_verifier": verifier,
        "code_challenge_method": "S256",
    }

    async with httpx.AsyncClient() as client:
        response = await client.post(
            TOKEN_URL,
            json=body,
            headers={"Accept": "application/json"},
            timeout=TOKEN_EXCHANGE_TIMEOUT_MS / 1000,
        )

        try:
            resp_json = response.json()
        except Exception:
            if response.is_success:
                raise RuntimeError("OpenRouter OAuth returned invalid JSON")
            resp_json = {}

        if not response.is_success:
            detail = _error_detail(resp_json)
            msg = f"OpenRouter OAuth key exchange failed (HTTP {response.status_code})"
            if detail:
                msg += f": {detail}"
            raise RuntimeError(msg)

        if not isinstance(resp_json.get("key"), str) or len(resp_json["key"]) == 0:
            raise RuntimeError('OpenRouter OAuth response carries no "key"')

        return {
            "type": "oauth",
            "access": resp_json["key"],
            "refresh": "",
            "expires": 9223372036854775807,  # Number.MAX_SAFE_INTEGER
        }


async def _login_openrouter(interaction: ProviderAuthInteraction) -> OAuthCredential:
    """Login to OpenRouter."""
    pkce = await generate_pkce()
    verifier = pkce["verifier"]
    challenge = pkce["challenge"]

    params = urlencode(
        {
            "callback_url": "http://localhost:0/oauth/callback",
            "code_challenge": challenge,
            "code_challenge_method": "S256",
        }
    )
    authorize_url = f"{AUTHORIZE_URL}?{params}"

    interaction.notify(
        {
            "type": "auth_url",
            "url": authorize_url,
            "instructions": (
                "Complete sign-in in your browser. "
                "If the browser is on another machine, paste the final redirect URL here."
            ),
        }
    )

    manual_input = await interaction.prompt(
        {
            "type": "manual_code",
            "message": "Complete sign-in in your browser, or paste the authorization code / redirect URL here:",
            "placeholder": "http://localhost:0/oauth/callback?code=...",
        }
    )

    if not manual_input:
        raise RuntimeError("Missing authorization code")

    code = _parse_authorization_input(manual_input)
    if not code:
        raise RuntimeError("Missing authorization code")

    interaction.notify(
        {
            "type": "progress",
            "message": "Exchanging authorization code for an API key...",
        }
    )
    return await _exchange_authorization_code(code, verifier, interaction.signal)


def _create_openrouter_oauth() -> OAuthAuth:
    """Create the OpenRouter OAuthAuth instance."""

    class _OpenRouterOAuth:
        name: str = "OpenRouter OAuth"
        login_label: str | None = "Sign in with OpenRouter"

        async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
            return await _login_openrouter(interaction)

        async def refresh(
            self, credential: OAuthCredential, signal: Any
        ) -> OAuthCredential:
            # OpenRouter API key 是永久的，直接返回
            return credential

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return ModelAuth(api_key=credential["access"])

    return _OpenRouterOAuth()


openrouter_oauth: OAuthAuth = _create_openrouter_oauth()
