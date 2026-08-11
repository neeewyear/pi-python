"""GitHub Copilot OAuth 流程（对应 ``oauth/github-copilot.ts``）。"""

from __future__ import annotations

import base64
import re
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .device_code import (
    OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult,
    poll_oauth_device_code_flow,
)

_decode = base64.b64decode
CLIENT_ID = _decode(b"SXYxLmI1MDdhMDhjODdlY2ZlOTg=").decode("ascii")

COPILOT_HEADERS: dict[str, str] = {
    "User-Agent": "GitHubCopilotChat/0.35.0",
    "Editor-Version": "vscode/1.107.0",
    "Editor-Plugin-Version": "copilot-chat/0.35.0",
    "Copilot-Integration-Id": "vscode-chat",
}
COPILOT_API_VERSION = "2026-06-01"


def _normalize_domain(input: str) -> str | None:
    """Normalize an enterprise domain input."""
    trimmed = input.strip()
    if not trimmed:
        return None
    if "://" not in trimmed:
        trimmed = f"https://{trimmed}"
    try:
        from urllib.parse import urlparse

        parsed = urlparse(trimmed)
        return parsed.hostname
    except Exception:
        return None


def _get_urls(domain: str) -> dict[str, str]:
    """Get OAuth URLs for a given domain."""
    return {
        "device_code_url": f"https://{domain}/login/device/code",
        "access_token_url": f"https://{domain}/login/oauth/access_token",
        "copilot_token_url": f"https://api.{domain}/copilot_internal/v2/token",
    }


def _get_base_url_from_token(token: str) -> str | None:
    """Extract the API base URL from a Copilot token's proxy-ep claim."""
    match = re.search(r"proxy-ep=([^;]+)", token)
    if not match:
        return None
    proxy_host = match.group(1)
    api_host = re.sub(r"^proxy\.", "api.", proxy_host)
    return f"https://{api_host}"


def _get_github_copilot_base_url(
    token: str | None = None,
    enterprise_domain: str | None = None,
) -> str:
    """Determine the GitHub Copilot API base URL."""
    if token:
        url_from_token = _get_base_url_from_token(token)
        if url_from_token:
            return url_from_token
    if enterprise_domain:
        return f"https://copilot-api.{enterprise_domain}"
    return "https://api.individual.githubcopilot.com"


def _as_record(value: Any) -> dict[str, Any] | None:
    """Type guard: return value as dict if it's a mapping."""
    if isinstance(value, dict):
        return value
    return None


def _is_selectable_copilot_model(item: dict[str, Any]) -> bool:
    """Check if a Copilot model is selectable."""
    policy = _as_record(item.get("policy"))
    capabilities = _as_record(item.get("capabilities"))
    supports = _as_record(capabilities.get("supports") if capabilities else None)
    return (
        item.get("model_picker_enabled") is True
        and (policy is None or policy.get("state") != "disabled")
        and (supports is None or supports.get("tool_calls") is not False)
    )


def _parse_available_copilot_model_ids(raw: Any) -> list[str]:
    """Parse available Copilot model IDs from the models response."""
    data = _as_record(raw)
    if data is None:
        raise ValueError("Invalid Copilot models response")
    models_list = data.get("data")
    if not isinstance(models_list, list):
        raise ValueError("Invalid Copilot models response")

    ids: list[str] = []
    for raw_item in models_list:
        item = _as_record(raw_item)
        if item is None:
            continue
        item_id = item.get("id")
        if isinstance(item_id, str) and _is_selectable_copilot_model(item):
            ids.append(item_id)
    return ids


async def _fetch_available_github_copilot_model_ids(
    copilot_token: str,
    enterprise_domain: str | None,
    signal: Any,
) -> list[str]:
    """Fetch available GitHub Copilot model IDs."""
    base_url = _get_github_copilot_base_url(copilot_token, enterprise_domain)
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {copilot_token}",
        **COPILOT_HEADERS,
        "X-GitHub-Api-Version": COPILOT_API_VERSION,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{base_url}/models",
            headers=headers,
            timeout=httpx.Timeout(5.0),
        )
        response.raise_for_status()
        raw = response.json()
    return _parse_available_copilot_model_ids(raw)


async def _fetch_json(
    url: str,
    *,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    data: dict[str, str] | None = None,
    signal: Any = None,
) -> Any:
    """HTTP fetch helper."""
    async with httpx.AsyncClient() as client:
        response = await client.request(
            method,
            url,
            headers=headers,
            data=data,
        )
        if response.status_code >= 400:
            text = response.text
            raise RuntimeError(
                f"{response.status_code} {response.reason_phrase}: {text}"
            )
        return response.json()


async def _start_device_flow(domain: str, signal: Any) -> dict[str, Any]:
    """Start the GitHub device code flow."""
    urls = _get_urls(domain)
    data = await _fetch_json(
        urls["device_code_url"],
        method="POST",
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": "GitHubCopilotChat/0.35.0",
        },
        data={
            "client_id": CLIENT_ID,
            "scope": "read:user",
        },
        signal=signal,
    )

    if not isinstance(data, dict):
        raise RuntimeError("Invalid device code response")

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    interval = data.get("interval")
    expires_in = data.get("expires_in")

    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or (interval is not None and not isinstance(interval, (int, float)))
        or not isinstance(expires_in, (int, float))
    ):
        raise RuntimeError("Invalid device code response fields")

    # Validate verification URI
    parsed = urlparse(verification_uri)
    if parsed.scheme not in ("https", "http"):
        raise RuntimeError("Untrusted verification_uri in device code response")

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "interval": int(interval) if interval is not None else None,
        "expires_in": int(expires_in),
    }


async def _poll_for_github_access_token(
    domain: str,
    device: dict[str, Any],
    signal: Any,
) -> str:
    """Poll for the GitHub access token using device code flow."""
    urls = _get_urls(domain)

    async def _poll(poll_device_code: str, sig: Any) -> OAuthDeviceCodePollResult:
        raw = await _fetch_json(
            urls["access_token_url"],
            method="POST",
            headers={
                "Accept": "application/json",
                "Content-Type": "application/x-www-form-urlencoded",
                "User-Agent": "GitHubCopilotChat/0.35.0",
            },
            data={
                "client_id": CLIENT_ID,
                "device_code": poll_device_code,
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
            },
            signal=sig,
        )

        if isinstance(raw, dict):
            if isinstance(raw.get("access_token"), str):
                return {"status": "complete", "value": raw["access_token"]}

            error = raw.get("error")
            if isinstance(error, str):
                description = raw.get("error_description")
                if error == "authorization_pending":
                    return {"status": "pending"}
                if error == "slow_down":
                    interval = raw.get("interval")
                    result: OAuthDeviceCodePollResult = {"status": "slow_down"}
                    if isinstance(interval, (int, float)):
                        result["interval_seconds"] = int(interval)
                    return result
                desc_suffix = f": {description}" if isinstance(description, str) else ""
                return {
                    "status": "failed",
                    "message": f"Device flow failed: {error}{desc_suffix}",
                }

        return {"status": "failed", "message": "Invalid device token response"}

    options: OAuthDeviceCodePollOptions = {
        "device_code": device["device_code"],
        "expires_in_seconds": device["expires_in"],
        "signal": signal,
        "poll": _poll,
    }
    interval = device.get("interval")
    if isinstance(interval, (int, float)):
        options["interval_seconds"] = int(interval)
    return cast(str, await poll_oauth_device_code_flow(options))


async def _refresh_github_copilot_access_token(
    refresh_token: str,
    enterprise_domain: str | None,
    signal: Any,
) -> OAuthCredential:
    """Refresh the GitHub Copilot access token."""
    domain = enterprise_domain or "github.com"
    urls = _get_urls(domain)

    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {refresh_token}",
        **COPILOT_HEADERS,
    }
    async with httpx.AsyncClient() as client:
        response = await client.get(urls["copilot_token_url"], headers=headers)
        response.raise_for_status()
        raw = response.json()

    if not isinstance(raw, dict):
        raise RuntimeError("Invalid Copilot token response")

    token = raw.get("token")
    expires_at = raw.get("expires_at")

    if not isinstance(token, str) or not isinstance(expires_at, (int, float)):
        raise RuntimeError("Invalid Copilot token response fields")

    result: OAuthCredential = {
        "type": "oauth",
        "refresh": refresh_token,
        "access": token,
        "expires": int(expires_at * 1000 - 5 * 60 * 1000),
    }
    if enterprise_domain is not None:
        result["enterprise_url"] = enterprise_domain
    return result


async def _refresh_github_copilot_token(
    refresh_token: str,
    enterprise_domain: str | None,
    signal: Any,
) -> OAuthCredential:
    """Refresh GitHub Copilot token and fetch available models."""
    credentials = await _refresh_github_copilot_access_token(
        refresh_token, enterprise_domain, signal
    )
    credentials[
        "available_model_ids"
    ] = await _fetch_available_github_copilot_model_ids(
        credentials["access"], enterprise_domain, signal
    )
    return credentials


async def _enable_github_copilot_model(
    token: str,
    model_id: str,
    enterprise_domain: str | None,
    signal: Any,
) -> bool:
    """Enable a model for the user's GitHub Copilot account."""
    base_url = _get_github_copilot_base_url(token, enterprise_domain)
    url = f"{base_url}/models/{model_id}/policy"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        **COPILOT_HEADERS,
        "openai-intent": "chat-policy",
        "x-interaction-type": "chat-policy",
    }
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url, json={"state": "enabled"}, headers=headers
            )
            return response.is_success
    except Exception:
        return False


# 15.6 阶段：从 github_copilot_models 导入
_GITHUB_COPILOT_MODELS: dict[str, Any] = {}


async def _enable_all_github_copilot_models(
    token: str,
    enterprise_domain: str | None,
    signal: Any,
) -> None:
    """Enable all known GitHub Copilot models."""
    _models = _GITHUB_COPILOT_MODELS if _GITHUB_COPILOT_MODELS is not None else {}
    for model in _models.values():
        await _enable_github_copilot_model(
            token, model["id"], enterprise_domain, signal
        )


async def _login_github_copilot(
    interaction: ProviderAuthInteraction,
) -> OAuthCredential:
    """Login to GitHub Copilot."""
    input_text = await interaction.prompt(
        {
            "type": "text",
            "message": "GitHub Enterprise URL/domain (blank for github.com)",
            "placeholder": "company.ghe.com",
        }
    )
    if interaction.signal.aborted:
        raise RuntimeError("Login cancelled")

    trimmed = input_text.strip()
    enterprise_domain = _normalize_domain(input_text) if trimmed else None
    if trimmed and not enterprise_domain:
        raise RuntimeError("Invalid GitHub Enterprise URL/domain")
    domain = enterprise_domain or "github.com"

    device = await _start_device_flow(domain, interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri"],
            "interval_seconds": device.get("interval"),
            "expires_in_seconds": device["expires_in"],
        }
    )

    github_access_token = await _poll_for_github_access_token(
        domain, device, interaction.signal
    )
    credentials = await _refresh_github_copilot_access_token(
        github_access_token,
        enterprise_domain,
        interaction.signal,
    )
    interaction.notify({"type": "progress", "message": "Enabling models..."})
    await _enable_all_github_copilot_models(
        credentials["access"], enterprise_domain, interaction.signal
    )
    credentials[
        "available_model_ids"
    ] = await _fetch_available_github_copilot_model_ids(
        credentials["access"],
        enterprise_domain,
        interaction.signal,
    )
    return credentials


def _copilot_enterprise_domain(credential: OAuthCredential) -> str | None:
    """Extract enterprise domain from credential."""
    enterprise_url = credential.get("enterprise_url")
    if not isinstance(enterprise_url, str) or not enterprise_url:
        return None
    normalized = _normalize_domain(enterprise_url)
    return normalized if normalized else None


def _create_github_copilot_oauth() -> OAuthAuth:
    """Create the GitHub Copilot OAuthAuth instance."""

    class _GitHubCopilotOAuth:
        name: str = "GitHub Copilot"
        login_label: str | None = None

        async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
            return await _login_github_copilot(interaction)

        async def refresh(
            self, credential: OAuthCredential, signal: Any
        ) -> OAuthCredential:
            return await _refresh_github_copilot_token(
                credential["refresh"],
                _copilot_enterprise_domain(credential),
                signal,
            )

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return ModelAuth(
                api_key=credential["access"],
                base_url=_get_github_copilot_base_url(
                    credential["access"],
                    _copilot_enterprise_domain(credential),
                ),
            )

    return _GitHubCopilotOAuth()


github_copilot_oauth: OAuthAuth = _create_github_copilot_oauth()
