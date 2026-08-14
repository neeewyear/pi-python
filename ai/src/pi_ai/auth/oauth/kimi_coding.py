"""Kimi Code (subscription) OAuth 设备码流程。

RFC 8628 device authorization grant 针对 https://auth.kimi.com，
使用 JSON 响应。access token 用于向 https://api.kimi.com/coding 发起请求，
以 ``Authorization: Bearer`` 头部传递。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from ...utils.provider_env import get_provider_env_value
from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .device_code import (
    OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult,
    poll_oauth_device_code_flow,
)

CLIENT_ID = "17e5f671-d194-4dfb-9706-5516cb48c098"
DEFAULT_OAUTH_HOST = "https://auth.kimi.com"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
DEFAULT_POLL_INTERVAL_SECONDS = 5
REQUEST_TIMEOUT_MS = 30 * 1000
REFRESH_MAX_RETRIES = 3


def _get_oauth_host() -> str:
    """Get the OAuth host, with override support."""
    override = get_provider_env_value("KIMI_CODE_OAUTH_HOST") or get_provider_env_value(
        "KIMI_OAUTH_HOST"
    )
    return (override or DEFAULT_OAUTH_HOST).rstrip("/")


def _trusted_http_url(value: Any) -> str | None:
    """Validate that the URL uses http or https protocol."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = urlparse(value)
        if parsed.scheme not in ("https", "http"):
            return None
        return value
    except Exception:
        return None


async def _start_device_authorization(oauth_host: str, signal: Any) -> dict[str, Any]:
    """Start the Kimi Code device authorization flow."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{oauth_host}/api/oauth/device_authorization",
            data={"client_id": CLIENT_ID},
            headers={
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            },
        )

    if not response.is_success:
        text = response.text
        raise RuntimeError(
            f"Kimi Code device authorization failed with status {response.status_code}"
            f"{f': {text}' if text else ''}"
        )

    try:
        json_data = response.json()
    except Exception:
        json_data = None

    if not isinstance(json_data, dict):
        raise RuntimeError(
            f"Invalid Kimi Code device authorization response: {response.text}"
        )

    device_code = json_data.get("device_code")
    user_code = json_data.get("user_code")
    verification_uri = json_data.get("verification_uri")
    verification_uri_complete = json_data.get("verification_uri_complete")

    if (
        not isinstance(device_code, str)
        or not isinstance(user_code, str)
        or not isinstance(verification_uri, str)
        or not isinstance(verification_uri_complete, str)
        or not _trusted_http_url(verification_uri_complete)
        or not _trusted_http_url(verification_uri)
    ):
        raise RuntimeError(
            f"Invalid Kimi Code device authorization response: {json_data}"
        )

    interval = json_data.get("interval")
    expires_in = json_data.get("expires_in")

    return {
        "device_code": device_code,
        "user_code": user_code,
        "verification_uri": verification_uri,
        "verification_uri_complete": verification_uri_complete,
        "interval_seconds": (
            int(interval)
            if isinstance(interval, (int, float)) and interval > 0
            else DEFAULT_POLL_INTERVAL_SECONDS
        ),
        "expires_in_seconds": (
            int(expires_in)
            if isinstance(expires_in, (int, float)) and expires_in > 0
            else DEVICE_CODE_TIMEOUT_SECONDS
        ),
    }


def _parse_token_response(
    json_data: dict[str, Any] | None, operation: str
) -> dict[str, Any]:
    """Parse token response from Kimi Code."""
    if json_data is None:
        raise RuntimeError(f"Kimi Code token {operation} response missing fields: null")

    access_token = json_data.get("access_token")
    refresh_token = json_data.get("refresh_token")
    expires_in = json_data.get("expires_in")

    if (
        not isinstance(access_token, str)
        or not access_token
        or not isinstance(refresh_token, str)
        or not refresh_token
        or not isinstance(expires_in, (int, float))
        or expires_in <= 0
    ):
        raise RuntimeError(
            f"Kimi Code token {operation} response missing fields: {json_data}"
        )

    return {
        "access": access_token,
        "refresh": refresh_token,
        "expires": int(time.time() * 1000 + expires_in * 1000),
    }


async def _poll_for_token(
    oauth_host: str,
    device: dict[str, Any],
    signal: Any,
) -> dict[str, Any]:
    """Poll for the Kimi Code token using device code flow."""

    async def _poll(poll_device_code: str, sig: Any) -> OAuthDeviceCodePollResult:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{oauth_host}/api/oauth/token",
                data={
                    "client_id": CLIENT_ID,
                    "device_code": poll_device_code,
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                },
                headers={
                    "Content-Type": "application/x-www-form-urlencoded",
                    "Accept": "application/json",
                },
            )

        if response.status_code >= 500:
            text = response.text
            return {
                "status": "failed",
                "message": (
                    f"Kimi Code device token request failed with status {response.status_code}"
                    f"{f': {text}' if text else ''}"
                ),
            }

        try:
            json_data = response.json()
        except Exception:
            json_data = None

        if not isinstance(json_data, dict):
            json_data = {}

        if response.is_success and isinstance(json_data.get("access_token"), str):
            try:
                return {
                    "status": "complete",
                    "value": _parse_token_response(json_data, "poll"),
                }
            except RuntimeError as e:
                return {"status": "failed", "message": str(e)}

        error = json_data.get("error")
        description = json_data.get("error_description")
        desc_suffix = f": {description}" if isinstance(description, str) else ""

        if error == "authorization_pending":
            return {"status": "pending"}
        if error == "slow_down":
            interval = json_data.get("interval")
            result: OAuthDeviceCodePollResult = {"status": "slow_down"}
            if isinstance(interval, (int, float)) and interval > 0:
                result["interval_seconds"] = int(interval)
            return result
        if error == "expired_token":
            return {
                "status": "failed",
                "message": "Kimi Code device authorization expired. Please restart login.",
            }
        if error == "access_denied":
            return {"status": "failed", "message": "Kimi Code login was denied."}

        return {
            "status": "failed",
            "message": (
                f"Kimi Code device token request failed (status {response.status_code})"
                f"{f': {error}{desc_suffix}' if isinstance(error, str) else ''}"
            ),
        }

    options: OAuthDeviceCodePollOptions = {
        "device_code": device["device_code"],
        "interval_seconds": device["interval_seconds"],
        "expires_in_seconds": device["expires_in_seconds"],
        "signal": signal,
        "poll": _poll,
    }
    return cast(dict[str, Any], await poll_oauth_device_code_flow(options))


def _is_retryable_refresh_failure(status_code: int) -> bool:
    """Check if a refresh failure is retryable."""
    return status_code == 429 or status_code >= 500


async def _refresh_token(
    oauth_host: str,
    refresh_token_value: str,
    signal: Any,
) -> dict[str, Any]:
    """Refresh the Kimi Code token with retry logic."""
    last_error: Exception | None = None
    for attempt in range(REFRESH_MAX_RETRIES + 1):
        if attempt > 0:
            if signal and getattr(signal, "aborted", False):
                raise RuntimeError("Kimi Code token refresh aborted")
            await asyncio.sleep(1.0 * 2 ** (attempt - 1))

        if signal and getattr(signal, "aborted", False):
            raise RuntimeError("Kimi Code token refresh aborted")

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{oauth_host}/api/oauth/token",
                    data={
                        "client_id": CLIENT_ID,
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token_value,
                    },
                    headers={
                        "Content-Type": "application/x-www-form-urlencoded",
                        "Accept": "application/json",
                    },
                )
        except Exception as e:
            last_error = e if isinstance(e, Exception) else RuntimeError(str(e))
            continue

        try:
            json_data = response.json()
        except Exception:
            json_data = None

        if not isinstance(json_data, dict):
            json_data = {}

        if response.is_success:
            return _parse_token_response(json_data, "refresh")

        # Unauthorized: 凭据已失效，需要重新登录
        if (
            response.status_code in (401, 403)
            or json_data.get("error") == "invalid_grant"
        ):
            description = json_data.get("error_description")
            desc_suffix = f": {description}" if isinstance(description, str) else ""
            raise RuntimeError(
                f"Kimi Code token refresh unauthorized (status {response.status_code}){desc_suffix}"
            )

        if (
            _is_retryable_refresh_failure(response.status_code)
            and attempt < REFRESH_MAX_RETRIES
        ):
            last_error = RuntimeError(
                f"Kimi Code token refresh failed with status {response.status_code}"
            )
            continue

        raise RuntimeError(
            f"Kimi Code token refresh failed with status {response.status_code}"
            f"{f': {json_data}' if json_data else ''}"
        )

    raise last_error or RuntimeError("Kimi Code token refresh failed")


async def _login_kimi_coding(interaction: ProviderAuthInteraction) -> OAuthCredential:
    """Login to Kimi Code."""
    oauth_host = _get_oauth_host()
    device = await _start_device_authorization(oauth_host, interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri_complete"],
            "interval_seconds": device["interval_seconds"],
            "expires_in_seconds": device["expires_in_seconds"],
        }
    )
    token = await _poll_for_token(oauth_host, device, interaction.signal)
    return {
        "type": "oauth",
        "access": token["access"],
        "refresh": token["refresh"],
        "expires": token["expires"],
    }


def _create_kimi_coding_oauth() -> OAuthAuth:
    """Create the Kimi Code OAuthAuth instance."""

    class _KimiCodingOAuth:
        name: str = "Kimi Code (subscription)"
        login_label: str | None = "Sign in with Kimi Code"

        async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
            return await _login_kimi_coding(interaction)

        async def refresh(
            self, credential: OAuthCredential, signal: Any
        ) -> OAuthCredential:
            token = await _refresh_token(
                _get_oauth_host(), credential["refresh"], signal
            )
            return {
                "type": "oauth",
                "access": token["access"],
                "refresh": token["refresh"],
                "expires": token["expires"],
            }

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return ModelAuth(
                api_key=credential["access"],
                headers={"Authorization": f"Bearer {credential['access']}"},
            )

    return _KimiCodingOAuth()


kimi_coding_oauth: OAuthAuth = _create_kimi_coding_oauth()
