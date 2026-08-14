"""Radius gateway OAuth 设备码流程。

Radius 是一个 pi-messages 网关。OAuth 客户端 API 位于配置的网关上。
此模块仅实现设备码流程（不含浏览器回调服务器）。
"""

from __future__ import annotations

import time
from typing import Any, cast
from urllib.parse import urljoin

import httpx

from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .device_code import (
    OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult,
    poll_oauth_device_code_flow,
)

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 1456
CALLBACK_PATH = "/oauth/callback"
REDIRECT_URI = f"http://{CALLBACK_HOST}:{CALLBACK_PORT}{CALLBACK_PATH}"
TOKEN_EXPIRY_SKEW_MS = 60_000
LOGIN_METHOD_BROWSER = "browser"
LOGIN_METHOD_DEVICE_CODE = "device-code"
OAUTH_CLIENT_ID = "pi-gateway"
OAUTH_SCOPE = "gateway offline_access"
OAUTH_DEVICE_CODE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:device_code"


class RadiusOAuthOptions:
    """Radius OAuth 配置选项。"""

    def __init__(self, name: str, gateway: str) -> None:
        self.name = name
        self.gateway = gateway


class OAuthResponseError(RuntimeError):
    """OAuth 响应错误。"""

    def __init__(
        self,
        status: int,
        oauth_error: str | None,
        description: str | None,
        message: str,
    ) -> None:
        detail = (
            f"{oauth_error}: {description}"
            if oauth_error and description
            else (oauth_error or description or str(status))
        )
        super().__init__(f"{message}: {detail}")
        self.status = status
        self.oauth_error = oauth_error


async def _read_oauth_response_error(
    response: httpx.Response, message: str
) -> OAuthResponseError:
    """Read OAuth error from response."""
    text = response.text
    oauth_error: str | None = None
    description: str | None = None

    if text:
        try:
            data = response.json()
            if isinstance(data, dict):
                err = data.get("error")
                oauth_error = str(err) if isinstance(err, str) else None
                desc = data.get("error_description")
                description = str(desc) if isinstance(desc, str) else None
        except Exception:
            description = text

    return OAuthResponseError(response.status_code, oauth_error, description, message)


async def _request_oauth_token(
    gateway: str,
    body: dict[str, str],
    signal: Any,
) -> OAuthCredential:
    """Request an OAuth token from the Radius gateway."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                urljoin(gateway, "/v1/oauth/token"),
                data=body,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
    except Exception:
        if signal and getattr(signal, "aborted", False):
            raise RuntimeError("Login cancelled") from None
        raise

    if not response.is_success:
        raise await _read_oauth_response_error(
            response, "Radius OAuth token request failed"
        )

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError("Radius OAuth token request returned invalid JSON")

    credential: OAuthCredential = {
        "type": "oauth",
        "access": data.get("access_token", ""),
        "refresh": data.get("refresh_token", ""),
        "expires": int(
            time.time() * 1000 + data.get("expires_in", 0) * 1000 - TOKEN_EXPIRY_SKEW_MS
        ),
    }
    scope = data.get("scope")
    if isinstance(scope, str):
        credential["scope"] = scope
    return credential


async def _request_device_authorization(gateway: str, signal: Any) -> dict[str, Any]:
    """Request device authorization from the Radius gateway."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                urljoin(gateway, "/v1/oauth/device"),
                data={"client_id": OAUTH_CLIENT_ID, "scope": OAUTH_SCOPE},
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
    except Exception:
        if signal and getattr(signal, "aborted", False):
            raise RuntimeError("Login cancelled") from None
        raise

    if not response.is_success:
        raise await _read_oauth_response_error(
            response, "Radius OAuth device authorization failed"
        )

    data = response.json()
    if not isinstance(data, dict):
        raise RuntimeError(
            "Radius OAuth device authorization response is missing required fields"
        )

    device_code = data.get("device_code")
    user_code = data.get("user_code")
    verification_uri = data.get("verification_uri")
    expires_in = data.get("expires_in")

    if not device_code or not user_code or not verification_uri or not expires_in:
        raise RuntimeError(
            "Radius OAuth device authorization response is missing required fields"
        )

    return {
        "device_code": str(device_code),
        "user_code": str(user_code),
        "verification_uri": str(verification_uri),
        "expires_in": int(expires_in),
        "interval": int(data["interval"]) if data.get("interval") is not None else None,
    }


async def _login_with_device_code(
    gateway: str, interaction: ProviderAuthInteraction
) -> OAuthCredential:
    """Login to Radius using device code flow."""
    device = await _request_device_authorization(gateway, interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device["verification_uri"],
            "interval_seconds": device.get("interval"),
            "expires_in_seconds": device["expires_in"],
        }
    )

    async def _poll(poll_device_code: str, sig: Any) -> OAuthDeviceCodePollResult:
        try:
            credentials = await _request_oauth_token(
                gateway,
                {
                    "grant_type": OAUTH_DEVICE_CODE_GRANT_TYPE,
                    "client_id": OAUTH_CLIENT_ID,
                    "device_code": poll_device_code,
                },
                sig,
            )
            return {"status": "complete", "value": credentials}
        except OAuthResponseError as e:
            if e.oauth_error == "authorization_pending":
                return {"status": "pending"}
            if e.oauth_error == "slow_down":
                return {"status": "slow_down"}
            if e.oauth_error == "expired_token":
                return {"status": "failed", "message": "Device authorization expired."}
            if e.oauth_error == "access_denied":
                return {
                    "status": "failed",
                    "message": "Device authorization was denied.",
                }
            raise

    options: OAuthDeviceCodePollOptions = {
        "device_code": device["device_code"],
        "expires_in_seconds": device["expires_in"],
        "signal": interaction.signal,
        "poll": _poll,
    }
    interval = device.get("interval")
    if isinstance(interval, (int, float)):
        options["interval_seconds"] = int(interval)
    return cast(OAuthCredential, await poll_oauth_device_code_flow(options))


async def _login_radius(
    options: RadiusOAuthOptions,
    interaction: ProviderAuthInteraction,
) -> OAuthCredential:
    """Login to Radius with method selection."""
    login_method = await interaction.prompt(
        {
            "type": "select",
            "message": f"Sign in to {options.name}:",
            "options": [
                {"id": LOGIN_METHOD_DEVICE_CODE, "label": "Sign in with device code"},
            ],
        }
    )

    if login_method == LOGIN_METHOD_DEVICE_CODE:
        return await _login_with_device_code(options.gateway, interaction)
    raise RuntimeError(f"Unknown {options.name} sign-in method: {login_method}")


def create_radius_oauth(options: RadiusOAuthOptions) -> OAuthAuth:
    """Create a Radius OAuthAuth instance."""

    class _RadiusOAuth:
        name: str = options.name
        login_label: str | None = None

        async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
            return await _login_radius(options, interaction)

        async def refresh(
            self, credential: OAuthCredential, signal: Any
        ) -> OAuthCredential:
            return await _request_oauth_token(
                options.gateway,
                {
                    "grant_type": "refresh_token",
                    "client_id": OAUTH_CLIENT_ID,
                    "refresh_token": credential["refresh"],
                },
                signal,
            )

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return ModelAuth(api_key=credential["access"])

    return _RadiusOAuth()
