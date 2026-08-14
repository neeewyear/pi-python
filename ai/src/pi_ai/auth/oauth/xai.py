"""xAI OAuth 设备码流程。"""

from __future__ import annotations

import time
from typing import Any, cast
from urllib.parse import urlparse

import httpx

from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .device_code import (
    OAuthDeviceCodePollOptions,
    OAuthDeviceCodePollResult,
    poll_oauth_device_code_flow,
)

XAI_CLIENT_ID = "b1a00492-073a-47ea-816f-4c329264a828"
XAI_SCOPE = "openid profile email offline_access grok-cli:access api:access"
XAI_DEVICE_CODE_URL = "https://auth.x.ai/oauth2/device/code"
XAI_TOKEN_URL = "https://auth.x.ai/oauth2/token"
# Refresh slightly before the reported expiry to avoid using a token that dies mid-request.
REFRESH_SKEW_MS = 5 * 60 * 1000
DEFAULT_TOKEN_LIFETIME_SECONDS = 3600


def _required_string(body: dict[str, Any], field: str) -> str:
    """Get a required string field from a response body."""
    value = body.get(field)
    if not isinstance(value, str) or len(value) == 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return value


def _positive_number(body: dict[str, Any], field: str) -> int:
    """Get a required positive number field from a response body."""
    value = body.get(field)
    if not isinstance(value, (int, float)) or value <= 0:
        raise RuntimeError(f"Invalid xAI OAuth response field: {field}")
    return int(value)


def _validate_verification_uri(raw: str) -> str:
    """Validate that the verification URI is https."""
    try:
        parsed = urlparse(raw)
    except Exception:
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    if parsed.scheme != "https":
        raise RuntimeError("Untrusted verification URI in xAI OAuth response")
    return raw


async def _post_form(url: str, fields: dict[str, str], signal: Any) -> dict[str, Any]:
    """POST form-encoded data to the given URL."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data=fields,
                headers={
                    "Accept": "application/json",
                    "Content-Type": "application/x-www-form-urlencoded",
                },
            )
    except Exception:
        if signal and getattr(signal, "aborted", False):
            raise RuntimeError("Login cancelled") from None
        raise

    try:
        body = response.json()
    except Exception:
        if signal and getattr(signal, "aborted", False):
            raise RuntimeError("Login cancelled") from None
        raise RuntimeError(
            f"xAI OAuth returned invalid JSON (HTTP {response.status_code})"
        )

    if not isinstance(body, dict):
        body = {}

    return {
        "ok": response.is_success,
        "status": response.status_code,
        "body": body,
    }


def _request_failure(action: str, response_data: dict[str, Any]) -> RuntimeError:
    """Create an error for a failed request."""
    body = response_data.get("body", {}) if isinstance(response_data, dict) else {}
    if not isinstance(body, dict):
        body = {}
    error = body.get("error")
    description = body.get("error_description")
    error_str = str(error) if isinstance(error, str) else None
    desc_str = str(description) if isinstance(description, str) else None
    detail = ": ".join(filter(None, [error_str, desc_str]))
    return RuntimeError(
        f"xAI OAuth {action} failed (HTTP {response_data.get('status', '?')})"
        f"{f': {detail}' if detail else ''}"
    )


def _parse_device_code(body: dict[str, Any]) -> dict[str, Any]:
    """Parse device code response from xAI."""
    interval = body.get("interval")
    interval_seconds = (
        int(interval) if isinstance(interval, (int, float)) and interval > 0 else None
    )
    verification_uri_complete = body.get("verification_uri_complete")
    if (
        isinstance(verification_uri_complete, str)
        and len(verification_uri_complete) > 0
    ):
        verification_uri_complete = _validate_verification_uri(
            verification_uri_complete
        )
    else:
        verification_uri_complete = None

    return {
        "device_code": _required_string(body, "device_code"),
        "user_code": _required_string(body, "user_code"),
        "verification_uri": _validate_verification_uri(
            _required_string(body, "verification_uri")
        ),
        "verification_uri_complete": verification_uri_complete,
        "interval_seconds": interval_seconds,
        "expires_in_seconds": _positive_number(body, "expires_in"),
    }


def _credentials_from_token_response(
    body: dict[str, Any],
    previous_refresh_token: str | None = None,
) -> OAuthCredential:
    """Parse credentials from a token response."""
    access = _required_string(body, "access_token")
    # xAI may omit refresh_token on refresh when the token is not rotated.
    if body.get("refresh_token") is None and previous_refresh_token is not None:
        refresh = previous_refresh_token
    else:
        refresh = _required_string(body, "refresh_token")

    expires_in_seconds = (
        _positive_number(body, "expires_in")
        if body.get("expires_in") is not None
        else DEFAULT_TOKEN_LIFETIME_SECONDS
    )
    return {
        "type": "oauth",
        "access": access,
        "refresh": refresh,
        "expires": int(
            time.time() * 1000 + expires_in_seconds * 1000 - REFRESH_SKEW_MS
        ),
    }


async def _request_device_code(signal: Any) -> dict[str, Any]:
    """Request a device code from xAI."""
    response = await _post_form(
        XAI_DEVICE_CODE_URL,
        {
            "client_id": XAI_CLIENT_ID,
            "scope": XAI_SCOPE,
            "referrer": "pi",
        },
        signal,
    )
    if not response.get("ok"):
        raise _request_failure("device authorization", response)
    return _parse_device_code(response.get("body", {}))


async def _poll_for_tokens(device: dict[str, Any], signal: Any) -> OAuthCredential:
    """Poll for tokens using device code flow."""

    async def _poll(poll_device_code: str, sig: Any) -> OAuthDeviceCodePollResult:
        response = await _post_form(
            XAI_TOKEN_URL,
            {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": XAI_CLIENT_ID,
                "device_code": poll_device_code,
            },
            sig,
        )

        if response.get("ok"):
            return {
                "status": "complete",
                "value": _credentials_from_token_response(response.get("body", {})),
            }

        body = response.get("body", {})
        if not isinstance(body, dict):
            body = {}
        error = body.get("error")

        if error == "authorization_pending":
            return {"status": "pending"}
        if error == "slow_down":
            interval = body.get("interval")
            result: OAuthDeviceCodePollResult = {"status": "slow_down"}
            if isinstance(interval, (int, float)):
                result["interval_seconds"] = int(interval)
            return result
        if error in ("access_denied", "authorization_denied"):
            return {
                "status": "failed",
                "message": "xAI device authorization was denied",
            }
        if error == "expired_token":
            return {"status": "failed", "message": "xAI device code expired"}
        return {
            "status": "failed",
            "message": _request_failure("device token polling", response).args[0],
        }

    options: OAuthDeviceCodePollOptions = {
        "device_code": device["device_code"],
        "expires_in_seconds": device["expires_in_seconds"],
        "signal": signal,
        "poll": _poll,
    }
    interval = device.get("interval_seconds")
    if isinstance(interval, (int, float)):
        options["interval_seconds"] = int(interval)
    return cast(OAuthCredential, await poll_oauth_device_code_flow(options))


async def _login_xai(interaction: ProviderAuthInteraction) -> OAuthCredential:
    """Login to xAI."""
    device = await _request_device_code(interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": device.get("verification_uri_complete")
            or device["verification_uri"],
            "interval_seconds": device.get("interval_seconds"),
            "expires_in_seconds": device["expires_in_seconds"],
        }
    )
    return await _poll_for_tokens(device, interaction.signal)


async def _refresh_xai_token(refresh_token: str, signal: Any) -> OAuthCredential:
    """Refresh xAI token."""
    response = await _post_form(
        XAI_TOKEN_URL,
        {
            "grant_type": "refresh_token",
            "client_id": XAI_CLIENT_ID,
            "refresh_token": refresh_token,
        },
        signal,
    )
    if not response.get("ok"):
        raise _request_failure("token refresh", response)
    return _credentials_from_token_response(response.get("body", {}), refresh_token)


def _create_xai_oauth() -> OAuthAuth:
    """Create the xAI OAuthAuth instance."""

    class _XaiOAuth:
        name: str = "xAI (Grok/X subscription)"
        login_label: str | None = "Sign in with SuperGrok or X Premium"

        async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
            return await _login_xai(interaction)

        async def refresh(
            self, credential: OAuthCredential, signal: Any
        ) -> OAuthCredential:
            return await _refresh_xai_token(credential["refresh"], signal)

        async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
            return ModelAuth(api_key=credential["access"])

    return _XaiOAuth()


xai_oauth: OAuthAuth = _create_xai_oauth()
