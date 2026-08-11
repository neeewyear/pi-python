"""OpenAI Codex (ChatGPT OAuth) flow."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import secrets
import time
from typing import Any, cast
from urllib.parse import parse_qs, urlencode, urlparse

import httpx

from ...utils.provider_env import get_provider_env_value
from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .device_code import OAuthDeviceCodePollResult, poll_oauth_device_code_flow
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

logger = logging.getLogger(__name__)

CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
AUTH_BASE_URL = "https://auth.openai.com"
AUTHORIZE_URL = f"{AUTH_BASE_URL}/oauth/authorize"
TOKEN_URL = f"{AUTH_BASE_URL}/oauth/token"
REDIRECT_URI = "http://localhost:1455/auth/callback"
DEVICE_USER_CODE_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/usercode"
DEVICE_TOKEN_URL = f"{AUTH_BASE_URL}/api/accounts/deviceauth/token"
DEVICE_VERIFICATION_URI = f"{AUTH_BASE_URL}/codex/device"
DEVICE_REDIRECT_URI = f"{AUTH_BASE_URL}/deviceauth/callback"
DEVICE_CODE_TIMEOUT_SECONDS = 15 * 60
OPENAI_CODEX_BROWSER_LOGIN_METHOD = "browser"
OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD = "device_code"
SCOPE = "openid profile email offline_access"
JWT_CLAIM_PATH = "https://api.openai.com/auth"


def _get_callback_host() -> str:
    """Get the callback host for the local OAuth server."""
    return get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"


def _create_state() -> str:
    """Create a random state string for OAuth."""
    return secrets.token_hex(16)


def parse_authorization_input(input_str: str) -> dict[str, str]:
    """Parse authorization code from user input (URL or raw code)."""
    value = input_str.strip()
    if not value:
        return {}

    # Try parsing as URL
    if "://" in value:
        try:
            parsed = urlparse(value)
            params = parse_qs(parsed.query)
            result: dict[str, str] = {}
            if "code" in params:
                result["code"] = params["code"][0]
            if "state" in params:
                result["state"] = params["state"][0]
            return result
        except Exception:
            pass

    # Try hash format: code#state
    if "#" in value:
        parts = value.split("#", 1)
        result = {"code": parts[0]}
        if len(parts) > 1 and parts[1]:
            result["state"] = parts[1]
        return result

    # Try query string format
    if "code=" in value:
        params = parse_qs(value)
        qs_result: dict[str, str] = {}
        if "code" in params:
            qs_result["code"] = params["code"][0]
        if "state" in params:
            qs_result["state"] = params["state"][0]
        return qs_result

    return {"code": value}


def _decode_jwt(token: str) -> dict[str, Any] | None:
    """Decode a JWT token payload (without verification)."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        payload = parts[1]
        # Add padding for base64url decoding
        padding = 4 - len(payload) % 4
        if padding != 4:
            payload += "=" * padding
        decoded = base64.urlsafe_b64decode(payload)
        return cast("dict[str, Any]", json.loads(decoded))
    except Exception:
        return None


async def _fetch_with_login_cancellation(
    url: str,
    method: str,
    headers: dict[str, str] | None = None,
    json_body: dict[str, Any] | None = None,
    form_body: dict[str, str] | None = None,
    signal: Any = None,
) -> httpx.Response:
    """Make an HTTP request with cancellation support."""
    if getattr(signal, "aborted", False):
        raise RuntimeError("Login cancelled")

    async with httpx.AsyncClient() as client:
        try:
            if form_body is not None:
                response = await client.post(url, data=form_body, headers=headers)
            elif json_body is not None:
                response = await client.post(url, json=json_body, headers=headers)
            else:
                response = await client.request(method, url, headers=headers)
            return response
        except httpx.HTTPError as e:
            if getattr(signal, "aborted", False):
                raise RuntimeError("Login cancelled") from e
            raise


async def _read_token_response(
    response: httpx.Response,
    operation: str,
) -> dict[str, str | int]:
    """Read and validate token response."""
    if not response.is_success:
        text = ""
        try:
            text = response.text
        except Exception:
            pass
        raise RuntimeError(
            f"OpenAI Codex token {operation} failed ({response.status_code}): "
            f"{text or response.reason_phrase}"
        )

    raw_json = response.json()
    json_data = raw_json if isinstance(raw_json, dict) else {}
    access_token = cast(str, json_data.get("access_token"))
    refresh_token = cast(str, json_data.get("refresh_token"))
    expires_in = json_data.get("expires_in")

    if (
        not access_token
        or not refresh_token
        or not isinstance(expires_in, (int, float))
    ):
        raise RuntimeError(
            f"OpenAI Codex token {operation} response missing fields: {json.dumps(json_data)}"
        )

    return {
        "access": access_token,
        "refresh": refresh_token,
        "expires": int(time.time() * 1000) + int(expires_in) * 1000,
    }


async def _exchange_authorization_code(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Any,
) -> dict[str, str | int]:
    """Exchange authorization code for OAuth tokens."""
    response = await _fetch_with_login_cancellation(
        TOKEN_URL,
        "POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        form_body={
            "grant_type": "authorization_code",
            "client_id": CLIENT_ID,
            "code": code,
            "code_verifier": verifier,
            "redirect_uri": redirect_uri,
        },
        signal=signal,
    )
    return await _read_token_response(response, "exchange")


async def _refresh_access_token(
    refresh_token: str,
    signal: Any,
) -> dict[str, str | int]:
    """Refresh OpenAI Codex OAuth token."""
    try:
        response = await _fetch_with_login_cancellation(
            TOKEN_URL,
            "POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            form_body={
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": CLIENT_ID,
            },
            signal=signal,
        )
    except Exception as error:
        raise RuntimeError(f"OpenAI Codex token refresh error: {error}") from error

    return await _read_token_response(response, "refresh")


async def _start_openai_codex_device_auth(
    signal: Any,
) -> dict[str, Any]:
    """Start device code authentication flow."""
    response = await _fetch_with_login_cancellation(
        DEVICE_USER_CODE_URL,
        "POST",
        headers={"Content-Type": "application/json"},
        json_body={"client_id": CLIENT_ID},
        signal=signal,
    )

    if not response.is_success:
        if response.status_code == 404:
            raise RuntimeError(
                "OpenAI Codex device code login is not enabled for this server. "
                "Use browser login or verify the server URL."
            )
        response_body = ""
        try:
            response_body = response.text
        except Exception:
            pass
        raise RuntimeError(
            f"OpenAI Codex device code request failed with status {response.status_code}"
            f"{f': {response_body}' if response_body else ''}"
        )

    raw_json = response.json()
    json_data = raw_json if isinstance(raw_json, dict) else {}
    device_auth_id = json_data.get("device_auth_id")
    user_code = json_data.get("user_code")
    interval_raw = json_data.get("interval")

    interval_seconds: int | float | None = None
    if isinstance(interval_raw, str):
        try:
            interval_seconds = int(interval_raw.strip())
        except (ValueError, TypeError):
            interval_seconds = None
    elif isinstance(interval_raw, (int, float)):
        interval_seconds = interval_raw

    if (
        not device_auth_id
        or not user_code
        or not isinstance(interval_seconds, (int, float))
        or not (interval_seconds >= 0)
    ):
        raise RuntimeError(
            f"Invalid OpenAI Codex device code response: {json.dumps(json_data)}"
        )

    return {
        "device_auth_id": device_auth_id,
        "user_code": user_code,
        "interval_seconds": int(interval_seconds),
    }


async def _poll_openai_codex_device_auth(
    device: dict[str, Any],
    signal: Any,
) -> dict[str, str]:
    """Poll device code authentication until completion."""
    device_auth_id = device["device_auth_id"]
    user_code = device["user_code"]

    async def poll_fn(device_code: str, poll_signal: Any) -> OAuthDeviceCodePollResult:
        response = await _fetch_with_login_cancellation(
            DEVICE_TOKEN_URL,
            "POST",
            headers={"Content-Type": "application/json"},
            json_body={
                "device_auth_id": device_auth_id,
                "user_code": user_code,
            },
            signal=poll_signal,
        )

        if response.is_success:
            raw_json = response.json()
            json_data = raw_json if isinstance(raw_json, dict) else {}
            authorization_code = json_data.get("authorization_code")
            code_verifier = json_data.get("code_verifier")
            if not authorization_code or not code_verifier:
                return OAuthDeviceCodePollResult(
                    status="failed",
                    message=(
                        f"Invalid OpenAI Codex device auth token response: "
                        f"{json.dumps(json_data)}"
                    ),
                )
            return OAuthDeviceCodePollResult(
                status="complete",
                value={
                    "authorization_code": authorization_code,
                    "code_verifier": code_verifier,
                },
            )

        if response.status_code in (403, 404):
            return OAuthDeviceCodePollResult(status="incomplete")

        response_body = ""
        try:
            response_body = response.text
        except Exception:
            pass

        error_code: Any = None
        try:
            error_json = response.json()
            error_data = error_json if isinstance(error_json, dict) else {}
            error_field = error_data.get("error")
            if isinstance(error_field, dict):
                error_code = error_field.get("code")
            else:
                error_code = error_field
        except Exception:
            pass

        if error_code == "deviceauth_authorization_pending":
            return OAuthDeviceCodePollResult(status="incomplete")
        if error_code == "slow_down":
            return OAuthDeviceCodePollResult(status="slow_down")

        return OAuthDeviceCodePollResult(
            status="failed",
            message=(
                f"OpenAI Codex device auth failed with status {response.status_code}"
                f"{f': {response_body}' if response_body else ''}"
            ),
        )

    result = await poll_oauth_device_code_flow(
        {
            "interval_seconds": device["interval_seconds"],
            "expires_in_seconds": DEVICE_CODE_TIMEOUT_SECONDS,
            "signal": signal,
            "poll": poll_fn,
        }
    )

    return cast("dict[str, str]", result)


async def _create_authorization_flow(
    originator: str = "pi",
) -> dict[str, str]:
    """Create OAuth authorization flow parameters."""
    pkce = await generate_pkce()
    verifier = pkce["verifier"]
    challenge = pkce["challenge"]
    state = _create_state()

    params = {
        "response_type": "code",
        "client_id": CLIENT_ID,
        "redirect_uri": REDIRECT_URI,
        "scope": SCOPE,
        "code_challenge": challenge,
        "code_challenge_method": "S256",
        "state": state,
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "originator": originator,
    }
    url = f"{AUTHORIZE_URL}?{urlencode(params)}"

    return {"verifier": verifier, "state": state, "url": url}


async def _start_local_oauth_server(state: str) -> dict[str, Any]:
    """Start a local HTTP server for OAuth callback on port 1455."""
    code_future: asyncio.Future[dict[str, str]] = (
        asyncio.get_event_loop().create_future()
    )

    async def handle_request(
        reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            request_line = (await reader.readline()).decode("utf-8").strip()
            # Read headers
            while True:
                line = (await reader.readline()).decode("utf-8").strip()
                if not line:
                    break

            parts = request_line.split()
            if len(parts) < 2:
                writer.close()
                return

            path = parts[1]
            parsed = urlparse(path)
            params = parse_qs(parsed.query)

            if parsed.path != "/auth/callback":
                body = oauth_error_html("Callback route not found.")
                writer.write(
                    f"HTTP/1.1 404 Not Found\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            if params.get("state", [None])[0] != state:
                body = oauth_error_html("State mismatch.")
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            code = params.get("code", [None])[0]
            if not code:
                body = oauth_error_html("Missing authorization code.")
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            body = oauth_success_html(
                "OpenAI authentication completed. You can close this window."
            )
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
            )
            await writer.drain()
            writer.close()

            if not code_future.done():
                code_future.set_result({"code": code})
        except Exception:
            if not code_future.done():
                code_future.cancel()
            writer.close()

    callback_host = _get_callback_host()
    try:
        server = await asyncio.start_server(handle_request, callback_host, 1455)
    except OSError:
        # Port in use or other error - return dummy server
        return {
            "close": lambda: None,
            "cancel_wait": lambda: None,
            "wait_for_code": _async_none,
        }

    async def wait_for_code() -> dict[str, str] | None:
        try:
            result = await asyncio.wait_for(code_future, timeout=300)
            return result
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

    return {
        "close": lambda: server.close(),
        "cancel_wait": lambda: code_future.cancel() if not code_future.done() else None,
        "wait_for_code": wait_for_code,
    }


async def _async_none() -> None:
    """Return None for dummy server wait_for_code."""
    return


def _get_account_id(access_token: str) -> str | None:
    """Extract account ID from JWT token."""
    payload = _decode_jwt(access_token)
    if payload is None:
        return None
    auth_claim = payload.get(JWT_CLAIM_PATH)
    if not isinstance(auth_claim, dict):
        return None
    account_id = auth_claim.get("chatgpt_account_id")
    if isinstance(account_id, str) and len(account_id) > 0:
        return account_id
    return None


def _credentials_from_token(token: dict[str, str | int]) -> OAuthCredential:
    """Create OAuthCredential from token data."""
    access = token["access"]
    if not isinstance(access, str):
        raise RuntimeError("Invalid access token type")

    account_id = _get_account_id(access)
    if not account_id:
        raise RuntimeError("Failed to extract accountId from token")

    return OAuthCredential(
        type="oauth",
        access=access,
        refresh=cast(str, token["refresh"]),
        expires=cast(int, token["expires"]),
        account_id=account_id,
    )


async def _exchange_authorization_code_for_credentials(
    code: str,
    verifier: str,
    redirect_uri: str,
    signal: Any,
) -> OAuthCredential:
    """Exchange authorization code for credentials."""
    token = await _exchange_authorization_code(code, verifier, redirect_uri, signal)
    return _credentials_from_token(token)


class _ManualAbort:
    """Simple abort signal for manual prompt cancellation."""

    def __init__(self) -> None:
        self.aborted = False


async def _login_openai_codex_device_code(
    interaction: ProviderAuthInteraction,
) -> OAuthCredential:
    """Login using device code flow."""
    device = await _start_openai_codex_device_auth(interaction.signal)
    interaction.notify(
        {
            "type": "device_code",
            "user_code": device["user_code"],
            "verification_uri": DEVICE_VERIFICATION_URI,
            "interval_seconds": device["interval_seconds"],
            "expires_in_seconds": DEVICE_CODE_TIMEOUT_SECONDS,
        }
    )
    code = await _poll_openai_codex_device_auth(device, interaction.signal)
    return await _exchange_authorization_code_for_credentials(
        code["authorization_code"],
        code["code_verifier"],
        DEVICE_REDIRECT_URI,
        interaction.signal,
    )


async def _login_openai_codex(interaction: ProviderAuthInteraction) -> OAuthCredential:
    """Login using browser-based OAuth PKCE flow."""
    flow = await _create_authorization_flow()
    verifier = flow["verifier"]
    state = flow["state"]
    url = flow["url"]

    server_info = await _start_local_oauth_server(state)
    cancel_wait = server_info["cancel_wait"]
    wait_for_code = server_info["wait_for_code"]

    code: str | None = None
    manual_code: str | None = None
    manual_error: Exception | None = None
    manual_abort = _ManualAbort()

    interaction.notify(
        {
            "type": "auth_url",
            "url": url,
            "instructions": "A browser window should open. Complete login to finish.",
        }
    )

    try:

        async def do_manual_prompt() -> None:
            nonlocal manual_code, manual_error
            try:
                result = await interaction.prompt(
                    {
                        "type": "manual_code",
                        "message": (
                            "Complete login in your browser, "
                            "or paste the authorization code / redirect URL here:"
                        ),
                        "placeholder": REDIRECT_URI,
                        "signal": manual_abort,
                    }
                )
                manual_code = result
                cancel_wait()
            except Exception as exc:
                manual_error = (
                    exc if isinstance(exc, Exception) else Exception(str(exc))
                )
                cancel_wait()

        manual_task = asyncio.create_task(do_manual_prompt())

        result = await wait_for_code()
        if manual_error:
            raise manual_error
        if result is not None and result.get("code"):
            code = result["code"]
        elif manual_code:
            parsed = parse_authorization_input(manual_code)
            if parsed.get("state") and parsed["state"] != state:
                raise RuntimeError("State mismatch")
            code = parsed.get("code")

        if not code:
            await manual_task
            if manual_error:
                raise manual_error
            if manual_code:
                parsed = parse_authorization_input(manual_code)
                if parsed.get("state") and parsed["state"] != state:
                    raise RuntimeError("State mismatch")
                code = parsed.get("code")

        if not code:
            raise RuntimeError("Missing authorization code")

        return await _exchange_authorization_code_for_credentials(
            code, verifier, REDIRECT_URI, interaction.signal
        )
    finally:
        manual_abort.aborted = True
        server_info["close"]()


async def _login_openai_codex_main(
    interaction: ProviderAuthInteraction,
) -> OAuthCredential:
    """Main login function that prompts for login method."""
    method = await interaction.prompt(
        {
            "type": "select",
            "message": "Select OpenAI Codex login method:",
            "options": [
                {
                    "id": OPENAI_CODEX_BROWSER_LOGIN_METHOD,
                    "label": "Browser login (default)",
                },
                {
                    "id": OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD,
                    "label": "Device code login (headless)",
                },
            ],
        }
    )

    if method == OPENAI_CODEX_DEVICE_CODE_LOGIN_METHOD:
        return await _login_openai_codex_device_code(interaction)
    if method != OPENAI_CODEX_BROWSER_LOGIN_METHOD:
        raise RuntimeError(f"Unknown OpenAI Codex login method: {method}")

    return await _login_openai_codex(interaction)


async def _refresh_openai_codex_token(
    refresh_token: str,
    signal: Any,
) -> OAuthCredential:
    """Refresh OpenAI Codex OAuth token."""
    return _credentials_from_token(await _refresh_access_token(refresh_token, signal))


async def _openai_codex_to_auth(credential: OAuthCredential) -> ModelAuth:
    """Convert OAuth credential to ModelAuth."""
    return ModelAuth(api_key=credential["access"])


class _OpenAICodexOAuthImpl:
    """OpenAI Codex OAuthAuth implementation."""

    name: str = "OpenAI (ChatGPT Plus/Pro)"
    login_label: str | None = None

    async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
        return await _login_openai_codex_main(interaction)

    async def refresh(
        self, credential: OAuthCredential, signal: Any
    ) -> OAuthCredential:
        return await _refresh_openai_codex_token(credential["refresh"], signal)

    async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


openai_codex_oauth: OAuthAuth = _OpenAICodexOAuthImpl()
