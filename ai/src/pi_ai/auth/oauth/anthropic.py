"""Anthropic OAuth flow (Claude Pro/Max)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any
from urllib.parse import parse_qs, urlparse

import httpx

from ...utils.provider_env import get_provider_env_value
from ..types import ModelAuth, OAuthAuth, OAuthCredential, ProviderAuthInteraction
from .oauth_page import oauth_error_html, oauth_success_html
from .pkce import generate_pkce

logger = logging.getLogger(__name__)

CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CALLBACK_HOST = get_provider_env_value("PI_OAUTH_CALLBACK_HOST") or "127.0.0.1"
CALLBACK_PORT = 53692
CALLBACK_PATH = "/callback"
REDIRECT_URI = f"http://localhost:{CALLBACK_PORT}{CALLBACK_PATH}"
SCOPES = (
    "org:create_api_key user:profile user:inference "
    "user:sessions:claude_code user:mcp_servers user:file_upload"
)


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


def _format_error_details(error: Exception) -> str:
    """Format error details for error messages."""
    details = [f"{type(error).__name__}: {error}"]
    if hasattr(error, "code") and error.code:
        details.append(f"code={error.code}")
    if hasattr(error, "errno") and error.errno is not None:
        details.append(f"errno={error.errno}")
    if error.args:
        details.append(f"cause={error.args[0]}")
    if hasattr(error, "stack") and error.stack:
        details.append(f"stack={error.stack}")
    return "; ".join(details)


async def _start_callback_server(expected_state: str) -> dict[str, Any]:
    """Start async HTTP callback server listening for OAuth callback."""
    code_future: asyncio.Future[dict[str, str]] = (
        asyncio.get_event_loop().create_future()
    )
    cancel_event = asyncio.Event()

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

            if parsed.path != CALLBACK_PATH:
                body = oauth_error_html("Callback route not found.")
                writer.write(
                    f"HTTP/1.1 404 Not Found\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            error_param = params.get("error", [None])[0]
            if error_param:
                body = oauth_error_html(
                    "Anthropic authentication did not complete.",
                    f"Error: {error_param}",
                )
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            code = params.get("code", [None])[0]
            state = params.get("state", [None])[0]

            if not code or not state:
                body = oauth_error_html("Missing code or state parameter.")
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            if state != expected_state:
                body = oauth_error_html("State mismatch.")
                writer.write(
                    f"HTTP/1.1 400 Bad Request\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
                )
                await writer.drain()
                writer.close()
                return

            body = oauth_success_html(
                "Anthropic authentication completed. You can close this window."
            )
            writer.write(
                f"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\nContent-Length: {len(body)}\r\n\r\n{body}".encode()
            )
            await writer.drain()
            writer.close()

            if not code_future.done():
                code_future.set_result({"code": code, "state": state})
        except Exception:
            if not code_future.done():
                code_future.cancel()
            writer.close()

    server = await asyncio.start_server(handle_request, CALLBACK_HOST, CALLBACK_PORT)

    async def wait_for_code() -> dict[str, str] | None:
        try:
            result = await asyncio.wait_for(code_future, timeout=300)
            return result
        except (asyncio.TimeoutError, asyncio.CancelledError):
            return None

    return {
        "server": server,
        "redirect_uri": REDIRECT_URI,
        "cancel_wait": lambda: code_future.cancel() if not code_future.done() else None,
        "wait_for_code": wait_for_code,
    }


async def _post_json(url: str, body: dict[str, str | int], signal: Any) -> str:
    """Make an HTTP POST request with JSON body."""
    if getattr(signal, "aborted", False):
        raise RuntimeError("Request was cancelled")

    async with httpx.AsyncClient() as client:
        try:
            response = await asyncio.wait_for(
                client.post(
                    url,
                    json=body,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json",
                    },
                ),
                timeout=30,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(f"Request to {url} timed out")
        except httpx.HTTPError as e:
            raise RuntimeError(
                f"HTTP request failed. url={url}; details={_format_error_details(e)}"
            )

    response_body = response.text
    if not response.is_success:
        raise RuntimeError(
            f"HTTP request failed. status={response.status_code}; url={url}; body={response_body}"
        )

    return response_body


async def _exchange_authorization_code(
    code: str,
    state: str,
    verifier: str,
    redirect_uri: str,
    signal: Any,
) -> OAuthCredential:
    """Exchange authorization code for OAuth tokens."""
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": verifier,
            },
            signal,
        )
    except Exception as error:
        raise RuntimeError(
            f"Token exchange request failed. url={TOKEN_URL}; redirect_uri={redirect_uri}; "
            f"response_type=authorization_code; details={_format_error_details(error)}"
        ) from error

    try:
        token_data = json.loads(response_body)
    except Exception as error:
        raise RuntimeError(
            f"Token exchange returned invalid JSON. url={TOKEN_URL}; body={response_body}; "
            f"details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        type="oauth",
        refresh=token_data["refresh_token"],
        access=token_data["access_token"],
        expires=int(time.time() * 1000)
        + token_data["expires_in"] * 1000
        - 5 * 60 * 1000,
    )


class _ManualAbort:
    """Simple abort signal for manual prompt cancellation."""

    def __init__(self) -> None:
        self.aborted = False


async def login_anthropic(interaction: ProviderAuthInteraction) -> OAuthCredential:
    """Login to Anthropic using OAuth PKCE flow."""
    pkce = await generate_pkce()
    verifier = pkce["verifier"]
    challenge = pkce["challenge"]

    server_info = await _start_callback_server(verifier)
    server = server_info["server"]
    cancel_wait = server_info["cancel_wait"]
    wait_for_code = server_info["wait_for_code"]

    code: str | None = None
    state: str | None = None
    manual_input: str | None = None
    manual_error: Exception | None = None
    manual_abort = _ManualAbort()

    try:
        auth_params = {
            "code": "true",
            "client_id": CLIENT_ID,
            "response_type": "code",
            "redirect_uri": REDIRECT_URI,
            "scope": SCOPES,
            "code_challenge": challenge,
            "code_challenge_method": "S256",
            "state": verifier,
        }
        from urllib.parse import urlencode

        auth_url = f"{AUTHORIZE_URL}?{urlencode(auth_params)}"
        interaction.notify(
            {
                "type": "auth_url",
                "url": auth_url,
                "instructions": (
                    "Complete login in your browser. "
                    "If the browser is on another machine, paste the final redirect URL here."
                ),
            }
        )

        async def do_manual_prompt() -> None:
            nonlocal manual_input, manual_error
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
                manual_input = result
                cancel_wait()
            except Exception as exc:
                manual_error = (
                    exc if isinstance(exc, Exception) else Exception(str(exc))
                )
                cancel_wait()

        manual_task = asyncio.create_task(do_manual_prompt())

        result = await wait_for_code()
        if result is not None and result.get("code"):
            code = result["code"]
            state = result.get("state")
        elif manual_input:
            parsed = parse_authorization_input(manual_input)
            if parsed.get("state") and parsed["state"] != verifier:
                raise RuntimeError("OAuth state mismatch")
            code = parsed.get("code")
            state = parsed.get("state", verifier)

        if not code:
            await manual_task
            if manual_error:
                raise manual_error
            if manual_input:
                parsed = parse_authorization_input(manual_input)
                if parsed.get("state") and parsed["state"] != verifier:
                    raise RuntimeError("OAuth state mismatch")
                code = parsed.get("code")
                state = parsed.get("state", verifier)

        if not code:
            raise RuntimeError("Missing authorization code")
        if not state:
            raise RuntimeError("Missing OAuth state")

        interaction.notify(
            {
                "type": "progress",
                "message": "Exchanging authorization code for tokens...",
            }
        )
        return await _exchange_authorization_code(
            code, state, verifier, REDIRECT_URI, interaction.signal
        )
    finally:
        manual_abort.aborted = True
        server.close()
        await server.wait_closed()


async def refresh_anthropic_token(refresh_token: str, signal: Any) -> OAuthCredential:
    """Refresh Anthropic OAuth token."""
    try:
        response_body = await _post_json(
            TOKEN_URL,
            {
                "grant_type": "refresh_token",
                "client_id": CLIENT_ID,
                "refresh_token": refresh_token,
            },
            signal,
        )
    except Exception as error:
        raise RuntimeError(
            f"Anthropic token refresh request failed. url={TOKEN_URL}; details={_format_error_details(error)}"
        ) from error

    try:
        data = json.loads(response_body)
    except Exception as error:
        raise RuntimeError(
            f"Anthropic token refresh returned invalid JSON. url={TOKEN_URL}; "
            f"body={response_body}; details={_format_error_details(error)}"
        ) from error

    return OAuthCredential(
        type="oauth",
        refresh=data["refresh_token"],
        access=data["access_token"],
        expires=int(time.time() * 1000) + data["expires_in"] * 1000 - 5 * 60 * 1000,
    )


class _AnthropicOAuthImpl:
    """Anthropic OAuthAuth implementation."""

    name: str = "Anthropic (Claude Pro/Max)"
    login_label: str | None = None

    async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
        return await login_anthropic(interaction)

    async def refresh(
        self, credential: OAuthCredential, signal: Any
    ) -> OAuthCredential:
        return await refresh_anthropic_token(credential["refresh"], signal)

    async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
        return ModelAuth(api_key=credential["access"])


anthropic_oauth: OAuthAuth = _AnthropicOAuthImpl()
