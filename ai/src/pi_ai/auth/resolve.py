"""Provider 认证解析。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Literal, TypeAlias, TypedDict

from ..types import ProviderEnv
from ..utils.abort import (
    CancellationToken,
    combine_abort_signals,
    operation_signal,
    race_with_abort_signal,
)
from ..utils.diagnostics import format_thrown_value
from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthContext,
    AuthResult,
    Credential,
    CredentialStore,
    OAuthAuth,
    OAuthCredential,
    ProviderAuth,
)

ModelsErrorCode: TypeAlias = Literal[
    "model_source",
    "model_validation",
    "provider",
    "stream",
    "auth",
    "oauth",
]


class AuthResolutionOverrides(TypedDict, total=False):
    """认证解析覆盖选项。"""

    api_key: str | None
    env: ProviderEnv | None
    min_oauth_validity_ms: int | None
    signal: Any  # AbortSignal


class ProviderAuthInfo(TypedDict):
    """Provider 认证信息。"""

    id: str
    auth: ProviderAuth


class ModelsError(Exception):
    """Models 错误。"""

    def __init__(
        self, code: ModelsErrorCode, message: str, *, cause: Any = None
    ) -> None:
        self.code = code
        detail = format_thrown_value(cause).strip() if cause is not None else ""
        msg = f"{message}: {detail}" if detail and detail not in message else message
        super().__init__(msg)
        self.cause = cause


DEFAULT_OAUTH_MINIMUM_VALIDITY_MS = 5 * 60 * 1000
DEFAULT_OAUTH_REFRESH_TIMEOUT_MS = 15_000


async def resolve_provider_auth(
    provider: ProviderAuthInfo,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None = None,
) -> AuthResult | None:
    """解析 provider 认证。

    优先使用已存储的凭据，否则回退到环境变量/ambient 来源。
    """
    signal = operation_signal(overrides.get("signal") if overrides else None)
    return await race_with_abort_signal(  # type: ignore[return-value]
        _resolve_provider_auth_with_signal(
            provider, credentials, auth_context, overrides, signal
        ),
        signal,
    )


async def _resolve_provider_auth_with_signal(
    provider: ProviderAuthInfo,
    credentials: CredentialStore,
    auth_context: AuthContext,
    overrides: AuthResolutionOverrides | None,
    signal: CancellationToken,
) -> AuthResult | None:
    signal.throw_if_cancelled()

    override_env = overrides.get("env") if overrides else None
    request_auth_context = (
        _overlay_env_auth_context(auth_context, override_env)
        if override_env is not None
        else auth_context
    )

    # API key 覆盖
    override_api_key = overrides.get("api_key") if overrides else None
    provider_api_key = provider["auth"].get("api_key")
    provider_oauth = provider["auth"].get("oauth")

    if override_api_key is not None and provider_api_key is not None:
        return await _resolve_api_key(
            request_auth_context,
            provider_api_key,
            provider["id"],
            {
                "type": "api_key",
                "key": override_api_key,
                "env": override_env,
            },
            signal,
        )

    # 读取已存储的凭据
    stored = await _read_credential(credentials, provider["id"], signal)
    if stored is not None:
        if stored["type"] == "oauth" and provider_oauth is not None:
            return await _resolve_stored_oauth(
                credentials,
                provider["id"],
                provider_oauth,
                stored,
                signal,
                overrides.get("min_oauth_validity_ms") if overrides else None,
            )
        if stored["type"] == "api_key" and provider_api_key is not None:
            cred: ApiKeyCredential = stored
            if override_env is not None:
                cred = {
                    **cred,
                    "env": {**(cred.get("env") or {}), **override_env},
                }
            return await _resolve_api_key(
                request_auth_context,
                provider_api_key,
                provider["id"],
                cred,
                signal,
            )
        return None

    # Ambient 来源（环境变量等）
    if provider_api_key is not None:
        return await _resolve_api_key(
            request_auth_context,
            provider_api_key,
            provider["id"],
            None,
            signal,
        )
    return None


def _overlay_env_auth_context(base: AuthContext, env: ProviderEnv) -> AuthContext:
    """创建覆盖环境变量的 AuthContext。"""

    class _OverlayAuthContext:
        async def env(self, name: str) -> str | None:
            if name in env:
                return env[name]
            return await base.env(name)

        async def file_exists(self, path: str) -> bool:
            return await base.file_exists(path)

    return _OverlayAuthContext()


async def _resolve_stored_oauth(
    credentials: CredentialStore,
    provider_id: str,
    oauth: OAuthAuth,
    stored: OAuthCredential,
    signal: CancellationToken,
    min_oauth_validity_ms: int | None = None,
) -> AuthResult | None:
    """OAuth 凭据解析，含过期刷新和双检锁。"""
    minimum_validity_ms = max(
        DEFAULT_OAUTH_MINIMUM_VALIDITY_MS, min_oauth_validity_ms or 0
    )

    def _expires_soon(credential: OAuthCredential) -> bool:
        return (time.time() * 1000) + minimum_validity_ms >= credential["expires"]

    credential = stored

    if _expires_soon(credential):
        # 乐观检查认为已过期；权威检查在 modify 锁下进行
        post: Credential | None = None
        try:
            post = await credentials.modify(
                provider_id,
                _make_refresh_fn(oauth, provider_id, signal, minimum_validity_ms),
                {"signal": signal},
            )
        except Exception as error:
            if isinstance(error, ModelsError):
                raise
            raise ModelsError(
                "auth",
                f"Credential store modify failed for {provider_id}",
                cause=error,
            )

        if post is None or post.get("type") != "oauth":
            return None  # 期间已登出
        credential = post  # type: ignore[assignment]

        if min_oauth_validity_ms is not None and _expires_soon(credential):
            raise ModelsError(
                "oauth",
                f"OAuth refresh returned a token that expires too soon for {provider_id}",
            )

    try:
        return {"auth": await oauth.to_auth(credential), "source": "OAuth"}
    except Exception as error:
        raise ModelsError(
            "oauth",
            f"OAuth auth derivation failed for {provider_id}",
            cause=error,
        )


def _make_refresh_fn(
    oauth: OAuthAuth,
    provider_id: str,
    signal: CancellationToken,
    minimum_validity_ms: int,
) -> Any:
    """创建 OAuth refresh 的 modify 回调。"""

    def _expires_soon(credential: OAuthCredential) -> bool:
        return (time.time() * 1000) + minimum_validity_ms >= credential["expires"]

    async def _refresh_fn(current: Credential | None) -> Credential | None:
        if current is None or current.get("type") != "oauth":
            return None  # 期间已登出
        oauth_current: OAuthCredential = current  # type: ignore[assignment]
        if not _expires_soon(oauth_current):
            return None  # 另一个进程/请求已刷新
        try:
            # 创建带超时的组合取消信号
            timeout_token = CancellationToken()
            _schedule_timeout(timeout_token, DEFAULT_OAUTH_REFRESH_TIMEOUT_MS)
            combined = combine_abort_signals([signal, timeout_token])
            refresh_signal = combined.signal or CancellationToken()

            return await oauth.refresh(oauth_current, refresh_signal)
        except Exception as error:
            raise ModelsError(
                "oauth",
                f"OAuth refresh failed for {provider_id}",
                cause=error,
            )

    return _refresh_fn


def _schedule_timeout(token: CancellationToken, timeout_ms: int) -> None:
    """在指定超时后取消令牌。"""

    async def _timeout() -> None:
        await asyncio.sleep(timeout_ms / 1000.0)
        token.cancel()

    asyncio.ensure_future(_timeout())


async def _resolve_api_key(
    auth_context: AuthContext,
    api_key: ApiKeyAuth,
    provider_id: str,
    credential: ApiKeyCredential | None,
    signal: CancellationToken,
) -> AuthResult | None:
    """解析 API Key 凭据。"""
    try:
        return await api_key.resolve(
            {
                "ctx": auth_context,
                "credential": credential,
                "signal": signal,
            }
        )
    except Exception as error:
        raise ModelsError(
            "auth",
            f"API key auth failed for provider {provider_id}",
            cause=error,
        )


async def _read_credential(
    credentials: CredentialStore,
    provider_id: str,
    signal: CancellationToken,
) -> Credential | None:
    """从凭据存储读取凭据。"""
    try:
        return await credentials.read(provider_id, {"signal": signal})
    except Exception as error:
        raise ModelsError(
            "auth",
            f"Credential store read failed for {provider_id}",
            cause=error,
        )
