"""pi-ai 认证模块。"""

from __future__ import annotations

from .context import default_provider_auth_context
from .credential_store import InMemoryCredentialStore
from .helpers import env_api_key_auth, lazy_oauth_auth
from .resolve import (
    AuthResolutionOverrides,
    ModelsError,
    ModelsErrorCode,
    ProviderAuthInfo,
    resolve_provider_auth,
)
from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthContext,
    AuthEvent,
    AuthInfoLink,
    AuthInteraction,
    AuthOperationOptions,
    AuthPrompt,
    AuthResult,
    AuthType,
    Credential,
    CredentialInfo,
    CredentialStore,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    OAuthCredentials,
    ProviderAuth,
    ProviderAuthInteraction,
)

__all__ = [
    "ApiKeyAuth",
    "ApiKeyCredential",
    "AuthCheck",
    "AuthContext",
    "AuthEvent",
    "AuthInfoLink",
    "AuthInteraction",
    "AuthOperationOptions",
    "AuthPrompt",
    "AuthResolutionOverrides",
    "AuthResult",
    "AuthType",
    "Credential",
    "CredentialInfo",
    "CredentialStore",
    "InMemoryCredentialStore",
    "ModelAuth",
    "ModelsError",
    "ModelsErrorCode",
    "OAuthAuth",
    "OAuthCredential",
    "OAuthCredentials",
    "ProviderAuth",
    "ProviderAuthInfo",
    "ProviderAuthInteraction",
    "default_provider_auth_context",
    "env_api_key_auth",
    "lazy_oauth_auth",
    "resolve_provider_auth",
]
