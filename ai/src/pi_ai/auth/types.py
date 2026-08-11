"""认证类型定义（对应 ``auth/types.ts``）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, Literal, Protocol, TypedDict, runtime_checkable

from ..types import ProviderEnv, ProviderHeaders


class ModelAuth(TypedDict, total=False):
    """Model 请求认证。"""

    api_key: str
    headers: ProviderHeaders | None
    base_url: str


class ApiKeyCredential(TypedDict, total=False):
    """API Key 凭据。"""

    type: Literal["api_key"]
    key: str | None
    env: ProviderEnv | None


class OAuthCredentials(TypedDict, total=False):
    """OAuth token 数据。"""

    refresh: str
    access: str
    expires: int
    account_id: str
    enterprise_url: str  # GitHub Copilot enterprise domain
    available_model_ids: list[str]  # 可用模型 ID 列表
    scope: str  # OAuth scope


class OAuthCredential(OAuthCredentials):
    """OAuth 凭据。"""

    type: Literal["oauth"]


Credential = ApiKeyCredential | OAuthCredential


class CredentialInfo(TypedDict):
    """凭据元信息。"""

    provider_id: str
    type: Literal["api_key", "oauth"]


class AuthOperationOptions(TypedDict, total=False):
    """认证操作选项。"""

    signal: Any  # AbortSignal


@runtime_checkable
class CredentialStore(Protocol):
    """凭据存储接口。"""

    async def read(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> Credential | None: ...
    async def list(
        self, options: AuthOperationOptions | None = None
    ) -> list[CredentialInfo]: ...
    async def modify(
        self,
        provider_id: str,
        fn: Callable[[Credential | None], Awaitable[Credential | None]],
        options: AuthOperationOptions | None = None,
    ) -> Credential | None: ...
    async def delete(
        self, provider_id: str, options: AuthOperationOptions | None = None
    ) -> None: ...


@runtime_checkable
class AuthContext(Protocol):
    """认证上下文（环境变量访问）。"""

    async def env(self, name: str) -> str | None: ...
    async def file_exists(self, path: str) -> bool: ...


class AuthResult(TypedDict, total=False):
    """认证结果。"""

    auth: ModelAuth
    env: ProviderEnv | None
    source: str


class AuthCheck(TypedDict, total=False):
    """认证检查结果。"""

    source: str | None
    type: Literal["api_key", "oauth"]


AuthType = Literal["api_key", "oauth"]


class AuthPrompt(TypedDict, total=False):
    """认证提示。"""

    signal: Any  # AbortSignal
    type: Literal["text", "secret", "select", "manual_code"]
    message: str
    placeholder: str | None
    options: list[dict[str, str]]  # for "select" type: {id, label, description?}


class AuthInfoLink(TypedDict, total=False):
    """认证信息链接。"""

    url: str
    label: str | None


class AuthEvent(TypedDict, total=False):
    """认证事件。"""

    type: Literal["info", "auth_url", "device_code", "progress"]
    message: str | None
    links: list[AuthInfoLink] | None
    url: str | None  # for auth_url
    instructions: str | None  # for auth_url
    user_code: str | None  # for device_code
    verification_uri: str | None  # for device_code
    interval_seconds: int | None  # for device_code
    expires_in_seconds: int | None  # for device_code


@runtime_checkable
class AuthInteraction(Protocol):
    """认证交互接口。"""

    signal: Any  # AbortSignal

    async def prompt(self, prompt: AuthPrompt) -> str: ...
    def notify(self, event: AuthEvent) -> None: ...


ProviderAuthInteraction = AuthInteraction  # 带 signal 的规范化接口


@runtime_checkable
class ApiKeyAuth(Protocol):
    """API Key 认证。"""

    name: str

    async def login(self, interaction: ProviderAuthInteraction) -> ApiKeyCredential: ...
    async def check(self, input: dict[str, Any]) -> AuthCheck | None: ...
    async def resolve(self, input: dict[str, Any]) -> AuthResult | None: ...


@runtime_checkable
class OAuthAuth(Protocol):
    """OAuth 认证。"""

    name: str
    login_label: str | None

    async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential: ...
    async def refresh(
        self, credential: OAuthCredential, signal: Any
    ) -> OAuthCredential: ...
    async def to_auth(self, credential: OAuthCredential) -> ModelAuth: ...


class ProviderAuth(TypedDict, total=False):
    """Provider 认证。"""

    api_key: ApiKeyAuth | None
    oauth: OAuthAuth | None
