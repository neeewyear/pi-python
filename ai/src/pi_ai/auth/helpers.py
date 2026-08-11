"""认证辅助函数（对应 ``auth/helpers.ts``）。"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from .types import (
    ApiKeyAuth,
    ApiKeyCredential,
    AuthCheck,
    AuthResult,
    ModelAuth,
    OAuthAuth,
    OAuthCredential,
    ProviderAuthInteraction,
)


class _EnvApiKeyAuth:
    """标准 API Key 认证实现。"""

    def __init__(self, name: str, env_vars: list[str]) -> None:
        self.name = name
        self._env_vars = env_vars

    async def login(self, interaction: ProviderAuthInteraction) -> ApiKeyCredential:
        interaction.signal.throw_if_cancelled()
        key = await interaction.prompt(
            {
                "type": "secret",
                "message": f"Enter {self.name}",
                "signal": None,
                "placeholder": None,
                "options": [],
            }
        )
        interaction.signal.throw_if_cancelled()
        return {"type": "api_key", "key": key}

    async def check(self, input: dict[str, Any]) -> AuthCheck | None:
        return None

    async def resolve(self, input: dict[str, Any]) -> AuthResult | None:
        ctx: Any = input.get("ctx")
        credential: Any = input.get("credential")
        signal: Any = input.get("signal")
        if signal is not None:
            signal.throw_if_cancelled()
        if credential is not None and credential.get("key"):
            return {
                "auth": {"api_key": credential["key"]},
                "env": credential.get("env"),
                "source": "stored credential",
            }
        for env_var in self._env_vars:
            value = await ctx.env(env_var)
            if signal is not None:
                signal.throw_if_cancelled()
            if value:
                return {"auth": {"api_key": value}, "source": env_var}
        return None


class _LazyOAuthAuth:
    """延迟加载 OAuth 实现。"""

    def __init__(
        self,
        name: str,
        login_label: str | None,
        load: Callable[[], Awaitable[OAuthAuth]],
    ) -> None:
        self.name = name
        self.login_label = login_label
        self._load = load
        self._instance: OAuthAuth | None = None

    async def _loaded(self) -> OAuthAuth:
        if self._instance is None:
            self._instance = await self._load()
        return self._instance

    async def login(self, interaction: ProviderAuthInteraction) -> OAuthCredential:
        impl = await self._loaded()
        return await impl.login(interaction)

    async def refresh(
        self, credential: OAuthCredential, signal: Any
    ) -> OAuthCredential:
        impl = await self._loaded()
        return await impl.refresh(credential, signal)

    async def to_auth(self, credential: OAuthCredential) -> ModelAuth:
        impl = await self._loaded()
        return await impl.to_auth(credential)


def env_api_key_auth(name: str, env_vars: list[str]) -> ApiKeyAuth:
    """标准 API Key 认证辅助函数。

    创建一个 ``ApiKeyAuth`` 实现：
    - ``login`` 通过交互式 secret 提示输入 key
    - ``resolve`` 优先使用已存储的凭据，否则依次检查环境变量
    """
    return _EnvApiKeyAuth(name, env_vars)


def lazy_oauth_auth(
    name: str,
    login_label: str | None,
    load: Callable[[], Awaitable[OAuthAuth]],
) -> OAuthAuth:
    """延迟加载 OAuth 实现。

    包装一个动态导入的 ``OAuthAuth``，首次调用 ``login``/``refresh``/``toAuth`` 时加载。
    """
    return _LazyOAuthAuth(name, login_label, load)
