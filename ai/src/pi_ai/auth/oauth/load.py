"""OAuth 动态加载器。"""


from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeAlias

from ..types import OAuthAuth


OAuthFlowLoaders: TypeAlias = dict[str, Callable[[], Awaitable[OAuthAuth]]]


_loaders: OAuthFlowLoaders = {}


def register_bundled_oauth_flow_loaders(loaders: OAuthFlowLoaders) -> None:
    """注册内置 OAuth 流程加载器。"""
    _loaders.update(loaders)


async def _load_from_registry(name: str) -> OAuthAuth:
    """从注册的加载器中加载 OAuth 流程。"""
    if name in _loaders:
        return await _loaders[name]()
    raise ImportError(f"No bundled OAuth loader registered for '{name}'")


async def load_anthropic_oauth() -> OAuthAuth:
    """加载 Anthropic OAuth 流程。"""
    return await _load_from_registry("anthropic")


async def load_openai_codex_oauth() -> OAuthAuth:
    """加载 OpenAI Codex OAuth 流程。"""
    return await _load_from_registry("openai_codex")


async def load_github_copilot_oauth() -> OAuthAuth:
    """加载 GitHub Copilot OAuth 流程。"""
    return await _load_from_registry("github_copilot")


async def load_openrouter_oauth() -> OAuthAuth:
    """加载 OpenRouter OAuth 流程。"""
    return await _load_from_registry("openrouter")


async def load_kimi_coding_oauth() -> OAuthAuth:
    """加载 Kimi Coding OAuth 流程。"""
    return await _load_from_registry("kimi_coding")


async def load_xai_oauth() -> OAuthAuth:
    """加载 xAI OAuth 流程。"""
    return await _load_from_registry("xai")


async def load_radius_oauth(options: dict[str, str]) -> OAuthAuth:
    """加载 Radius OAuth 流程。"""
    name = options.get("name", "radius")
    return await _load_from_registry(name)