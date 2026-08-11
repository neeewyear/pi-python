"""Bun 运行时 OAuth 注册（对应 ``bun-oauth.ts``）。"""

from __future__ import annotations

from typing import Any, cast

from .auth.oauth.anthropic import anthropic_oauth
from .auth.oauth.github_copilot import github_copilot_oauth
from .auth.oauth.kimi_coding import kimi_coding_oauth
from .auth.oauth.load import OAuthFlowLoaders, register_bundled_oauth_flow_loaders
from .auth.oauth.openai_codex import openai_codex_oauth
from .auth.oauth.openrouter import openrouter_oauth
from .auth.oauth.radius import create_radius_oauth
from .auth.oauth.xai import xai_oauth


def register_bun_oauth_flows() -> None:
    """注册 Bun 二进制内置的 OAuth 流程。"""
    register_bundled_oauth_flow_loaders(
        cast(
            OAuthFlowLoaders,
            {
                "anthropic": lambda: anthropic_oauth,
                "openai_codex": lambda: openai_codex_oauth,
                "github_copilot": lambda: github_copilot_oauth,
                "openrouter": lambda: openrouter_oauth,
                "kimi_coding": lambda: kimi_coding_oauth,
                "xai": lambda: xai_oauth,
                "radius": lambda: create_radius_oauth(cast(Any, {})),
            },
        )
    )
