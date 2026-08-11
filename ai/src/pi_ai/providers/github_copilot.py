"""GitHub Copilot Provider（对应 ``github-copilot.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.openai_completions_lazy import openai_completions_api
from ..models import CreateProviderOptions, create_provider
from .github_copilot_models import GITHUB_COPILOT_MODELS


def github_copilot_provider() -> Any:
    """创建 GitHub Copilot Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="github-copilot",
            name="GitHub Copilot",
            base_url="https://api.individual.githubcopilot.com",
            models=list(GITHUB_COPILOT_MODELS.values()),
            api=openai_completions_api(),
        )
    )
