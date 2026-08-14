"""Google Provider。"""

from __future__ import annotations

from typing import Any

from ..api.google_generative_ai_lazy import google_generative_ai_api
from ..models import CreateProviderOptions, create_provider
from .google_models import GOOGLE_MODELS


def google_provider() -> Any:
    """创建 Google Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="google",
            name="Google",
            base_url="https://generativelanguage.googleapis.com/v1beta",
            models=list(GOOGLE_MODELS.values()),
            api=google_generative_ai_api(),
        )
    )
