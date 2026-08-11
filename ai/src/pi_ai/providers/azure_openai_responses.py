"""Azure OpenAI Responses Provider（对应 ``azure-openai-responses.ts``）。"""

from __future__ import annotations

from typing import Any

from ..api.azure_openai_responses_lazy import azure_openai_responses_api
from ..models import CreateProviderOptions, create_provider
from .azure_openai_responses_models import AZURE_OPENAI_RESPONSES_MODELS


def azure_openai_responses_provider() -> Any:
    """创建 Azure OpenAI Responses Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="azure-openai-responses",
            name="Azure OpenAI",
            base_url="(from config)",
            models=list(AZURE_OPENAI_RESPONSES_MODELS.values()),
            api=azure_openai_responses_api(),
        )
    )
