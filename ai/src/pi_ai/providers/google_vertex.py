"""Google Vertex AI Provider。"""

from __future__ import annotations

from typing import Any

from ..api.google_vertex_lazy import google_vertex_api
from ..models import CreateProviderOptions, create_provider
from .google_vertex_models import GOOGLE_VERTEX_MODELS


def google_vertex_provider() -> Any:
    """创建 Google Vertex AI Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="google-vertex",
            name="Google Vertex AI",
            base_url="https://*region*-aiplatform.googleapis.com/v1",
            models=list(GOOGLE_VERTEX_MODELS.values()),
            api=google_vertex_api(),
        )
    )
