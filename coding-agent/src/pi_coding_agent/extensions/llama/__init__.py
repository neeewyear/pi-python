"""Llama 扩展模块（对应 TS ``extensions/llama/index.ts``）。

导出 llama.cpp 集成所需的所有公共符号。
"""

from __future__ import annotations

from .client import (
    LlamaClient,
    LlamaModelArchitecture,
    LlamaModelEvent,
    LlamaModelInfo,
    LlamaModelMeta,
    LlamaModelStatus,
    LlamaModelStatusInfo,
    LlamaModelsResponse,
    LlamaProgress,
    llama_inference_url,
    normalize_llama_server_url,
)
from .huggingface import (
    DEFAULT_HUGGING_FACE_URL,
    HuggingFaceClient,
    HuggingFaceModel,
    HuggingFaceModelDetails,
    HuggingFaceQuantization,
    find_hugging_face_token,
)
from .provider import (
    DEFAULT_LLAMA_SERVER_URL,
    LLAMA_PROVIDER_ID,
    LlamaProviderController,
    create_llama_provider,
)
from .ui import (
    LlamaManagerAction,
    LlamaUi,
    ProgressState,
    model_description,
    run_with_progress,
    show_llama_ui,
)

__all__ = [
    # client
    "LlamaClient",
    "LlamaModelArchitecture",
    "LlamaModelEvent",
    "LlamaModelInfo",
    "LlamaModelMeta",
    "LlamaModelStatus",
    "LlamaModelStatusInfo",
    "LlamaModelsResponse",
    "LlamaProgress",
    "llama_inference_url",
    "normalize_llama_server_url",
    # huggingface
    "DEFAULT_HUGGING_FACE_URL",
    "HuggingFaceClient",
    "HuggingFaceModel",
    "HuggingFaceModelDetails",
    "HuggingFaceQuantization",
    "find_hugging_face_token",
    # provider
    "DEFAULT_LLAMA_SERVER_URL",
    "LLAMA_PROVIDER_ID",
    "LlamaProviderController",
    "create_llama_provider",
    # ui
    "LlamaManagerAction",
    "LlamaUi",
    "ProgressState",
    "model_description",
    "run_with_progress",
    "show_llama_ui",
]