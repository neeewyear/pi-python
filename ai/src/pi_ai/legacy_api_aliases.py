"""旧 API 别名映射。

提供已弃用的流式函数别名，用于向后兼容。这些函数在 API 格式化器模块
（15.4 阶段）创建后激活。

当前为占位骨架，返回 ``NotImplementedError``。
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# 占位：将在 API 格式化器模块（15.4 阶段）创建后激活
# ---------------------------------------------------------------------------
# 对应 TS 侧：
#   anthropicMessagesApi().stream  → stream_anthropic
#   openAIResponsesApi().stream    → stream_openai_responses
#   ...


def _not_implemented(name: str) -> None:
    """占位函数，提示 API 格式化器尚未迁移。"""
    raise NotImplementedError(
        f"{name} API 格式化器尚未迁移。"
        "请等待 15.4 阶段完成后再使用此函数。"
    )


def stream_anthropic(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``anthropic_messages_api().stream`` 替代。"""
    _not_implemented("stream_anthropic")
    return None  # unreachable


def stream_simple_anthropic(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``anthropic_messages_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_anthropic")
    return None


def stream_azure_openai_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``azure_openai_responses_api().stream`` 替代。"""
    _not_implemented("stream_azure_openai_responses")
    return None


def stream_simple_azure_openai_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``azure_openai_responses_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_azure_openai_responses")
    return None


def stream_google(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``google_generative_ai_api().stream`` 替代。"""
    _not_implemented("stream_google")
    return None


def stream_simple_google(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``google_generative_ai_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_google")
    return None


def stream_google_vertex(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``google_vertex_api().stream`` 替代。"""
    _not_implemented("stream_google_vertex")
    return None


def stream_simple_google_vertex(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``google_vertex_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_google_vertex")
    return None


def stream_mistral(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``mistral_conversations_api().stream`` 替代。"""
    _not_implemented("stream_mistral")
    return None


def stream_simple_mistral(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``mistral_conversations_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_mistral")
    return None


def stream_openai_codex_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_codex_responses_api().stream`` 替代。"""
    _not_implemented("stream_openai_codex_responses")
    return None


def stream_simple_openai_codex_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_codex_responses_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_openai_codex_responses")
    return None


def stream_openai_completions(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_completions_api().stream`` 替代。"""
    _not_implemented("stream_openai_completions")
    return None


def stream_simple_openai_completions(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_completions_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_openai_completions")
    return None


def stream_openai_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_responses_api().stream`` 替代。"""
    _not_implemented("stream_openai_responses")
    return None


def stream_simple_openai_responses(*args: object, **kwargs: object) -> object:
    """已弃用：使用 ``openai_responses_api().stream_simple`` 替代。"""
    _not_implemented("stream_simple_openai_responses")
    return None