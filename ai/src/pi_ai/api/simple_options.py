"""SimpleStreamOptions 转换逻辑（对应 ``simple-options.ts``）。"""

from __future__ import annotations

from typing import cast

from ..types import (
    Context,
    Model,
    SimpleStreamOptions,
    StreamOptions,
    ThinkingBudgets,
    ThinkingLevel,
)
from ..utils.estimate import estimate_context_tokens

CONTEXT_SAFETY_TOKENS = 4096
MIN_MAX_TOKENS = 1


def _get_model_context_window(model: Model) -> int:
    """安全获取 model.context_window。"""
    return cast(int, getattr(model, "context_window", 0))


def _get_model_sampling_params(model: Model) -> dict[str, object] | None:
    """安全获取 model.sampling_params。"""
    return cast("dict[str, object] | None", getattr(model, "sampling_params", None))


def _get_model_max_tokens(model: Model) -> int:
    """安全获取 model.max_tokens。"""
    return cast(int, getattr(model, "max_tokens", 0))


def clamp_max_tokens_to_context(
    model: Model,
    context: Context,
    max_tokens: int,
) -> int:
    """将 max_tokens 限制在上下文窗口内。"""
    context_window = _get_model_context_window(model)
    if context_window <= 0:
        return max(MIN_MAX_TOKENS, max_tokens)
    available = (
        context_window - estimate_context_tokens(context).tokens - CONTEXT_SAFETY_TOKENS
    )
    return min(max_tokens, max(MIN_MAX_TOKENS, available))


def build_base_options(
    model: Model,
    context: Context,
    options: SimpleStreamOptions | None = None,
    api_key: str | None = None,
) -> StreamOptions:
    """构建基础流式选项。"""
    model_sampling = _get_model_sampling_params(model)
    options_sampling = getattr(options, "sampling_params", None) if options else None
    sampling_params = None
    if model_sampling or options_sampling:
        merged = dict(model_sampling or {})
        if options_sampling:
            merged.update(options_sampling)
        sampling_params = merged

    options_max_tokens = getattr(options, "max_tokens", None) if options else None
    max_tokens = clamp_max_tokens_to_context(
        model, context, options_max_tokens or _get_model_max_tokens(model)
    )

    return StreamOptions(
        temperature=getattr(options, "temperature", None) if options else None,
        sampling_params=sampling_params,
        max_tokens=max_tokens,
        headers=getattr(options, "headers", None) if options else None,
        cache_retention=getattr(options, "cache_retention", None) if options else None,
        session_id=getattr(options, "session_id", None) if options else None,
        timeout_ms=getattr(options, "timeout_ms", None) if options else None,
        max_retries=getattr(options, "max_retries", None) if options else None,
        max_retry_delay_ms=getattr(options, "max_retry_delay_ms", None)
        if options
        else None,
        metadata=getattr(options, "metadata", None) if options else None,
        env=getattr(options, "env", None) if options else None,
        transport=getattr(options, "transport", None) if options else None,
        api_key=api_key or (getattr(options, "api_key", None) if options else None),
    )


def clamp_reasoning(
    effort: ThinkingLevel | None,
) -> ThinkingLevel | None:
    """限制思考级别。"""
    if effort in ("xhigh", "max"):
        return "high"
    return effort


def adjust_max_tokens_for_thinking(
    base_max_tokens: int | None,
    model_max_tokens: int,
    reasoning_level: ThinkingLevel,
    custom_budgets: ThinkingBudgets | None = None,
) -> tuple[int, int]:
    """调整 max_tokens 以适应思考预算。"""
    default_budgets: dict[str, int] = {
        "minimal": 1024,
        "low": 2048,
        "medium": 8192,
        "high": 16384,
    }
    budgets = dict(default_budgets)
    if custom_budgets:
        for key in ("minimal", "low", "medium", "high"):
            value = getattr(custom_budgets, key, None)
            if value is not None:
                budgets[key] = value

    min_output_tokens = 1024
    level = clamp_reasoning(reasoning_level) or "medium"
    thinking_budget = budgets.get(level, 8192)
    max_tokens = (
        model_max_tokens
        if base_max_tokens is None
        else min(base_max_tokens + thinking_budget, model_max_tokens)
    )

    if max_tokens <= thinking_budget:
        thinking_budget = max(0, max_tokens - min_output_tokens)

    return max_tokens, thinking_budget
