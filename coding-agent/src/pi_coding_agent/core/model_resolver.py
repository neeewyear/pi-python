"""模型解析、限定和初始选择（对应 TS ``model-resolver.ts``）。

提供模型模式解析、精确/模糊匹配、模型范围限定等功能。
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any, cast

from pi_ai.models import models_are_equal
from pi_ai.types import Model, ThinkingLevel

from .model_runtime import ModelRuntime

# ---------------------------------------------------------------------------
# 默认模型 ID
# ---------------------------------------------------------------------------

DEFAULT_MODEL_PER_PROVIDER: dict[str, str] = {
    "amazon-bedrock": "us.anthropic.claude-opus-4-6-v1",
    "ant-ling": "Ring-2.6-1T",
    "anthropic": "claude-opus-4-8",
    "openai": "gpt-5.5",
    "azure-openai-responses": "gpt-5.4",
    "openai-codex": "gpt-5.5",
    "radius": "auto",
    "nvidia": "nvidia/nemotron-3-super-120b-a12b",
    "deepseek": "deepseek-v4-pro",
    "google": "gemini-3.1-pro-preview",
    "google-vertex": "gemini-3.1-pro-preview",
    "github-copilot": "gpt-5.4",
    "openrouter": "moonshotai/kimi-k2.6",
    "vercel-ai-gateway": "zai/glm-5.1",
    "xai": "grok-4.5",
    "groq": "openai/gpt-oss-120b",
    "cerebras": "zai-glm-4.7",
    "zai": "glm-5.1",
    "zai-coding-cn": "glm-5.1",
    "mistral": "devstral-medium-latest",
    "minimax": "MiniMax-M2.7",
    "minimax-cn": "MiniMax-M2.7",
    "moonshotai": "kimi-k2.6",
    "moonshotai-cn": "kimi-k2.6",
    "huggingface": "moonshotai/Kimi-K2.6",
    "fireworks": "accounts/fireworks/models/kimi-k2p6",
    "together": "moonshotai/Kimi-K2.6",
    "baseten": "zai-org/GLM-5.2",
    "opencode": "kimi-k2.6",
    "opencode-go": "kimi-k2.6",
    "kimi-coding": "kimi-for-coding",
    "cloudflare-workers-ai": "@cf/moonshotai/kimi-k2.6",
    "cloudflare-ai-gateway": "workers-ai/@cf/moonshotai/kimi-k2.6",
    "qwen-token-plan": "qwen3.7-max",
    "qwen-token-plan-cn": "qwen3.7-max",
    "xiaomi": "mimo-v2.5-pro",
    "xiaomi-token-plan-cn": "mimo-v2.5-pro",
    "xiaomi-token-plan-ams": "mimo-v2.5-pro",
    "xiaomi-token-plan-sgp": "mimo-v2.5-pro",
}

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

VALID_THINKING_LEVELS: set[str] = {
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
}


def is_valid_thinking_level(level: str) -> bool:
    """检查字符串是否为有效的思考级别。"""
    return level in VALID_THINKING_LEVELS


class ScopedModel:
    """限定范围内的模型。"""

    def __init__(
        self,
        model: Model,
        thinking_level: ThinkingLevel | None = None,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level


class ParsedModelResult:
    """解析后的模型结果。"""

    def __init__(
        self,
        model: Model | None = None,
        thinking_level: ThinkingLevel | None = None,
        warning: str | None = None,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.warning = warning


class ModelScopeDiagnostic:
    """模型范围诊断信息。"""

    def __init__(
        self,
        type_: str,
        code: str,
        message: str,
        pattern: str,
    ) -> None:
        self.type = type_
        self.code = code
        self.message = message
        self.pattern = pattern


class ResolveModelScopeResult:
    """模型范围解析结果。"""

    def __init__(
        self,
        scoped_models: list[ScopedModel] | None = None,
        diagnostics: list[ModelScopeDiagnostic] | None = None,
    ) -> None:
        self.scoped_models: list[ScopedModel] = scoped_models or []
        self.diagnostics: list[ModelScopeDiagnostic] = diagnostics or []


class ResolveCliModelResult:
    """CLI 模型解析结果。"""

    def __init__(
        self,
        model: Model | None = None,
        thinking_level: ThinkingLevel | None = None,
        warning: str | None = None,
        error: str | None = None,
    ) -> None:
        self.model = model
        self.thinking_level = thinking_level
        self.warning = warning
        self.error = error


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _is_alias(model_id: str) -> bool:
    """检查模型 ID 是否为别名（无日期后缀）。"""
    if model_id.endswith("-latest"):
        return True
    date_pattern = re.compile(r"-\d{8}$")
    return not date_pattern.search(model_id)


def find_exact_model_reference_match(
    model_reference: str,
    available_models: list[Model],
) -> Model | None:
    """查找精确的模型引用匹配。

    支持：
    - 裸模型 ID
    - 规范格式 ``provider/modelId``
    - ``provider/modelId`` 格式
    """
    trimmed = model_reference.strip()
    if not trimmed:
        return None

    normalized = trimmed.lower()

    # 规范格式匹配：provider/modelId
    canonical = [
        m
        for m in available_models
        if f"{m.provider}/{m.model_id}".lower() == normalized
    ]
    if len(canonical) == 1:
        return canonical[0]
    if len(canonical) > 1:
        return None

    # 带斜杠的格式
    slash_idx = trimmed.find("/")
    if slash_idx != -1:
        provider = trimmed[:slash_idx].strip()
        model_id = trimmed[slash_idx + 1 :].strip()
        if provider and model_id:
            provider_matches = [
                m
                for m in available_models
                if m.provider.lower() == provider.lower()
                and m.model_id.lower() == model_id.lower()
            ]
            if len(provider_matches) == 1:
                return provider_matches[0]
            if len(provider_matches) > 1:
                return None

    # 只匹配模型 ID
    id_matches = [m for m in available_models if m.model_id.lower() == normalized]
    return id_matches[0] if len(id_matches) == 1 else None


def _try_match_model(
    model_pattern: str,
    available_models: list[Model],
) -> Model | None:
    """尝试匹配模型模式。"""
    exact = find_exact_model_reference_match(model_pattern, available_models)
    if exact:
        return exact

    # 模糊匹配
    pattern_lower = model_pattern.lower()
    matches = [
        m
        for m in available_models
        if pattern_lower in m.model_id.lower()
        or (cast("Any", m).name and pattern_lower in cast("Any", m).name.lower())
    ]

    if not matches:
        return None

    # 别名优先
    aliases = [m for m in matches if _is_alias(m.model_id)]
    dated = [m for m in matches if not _is_alias(m.model_id)]

    if aliases:
        aliases.sort(key=lambda m: m.model_id, reverse=True)
        return aliases[0]
    else:
        dated.sort(key=lambda m: m.model_id, reverse=True)
        return dated[0]


def _build_fallback_model(
    provider: str,
    model_id: str,
    available_models: list[Model],
) -> Model | None:
    """构建回退模型。"""
    provider_models = [m for m in available_models if m.provider == provider]
    if not provider_models:
        return None

    default_id = DEFAULT_MODEL_PER_PROVIDER.get(provider)
    base_model = None
    if default_id:
        base_model = next(
            (m for m in provider_models if m.model_id == default_id),
            provider_models[0],
        )
    else:
        base_model = provider_models[0]

    # 创建副本并覆盖 ID 和 name
    model_dict = cast("dict[str, Any]", base_model)
    model_dict["id"] = model_id
    model_dict["model_id"] = model_id
    model_dict["name"] = model_id
    return cast("Model", model_dict)


# ---------------------------------------------------------------------------
# 公共 API
# ---------------------------------------------------------------------------


def parse_model_pattern(
    pattern: str,
    available_models: list[Model],
    options: dict[str, bool] | None = None,
) -> ParsedModelResult:
    """解析模型模式，提取模型和思考级别。

    格式：``pattern:level``，其中 ``:level`` 可选。

    算法：
    1. 尝试完整匹配模式
    2. 如果找到则返回
    3. 如果未找到且有冒号，在最后一个冒号处分割：
       - 如果后缀是有效思考级别，递归前缀并应用该级别
       - 如果后缀无效，警告并递归前缀
    """
    options = options or {}
    allow_invalid_fallback = options.get("allow_invalid_thinking_level_fallback", True)

    # 精确匹配
    exact = _try_match_model(pattern, available_models)
    if exact:
        return ParsedModelResult(model=exact)

    # 尝试在最后一个冒号处分割
    last_colon = pattern.rfind(":")
    if last_colon == -1:
        return ParsedModelResult()

    prefix = pattern[:last_colon]
    suffix = pattern[last_colon + 1 :]

    if is_valid_thinking_level(suffix):
        result = parse_model_pattern(prefix, available_models, options)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                thinking_level=cast(
                    "ThinkingLevel | None", suffix if not result.warning else None
                ),
                warning=result.warning,
            )
        return result
    else:
        if not allow_invalid_fallback:
            return ParsedModelResult()

        result = parse_model_pattern(prefix, available_models, options)
        if result.model:
            return ParsedModelResult(
                model=result.model,
                warning=(
                    f'Invalid thinking level "{suffix}" in pattern '
                    f'"{pattern}". Using default instead.'
                ),
            )
        return result


def resolve_model_scope_from_models(
    patterns: list[str],
    models: list[Model],
) -> ResolveModelScopeResult:
    """从模型列表中解析模型范围。

    对每个模式：
    1. 检查是否包含 glob 字符（``*``, ``?``, ``[``）
    2. 如果是 glob，匹配 ``provider/modelId`` 或 ``modelId``
    3. 如果不是 glob，使用 ``parse_model_pattern``
    """
    available = list(models)
    scoped: list[ScopedModel] = []
    diagnostics: list[ModelScopeDiagnostic] = []

    for pattern in patterns:
        # glob 模式
        if any(c in pattern for c in ("*", "?", "[")):
            colon_idx = pattern.rfind(":")
            glob_pattern = pattern
            thinking_level: ThinkingLevel | None = None

            if colon_idx != -1:
                suffix = pattern[colon_idx + 1 :]
                if is_valid_thinking_level(suffix):
                    thinking_level = suffix  # type: ignore[assignment]
                    glob_pattern = pattern[:colon_idx]

            exact = find_exact_model_reference_match(glob_pattern, available)
            if exact:
                if not any(models_are_equal(sm.model, exact) for sm in scoped):
                    scoped.append(ScopedModel(exact, thinking_level))
                continue

            # 匹配 provider/modelId 或模型 ID
            matching = []
            for m in available:
                full_id = f"{m.provider}/{m.model_id}"
                if fnmatch.fnmatch(
                    full_id.lower(), glob_pattern.lower()
                ) or fnmatch.fnmatch(m.model_id.lower(), glob_pattern.lower()):
                    matching.append(m)

            if not matching:
                diagnostics.append(
                    ModelScopeDiagnostic(
                        type_="warning",
                        code="no-match",
                        message=f'No models match pattern "{pattern}"',
                        pattern=pattern,
                    )
                )
                continue

            for model in matching:
                if not any(models_are_equal(sm.model, model) for sm in scoped):
                    scoped.append(ScopedModel(model, thinking_level))
            continue

        # 非 glob 模式
        result = parse_model_pattern(pattern, available)

        if result.warning:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type_="warning",
                    code="invalid-thinking-level",
                    message=result.warning,
                    pattern=pattern,
                )
            )

        if not result.model:
            diagnostics.append(
                ModelScopeDiagnostic(
                    type_="warning",
                    code="no-match",
                    message=f'No models match pattern "{pattern}"',
                    pattern=pattern,
                )
            )
            continue

        if not any(models_are_equal(sm.model, result.model) for sm in scoped):
            scoped.append(
                ScopedModel(
                    result.model,
                    result.thinking_level,
                )
            )

    return ResolveModelScopeResult(
        scoped_models=scoped,
        diagnostics=diagnostics,
    )


async def resolve_model_scope_with_diagnostics(
    patterns: list[str],
    model_runtime: ModelRuntime,
    options: dict[str, Any] | None = None,
) -> ResolveModelScopeResult:
    """通过 ModelRuntime 解析模型范围。"""
    from pi_ai.auth.types import AuthOperationOptions

    available = await model_runtime.get_available(
        None, cast("AuthOperationOptions | None", options)
    )
    return resolve_model_scope_from_models(patterns, list(available))


async def resolve_model_scope(
    patterns: list[str],
    model_runtime: ModelRuntime,
    options: dict[str, Any] | None = None,
) -> list[ScopedModel]:
    """通过 ModelRuntime 解析模型范围，打印诊断信息。"""
    result = await resolve_model_scope_with_diagnostics(
        patterns, model_runtime, options
    )
    for diagnostic in result.diagnostics:
        import warnings

        warnings.warn(f"Warning: {diagnostic.message}")
    return result.scoped_models


def resolve_cli_model(
    cli_provider: str | None = None,
    cli_model: str | None = None,
    cli_thinking: ThinkingLevel | None = None,
    model_runtime: ModelRuntime | None = None,
) -> ResolveCliModelResult:
    """解析 CLI 标志中的单个模型。

    支持：
    - ``--provider <provider> --model <pattern>``
    - ``--model <provider>/<pattern>``
    """
    if not cli_model:
        return ResolveCliModelResult()

    if not model_runtime:
        return ResolveCliModelResult()

    # 使用所有模型（包括未配置认证的）
    available = list(model_runtime.get_models())

    if cli_provider:
        # 在指定 provider 中查找
        provider_models = [m for m in available if m.provider == cli_provider]
        if not provider_models:
            return ResolveCliModelResult(error=f'Unknown provider "{cli_provider}"')

        # 尝试 exact + fuzzy 匹配
        match = _try_match_model(cli_model, provider_models)
        if match:
            return ResolveCliModelResult(
                model=match,
                thinking_level=cli_thinking,
            )

        # 回退：创建虚拟模型
        fallback = _build_fallback_model(cli_provider, cli_model, provider_models)
        if fallback:
            return ResolveCliModelResult(
                model=fallback,
                thinking_level=cli_thinking,
                warning=f'Model "{cli_model}" not found for provider "{cli_provider}". '
                f"Using fallback configuration.",
            )

        return ResolveCliModelResult(
            error=f'No models found for provider "{cli_provider}"'
        )

    # 无 provider 指定，在所有模型中查找
    # 先尝试精确匹配 "provider/model"
    exact = find_exact_model_reference_match(cli_model, available)
    if exact:
        return ResolveCliModelResult(
            model=exact,
            thinking_level=cli_thinking,
        )

    # 尝试模糊匹配
    result = parse_model_pattern(cli_model, available)
    if result.model:
        return ResolveCliModelResult(
            model=result.model,
            thinking_level=result.thinking_level or cli_thinking,
            warning=result.warning,
        )

    # 尝试从默认 provider 查找
    for provider_id, default_id in DEFAULT_MODEL_PER_PROVIDER.items():
        provider_models = [m for m in available if m.provider == provider_id]
        if provider_models:
            match = _try_match_model(cli_model, provider_models)
            if match:
                return ResolveCliModelResult(
                    model=match,
                    thinking_level=cli_thinking,
                    warning=f'Matched "{cli_model}" to {provider_id}.',
                )

    return ResolveCliModelResult(
        error=f'Could not resolve model "{cli_model}". '
        f"Use --provider to specify a provider.",
    )


async def find_initial_model(
    scoped_models: list[ScopedModel] | None = None,
    is_continuing: bool = False,
    default_provider: str | None = None,
    default_model_id: str | None = None,
    default_thinking_level: ThinkingLevel | None = None,
    model_runtime: ModelRuntime | None = None,
) -> ResolveCliModelResult:
    """查找初始模型（当 CLI 未指定模型时使用）。

    优先级：
    1. ``scoped_models`` 中的第一个可用模型
    2. 默认 provider + 默认 model_id
    3. 默认 provider 下的第一个模型
    4. 任意 provider 下的第一个模型

    Args:
        scoped_models: 预选的模型范围列表。
        is_continuing: 是否正在继续会话。
        default_provider: 默认 provider 名称。
        default_model_id: 默认模型 ID。
        default_thinking_level: 默认思考级别。
        model_runtime: 模型运行时（用于获取可用模型）。

    Returns:
        ``ResolveCliModelResult``，包含找到的模型或错误信息。
    """
    if scoped_models:
        for sm in scoped_models:
            if sm.model:
                return ResolveCliModelResult(
                    model=sm.model,
                    thinking_level=default_thinking_level,
                )

    if not model_runtime:
        return ResolveCliModelResult(error="No model runtime available")

    available = list(model_runtime.get_models())
    if not available:
        return ResolveCliModelResult(error="No models available")

    # 有默认 provider 和 model_id
    if default_provider and default_model_id:
        for m in available:
            if m.provider == default_provider and m.model_id == default_model_id:
                return ResolveCliModelResult(
                    model=m,
                    thinking_level=default_thinking_level,
                )
        # provider 匹配但 model_id 未精确匹配 -> 模糊匹配
        provider_models = [m for m in available if m.provider == default_provider]
        if provider_models:
            match = _try_match_model(default_model_id, provider_models)
            if match:
                return ResolveCliModelResult(
                    model=match,
                    thinking_level=default_thinking_level,
                    warning=f'Matched "{default_model_id}" to {match.model_id}.',
                )

    # 只有默认 provider
    if default_provider:
        provider_models = [m for m in available if m.provider == default_provider]
        if provider_models:
            return ResolveCliModelResult(
                model=provider_models[0],
                thinking_level=default_thinking_level,
            )

    # 任意 provider 的第一个模型
    return ResolveCliModelResult(
        model=available[0],
        thinking_level=default_thinking_level,
        warning=f'No default provider configured, using "{available[0].provider}/{available[0].model_id}".',
    )
