"""不可变的 ``models.json`` 快照（对应 TS ``model-config.ts``）。

``ModelConfig`` 负责加载和验证 ``models.json`` 文件，返回一个不可变的
provider 配置映射。该映射不包含任何凭据信息（credential-blind）。
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pi_ai.types import Api


# ---------------------------------------------------------------------------
# 类型定义（对应 TS TypeBox Schema）
# ---------------------------------------------------------------------------


class PercentileCutoffs:
    """百分位截断配置。"""

    p50: Optional[float]
    p75: Optional[float]
    p90: Optional[float]
    p99: Optional[float]

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        self.p50 = data.get("p50") if data else None
        self.p75 = data.get("p75") if data else None
        self.p90 = data.get("p90") if data else None
        self.p99 = data.get("p99") if data else None


class OpenRouterRouting:
    """OpenRouter 路由配置。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.allow_fallbacks: Optional[bool] = d.get("allow_fallbacks")
        self.require_parameters: Optional[bool] = d.get("require_parameters")
        self.data_collection: Optional[str] = d.get("data_collection")
        self.zdr: Optional[bool] = d.get("zdr")
        self.enforce_distillable_text: Optional[bool] = d.get("enforce_distillable_text")
        self.order: Optional[List[str]] = d.get("order")
        self.only: Optional[List[str]] = d.get("only")
        self.ignore: Optional[List[str]] = d.get("ignore")
        self.quantizations: Optional[List[str]] = d.get("quantizations")
        self.sort: Any = d.get("sort")
        self.max_price: Optional[Dict[str, Any]] = d.get("max_price")
        self.preferred_min_throughput: Any = d.get("preferred_min_throughput")
        self.preferred_max_latency: Any = d.get("preferred_max_latency")


class VercelGatewayRouting:
    """Vercel AI Gateway 路由配置。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.only: Optional[List[str]] = d.get("only")
        self.order: Optional[List[str]] = d.get("order")


class ThinkingLevelMap:
    """思考级别映射。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.off: Optional[str] = d.get("off")
        self.minimal: Optional[str] = d.get("minimal")
        self.low: Optional[str] = d.get("low")
        self.medium: Optional[str] = d.get("medium")
        self.high: Optional[str] = d.get("high")
        self.xhigh: Optional[str] = d.get("xhigh")
        self.max: Optional[str] = d.get("max")


class ChatTemplateKwarg:
    """聊天模板参数。"""


class OpenAICompletionsCompat:
    """OpenAI Completions API 兼容配置。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.supports_store: Optional[bool] = d.get("supportsStore")
        self.supports_developer_role: Optional[bool] = d.get("supportsDeveloperRole")
        self.supports_reasoning_effort: Optional[bool] = d.get("supportsReasoningEffort")
        self.supports_usage_in_streaming: Optional[bool] = d.get("supportsUsageInStreaming")
        self.max_tokens_field: Optional[str] = d.get("maxTokensField")
        self.requires_tool_result_name: Optional[bool] = d.get("requiresToolResultName")
        self.requires_assistant_after_tool_result: Optional[bool] = d.get("requiresAssistantAfterToolResult")
        self.requires_thinking_as_text: Optional[bool] = d.get("requiresThinkingAsText")
        self.requires_reasoning_content_on_assistant_messages: Optional[bool] = d.get(
            "requiresReasoningContentOnAssistantMessages"
        )
        self.thinking_format: Optional[str] = d.get("thinkingFormat")
        self.chat_template_kwargs: Optional[Dict[str, Any]] = d.get("chatTemplateKwargs")
        self.chat_template_args: Optional[Dict[str, Any]] = d.get("chatTemplateArgs")
        self.cache_control_format: Optional[str] = d.get("cacheControlFormat")
        self.open_router_routing: Optional[Dict[str, Any]] = d.get("openRouterRouting")
        self.vercel_gateway_routing: Optional[Dict[str, Any]] = d.get("vercelGatewayRouting")
        self.supports_openai_grammar_tools: Optional[bool] = d.get("supportsOpenAIGrammarTools")
        self.supports_strict_mode: Optional[bool] = d.get("supportsStrictMode")
        self.send_session_affinity_headers: Optional[bool] = d.get("sendSessionAffinityHeaders")
        self.deferred_tools_mode: Optional[str] = d.get("deferredToolsMode")
        self.session_affinity_format: Optional[str] = d.get("sessionAffinityFormat")
        self.supports_long_cache_retention: Optional[bool] = d.get("supportsLongCacheRetention")


class OpenAIResponsesCompat:
    """OpenAI Responses API 兼容配置。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.supports_developer_role: Optional[bool] = d.get("supportsDeveloperRole")
        self.session_affinity_format: Optional[str] = d.get("sessionAffinityFormat")
        self.supports_long_cache_retention: Optional[bool] = d.get("supportsLongCacheRetention")
        self.supports_strict_mode: Optional[bool] = d.get("supportsStrictMode")
        self.supports_openai_grammar_tools: Optional[bool] = d.get("supportsOpenAIGrammarTools")
        self.supports_tool_search: Optional[bool] = d.get("supportsToolSearch")


class AnthropicMessagesCompat:
    """Anthropic Messages API 兼容配置。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.supports_eager_tool_input_streaming: Optional[bool] = d.get("supportsEagerToolInputStreaming")
        self.supports_long_cache_retention: Optional[bool] = d.get("supportsLongCacheRetention")
        self.send_session_affinity_headers: Optional[bool] = d.get("sendSessionAffinityHeaders")
        self.supports_cache_control_on_tools: Optional[bool] = d.get("supportsCacheControlOnTools")
        self.supports_temperature: Optional[bool] = d.get("supportsTemperature")
        self.force_adaptive_thinking: Optional[bool] = d.get("forceAdaptiveThinking")
        self.allow_empty_signature: Optional[bool] = d.get("allowEmptySignature")
        self.supports_strict_tools: Optional[bool] = d.get("supportsStrictTools")
        self.supports_tool_references: Optional[bool] = d.get("supportsToolReferences")


class ModelCostRates:
    """模型成本费率。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.input: float = d.get("input", 0)
        self.output: float = d.get("output", 0)
        self.cache_read: float = d.get("cacheRead", 0)
        self.cache_write: float = d.get("cacheWrite", 0)


class ModelCostTier:
    """模型成本层级。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.input_tokens_above: float = data["inputTokensAbove"]
        self.input: float = data.get("input", 0)
        self.output: float = data.get("output", 0)
        self.cache_read: float = data.get("cacheRead", 0)
        self.cache_write: float = data.get("cacheWrite", 0)


class ModelCost:
    """模型成本。"""

    def __init__(self, data: Optional[Dict[str, Any]] = None) -> None:
        d = data or {}
        self.input: float = d.get("input", 0)
        self.output: float = d.get("output", 0)
        self.cache_read: float = d.get("cacheRead", 0)
        self.cache_write: float = d.get("cacheWrite", 0)
        self.tiers: Optional[List[ModelCostTier]] = None
        if "tiers" in d:
            self.tiers = [ModelCostTier(t) for t in d["tiers"]]


# ---------------------------------------------------------------------------
# Provider 配置类型
# ---------------------------------------------------------------------------


class ModelsJsonModel:
    """models.json 中的模型定义。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.id: str = data["id"]
        self.name: Optional[str] = data.get("name")
        self.api: Optional[str] = data.get("api")
        self.base_url: Optional[str] = data.get("baseUrl")
        self.reasoning: Optional[bool] = data.get("reasoning")
        self.thinking_level_map: Optional[Dict[str, Optional[str]]] = data.get("thinkingLevelMap")
        self.input: Optional[List[str]] = data.get("input")
        self.cost: Optional[Dict[str, Any]] = data.get("cost")
        self.context_window: Optional[int] = data.get("contextWindow")
        self.max_tokens: Optional[int] = data.get("maxTokens")
        self.sampling_params: Optional[Dict[str, Any]] = data.get("samplingParams")
        self.headers: Optional[Dict[str, str]] = data.get("headers")
        self.compat: Optional[Dict[str, Any]] = data.get("compat")


class ModelsJsonModelOverride:
    """models.json 中的模型覆盖定义。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: Optional[str] = data.get("name")
        self.reasoning: Optional[bool] = data.get("reasoning")
        self.thinking_level_map: Optional[Dict[str, Optional[str]]] = data.get("thinkingLevelMap")
        self.input: Optional[List[str]] = data.get("input")
        self.cost: Optional[Dict[str, Any]] = data.get("cost")
        self.context_window: Optional[int] = data.get("contextWindow")
        self.max_tokens: Optional[int] = data.get("maxTokens")
        self.sampling_params: Optional[Dict[str, Any]] = data.get("samplingParams")
        self.headers: Optional[Dict[str, str]] = data.get("headers")
        self.compat: Optional[Dict[str, Any]] = data.get("compat")


class ModelsJsonProvider:
    """models.json 中的 provider 配置。"""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.name: Optional[str] = data.get("name")
        self.base_url: Optional[str] = data.get("baseUrl")
        self.api_key: Optional[str] = data.get("apiKey")
        self.api: Optional[str] = data.get("api")
        self.oauth: Optional[str] = data.get("oauth")
        self.headers: Optional[Dict[str, str]] = data.get("headers")
        self.compat: Optional[Dict[str, Any]] = data.get("compat")
        self.auth_header: Optional[bool] = data.get("authHeader")
        self.models: Optional[List[ModelsJsonModel]] = None
        if "models" in data:
            self.models = [ModelsJsonModel(m) for m in data["models"]]
        self.model_overrides: Optional[Dict[str, ModelsJsonModelOverride]] = None
        if "modelOverrides" in data:
            self.model_overrides = {
                k: ModelsJsonModelOverride(v) for k, v in data["modelOverrides"].items()
            }


# ---------------------------------------------------------------------------
# ModelConfig
# ---------------------------------------------------------------------------


def _deep_freeze(value: Any) -> Any:
    """递归冻结对象（浅冻结映射和序列）。"""
    if isinstance(value, dict):
        frozen: Dict[str, Any] = {}
        for k, v in value.items():
            frozen[k] = _deep_freeze(v)
        return frozen
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(v) for v in value)
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    return value


def _strip_json_comments(content: str) -> str:
    """移除 JSON 中的注释（简单实现，不支持字符串内的特殊情况）。"""
    import re as _re

    # 移除单行注释 //
    lines = content.split("\n")
    result: List[str] = []
    for line in lines:
        # 简单处理：忽略字符串内的 //
        stripped = _re.sub(r"//(?!$)", "#", line)
        # 只移除不在字符串中的 //
        idx = 0
        in_string = False
        string_char = None
        out = []
        while idx < len(stripped):
            ch = stripped[idx]
            if in_string:
                out.append(ch)
                if ch == "\\":
                    idx += 1
                    if idx < len(stripped):
                        out.append(stripped[idx])
                elif ch == string_char:
                    in_string = False
                    string_char = None
            else:
                if ch in ("\"", "'"):
                    in_string = True
                    string_char = ch
                    out.append(ch)
                elif ch == "/" and idx + 1 < len(stripped) and stripped[idx + 1] == "/":
                    break  # 忽略行尾注释
                else:
                    out.append(ch)
            idx += 1
        result.append("".join(out))
    return "\n".join(result)


def _format_validation_path(path: str, keyword: str, params: Any) -> str:
    """格式化验证错误路径。"""
    if keyword == "required":
        required_properties = params.get("requiredProperties") if isinstance(params, dict) else None
        if required_properties:
            required_property = required_properties[0]
            base_path = path.strip("/").replace("/", ".")
            return f"{base_path}.{required_property}" if base_path else required_property
    formatted = path.strip("/").replace("/", ".")
    return formatted or "root"


class ModelConfig:
    """一次性的 ``models.json`` 不可变加载。

    使用方式：
        config = await ModelConfig.load("/path/to/models.json")
        provider = config.get_provider("anthropic")
    """

    def __init__(
        self,
        providers: Dict[str, Any],
        error: Optional[str] = None,
    ) -> None:
        self._providers: Dict[str, Any] = _deep_freeze(providers)
        self._error = error

    @classmethod
    async def load(cls, models_json_path: Optional[str]) -> "ModelConfig":
        """加载并验证 models.json 文件。"""
        from pathlib import Path

        if not models_json_path:
            return cls({})

        path = Path(models_json_path).expanduser().resolve()
        if not path.exists():
            return cls({})

        try:
            import aiofiles

            async with aiofiles.open(str(path), "r", encoding="utf-8") as f:
                content = await f.read()
        except FileNotFoundError:
            return cls({})
        except Exception as error:
            return cls(
                {},
                f"Failed to load models.json: {error}\n\nFile: {path}",
            )

        try:
            import json as _json

            parsed = _json.loads(_strip_json_comments(content))
        except Exception as error:
            return cls(
                {},
                f"Failed to parse models.json: {error}\n\nFile: {path}",
            )

        # 验证结构
        if not isinstance(parsed, dict):
            return cls(
                {},
                f"Invalid models.json: expected object, got {type(parsed).__name__}\n\nFile: {path}",
            )

        if "providers" not in parsed or not isinstance(parsed["providers"], dict):
            return cls(
                {},
                f"Invalid models.json: missing 'providers' object\n\nFile: {path}",
            )

        providers_raw: Dict[str, Any] = parsed["providers"]

        # 基本验证：至少需要 id 字段
        errors: List[str] = []
        for provider_id, provider_data in providers_raw.items():
            if not isinstance(provider_data, dict):
                errors.append(f"  - providers.{provider_id}: expected object")
                continue
            models = provider_data.get("models")
            if isinstance(models, list):
                for i, model in enumerate(models):
                    if not isinstance(model, dict) or "id" not in model:
                        errors.append(
                            f"  - providers.{provider_id}.models[{i}]: missing 'id'"
                        )

        if errors:
            return cls(
                {},
                f"Invalid models.json schema:\n" + "\n".join(errors) + f"\n\nFile: {path}",
            )

        # 深拷贝并冻结
        providers = copy.deepcopy(providers_raw)
        return cls(providers)

    def get_provider(self, provider_id: str) -> Optional[Dict[str, Any]]:
        """获取指定 provider 的配置。"""
        return self._providers.get(provider_id)

    def get_provider_ids(self) -> List[str]:
        """获取所有 provider ID。"""
        return list(self._providers.keys())

    def get_error(self) -> Optional[str]:
        """获取加载错误信息。"""
        return self._error