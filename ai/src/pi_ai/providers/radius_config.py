"""Radius 配置工具（对应 ``radius-config.ts``）。"""

from __future__ import annotations

from typing import Any, TypedDict

from ..auth.types import OAuthCredential
from ..types import Model, ThinkingLevelMap

DEFAULT_RADIUS_GATEWAY = "https://radius.pi.dev"


class RadiusGatewayModel(TypedDict, total=False):
    """Radius 网关模型定义。"""

    id: str
    name: str
    reasoning: bool
    thinking_level_map: ThinkingLevelMap | None
    input: list[str]
    cost: dict[str, float]
    context_window: int
    max_tokens: int


class RadiusGatewayConfig(TypedDict, total=False):
    """Radius 网关配置。"""

    base_url: str
    models: list[RadiusGatewayModel]


class RadiusOAuthCredential(TypedDict, total=False):
    """Radius OAuth 凭据（含网关配置）。"""

    type: str
    refresh: str
    access: str
    expires: int
    account_id: str
    enterprise_url: str
    available_model_ids: list[str]
    scope: str
    gateway_config: RadiusGatewayConfig | None


def _is_radius_gateway_model(value: Any) -> bool:
    """检查值是否为有效的 RadiusGatewayModel。"""
    if not isinstance(value, dict):
        return False
    model = value
    return (
        isinstance(model.get("id"), str)
        and isinstance(model.get("name"), str)
        and isinstance(model.get("reasoning"), bool)
        and isinstance(model.get("input"), list)
        and isinstance(model.get("cost"), dict)
        and isinstance(model.get("context_window"), (int, float))
        and isinstance(model.get("max_tokens"), (int, float))
    )


def _sanitize_radius_gateway_config(config: Any) -> RadiusGatewayConfig | None:
    """清理并验证 Radius 网关配置。"""
    if not isinstance(config, dict):
        return None
    base_url = config.get("base_url")
    models = config.get("models")
    if not isinstance(base_url, str) or not isinstance(models, list):
        return None
    return {
        "base_url": base_url,
        "models": [m for m in models if _is_radius_gateway_model(m)],
    }


def normalize_radius_gateway_url(value: str) -> str:
    """标准化 Radius 网关 URL。"""
    if value.startswith("http://") or value.startswith("https://"):
        with_scheme = value
    else:
        with_scheme = f"https://{value}"
    return with_scheme.rstrip("/")


def get_radius_credential_config(
    credential: OAuthCredential | None,
) -> RadiusGatewayConfig | None:
    """从凭据中获取 Radius 网关配置。"""
    if credential is None:
        return None
    radius_cred: Any = credential
    if isinstance(radius_cred, dict):
        gateway_config = radius_cred.get("gateway_config")
        return _sanitize_radius_gateway_config(gateway_config)
    return None


def get_radius_models_from_config(
    provider_id: str,
    config: RadiusGatewayConfig,
) -> list[Model]:
    """从配置中提取模型列表。"""
    models: list[Model] = []
    for m in config.get("models", []):
        model: dict[str, Any] = {
            "model_id": m["id"],
            "id": m["id"],
            "name": m.get("name", m["id"]),
            "api": "pi-messages",
            "provider": provider_id,
            "base_url": config["base_url"],
            "reasoning": m.get("reasoning", False),
            "input": m.get("input", ["text"]),
            "input_types": m.get("input", ["text"]),
            "context_window": m.get("context_window", 0),
            "max_tokens": m.get("max_tokens", 0),
        }
        cost = m.get("cost")
        if cost:
            model["cost"] = cost
        thinking_level_map = m.get("thinking_level_map")
        if thinking_level_map:
            model["thinking_level_map"] = thinking_level_map
        models.append(model)  # type: ignore[arg-type]
    return models


def get_radius_models(
    provider_id: str,
    credential: OAuthCredential | None,
) -> list[Model]:
    """从凭据中获取 Radius 模型列表。"""
    config = get_radius_credential_config(credential)
    if config:
        return get_radius_models_from_config(provider_id, config)
    return []


def _truncate_http_body(body: str) -> str:
    """截断 HTTP 响应体（用于错误消息）。"""
    trimmed = body.strip()
    if len(trimmed) > 512:
        return f"{trimmed[:512]}…"
    return trimmed


async def load_radius_gateway_config(
    gateway: str,
    api_key: str | None = None,
    signal: Any = None,
) -> RadiusGatewayConfig:
    """异步加载 Radius 网关配置。"""
    import json

    headers: dict[str, str] = {"accept": "application/json"}
    if api_key:
        headers["authorization"] = f"Bearer {api_key}"
    # 使用 Python 3.11+ 标准库
    import urllib.error
    import urllib.request

    url = f"{gateway}/v1/config"
    req = urllib.request.Request(url, headers=headers)
    # 异步执行 HTTP 请求
    import asyncio

    loop = asyncio.get_event_loop()

    def _fetch() -> tuple[int, str]:
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, resp.read().decode("utf-8")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8")

    status, body = await loop.run_in_executor(None, _fetch)
    if status != 200:
        raise RuntimeError(
            f"Could not load Radius config from {gateway}: "
            f"{status}: {_truncate_http_body(body)}"
        )
    config = _sanitize_radius_gateway_config(json.loads(body))
    if not config:
        raise RuntimeError(f"Invalid Radius config from {gateway}")
    return config
