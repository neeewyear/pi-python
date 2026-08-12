"""远程模型目录提供者（对应 TS ``remote-catalog-provider.ts``）。

为一个静态内置 provider 添加 pi.dev 远程模型目录覆盖层，支持：
- 缓存 ETag 验证
- 本地生成时间戳比较
- 过期自动刷新（4 小时）
"""

from __future__ import annotations

import time
from typing import Any, cast

from pi_ai.models import ModelRecord, Provider, RefreshModelsContext

from pi_coding_agent.config import VERSION

DEFAULT_CATALOG_BASE_URL = "https://pi.dev"
REMOTE_CATALOG_REFRESH_INTERVAL_MS = 4 * 60 * 60 * 1000


def _merge_models(
    baseline: list[dict[str, Any]],
    dynamic: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """合并基线模型和动态模型。动态模型按 ID 覆盖或追加。"""
    merged = list(baseline)
    for model in dynamic:
        index = next(
            (i for i, m in enumerate(merged) if m["id"] == model["id"]),
            -1,
        )
        if index >= 0:
            merged[index] = model
        else:
            merged.append(model)
    return merged


def _dict_to_model(d: dict[str, Any]) -> Any:
    """将模型字典转换为 ``ModelRecord``（属性为 snake_case，供 API 层访问）。"""
    try:
        return ModelRecord(**d)
    except Exception:
        # 字典字段无法通过 ModelRecord 校验时，回退为 SimpleNamespace
        from types import SimpleNamespace

        obj = SimpleNamespace(**d)
        if "id" in d and not hasattr(obj, "model_id"):
            obj.model_id = d["id"]
        return obj


def _parse_catalog(provider_id: str, value: Any) -> list[dict[str, Any]]:
    """解析远程目录返回的模型列表。"""
    entries: Any = None
    if isinstance(value, list):
        entries = value
    elif isinstance(value, dict):
        if "models" in value and isinstance(value["models"], list):
            entries = value["models"]
        else:
            entries = list(value.values())
    if not entries:
        raise ValueError(f'Invalid model catalog for provider "{provider_id}"')
    result: list[dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and "id" in entry:
            model = dict(entry)
            model["provider"] = provider_id
            result.append(model)
    return result


def _remote_models(
    entry: dict[str, Any] | None,
    local_generated_at: int | None,
) -> list[dict[str, Any]]:
    """获取远程模型列表（如果本地版本更新则返回空）。"""
    if not entry:
        return []
    if local_generated_at is not None:
        last_modified = entry.get("lastModified")
        if last_modified is None or last_modified <= local_generated_at:
            return []
    return cast("list[dict[str, Any]]", entry.get("models", []))


def with_remote_catalog(
    provider: Provider,
    catalog_base_url: str = DEFAULT_CATALOG_BASE_URL,
    local_generated_at: int | None = None,
) -> Provider:
    """为内置 provider 添加远程目录覆盖层。"""
    dynamic_models: list[dict[str, Any]] = []

    async def refresh_models(context: RefreshModelsContext) -> None:
        nonlocal dynamic_models
        stored = cast(Any, context.stored)
        restored = [
            m
            for m in _remote_models(stored, local_generated_at)
            if m.get("provider") == provider.id
        ]
        # 先恢复缓存
        dynamic_models[:] = restored

        if not context.allow_network:
            return
        if (
            context.signal
            and hasattr(context.signal, "aborted")
            and context.signal.aborted
        ):
            return

        checked_at = int(time.time() * 1000)
        if (
            not context.force
            and stored
            and stored.get("checkedAt") is not None
            and stored.get("lastModified") is not None
            and checked_at - stored["checkedAt"] < REMOTE_CATALOG_REFRESH_INTERVAL_MS
        ):
            return

        validator = stored["etag"] if stored and stored.get("models") else None
        import urllib.parse

        url = f"{catalog_base_url}/api/models/providers/{urllib.parse.quote(provider.id, safe='')}"

        try:
            import httpx

            headers: dict[str, str] = {
                "accept": "application/json",
                "User-Agent": f"pi-coding-agent/{VERSION}",
            }
            if validator:
                headers["if-none-match"] = validator

            async with httpx.AsyncClient() as client:
                response = await client.get(url, headers=headers, timeout=30.0)
        except Exception:
            return

        if (
            context.signal
            and hasattr(context.signal, "aborted")
            and context.signal.aborted
        ):
            return

        # 304: 未变更
        if response.status_code == 304 and stored:
            stored["checkedAt"] = checked_at
            return

        if response.status_code in (404, 501):
            return

        if not response.is_success:
            raise RuntimeError(
                f"Model catalog request failed for {provider.id}: {response.status_code}"
            )

        refreshed = _parse_catalog(provider.id, response.json())

        last_modified_str = response.headers.get("last-modified")
        last_modified = 0
        if last_modified_str:
            try:
                from email.utils import parsedate_to_datetime

                last_modified = int(
                    parsedate_to_datetime(last_modified_str).timestamp() * 1000
                )
            except Exception:
                pass

        published = _remote_models(
            {
                "models": refreshed,
                "checkedAt": checked_at,
                "lastModified": last_modified,
                "etag": response.headers.get("etag"),
            },
            local_generated_at,
        )
        dynamic_models[:] = published

    # 修改 provider 的 get_models 和 refresh_models 方法
    # 在原始 _ProviderImpl 实例上直接修改 _get_models 和 _refresh_models
    from pi_ai.models import _ProviderImpl

    if isinstance(provider, _ProviderImpl):
        original_get_models = provider._get_models
        original_refresh_models = provider._refresh_models

        provider._get_models = lambda: [
            _dict_to_model(m)
            for m in _merge_models(
                [
                    cast("dict[str, Any]", m.model_dump(by_alias=True))
                    if hasattr(m, "model_dump")
                    else cast("dict[str, Any]", m)
                    for m in original_get_models()
                ],
                dynamic_models,
            )
        ]
        provider._refresh_models = refresh_models
        return provider

    # 非 _ProviderImpl 类型，使用 dict 包装
    return cast(
        Provider,
        {
            "id": provider.id,
            "name": provider.name,
            "base_url": provider.base_url,
            "headers": provider.headers,
            "get_models": lambda: _merge_models(
                [cast("dict[str, Any]", m) for m in provider.get_models()],
                dynamic_models,
            ),
            "refresh_models": refresh_models,
            "filter_models": provider.filter_models,
            "stream": provider.stream,
            "stream_simple": provider.stream_simple,
            "fetch_deferred": provider.fetch_deferred,
            "cancel_deferred": provider.cancel_deferred,
        },
    )
