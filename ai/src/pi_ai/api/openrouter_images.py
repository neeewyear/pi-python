"""OpenRouter 图片生成 API。"""

from __future__ import annotations

import re
import time
from typing import Any, TypedDict, cast

from openai import AsyncOpenAI

from ..utils.error_body import format_provider_error, normalize_provider_error
from ..utils.headers import headers_to_record
from ..utils.provider_retry import ProviderRetryOptions, retry_provider_request
from ..utils.sanitize_unicode import sanitize_surrogates


class OpenRouterGeneratedImage(TypedDict, total=False):
    """OpenRouter 图片生成响应中的单个图片。"""

    image_url: str | dict[str, str] | None


async def generate_images(
    model: Any,
    context: Any,
    options: Any | None = None,
) -> dict[str, Any]:
    """生成图片。

    Args:
        model: 图片生成模型句柄。
        context: 图片生成上下文。
        options: 可选请求选项。

    Returns:
        AssistantImages 格式字典。
    """
    output: dict[str, Any] = {
        "api": getattr(model, "api", ""),
        "provider": getattr(model, "provider", ""),
        "model": getattr(model, "id", getattr(model, "model_id", "")),
        "output": [],
        "stop_reason": "stop",
        "timestamp": int(time.time() * 1000),
    }

    try:
        api_key = getattr(options, "api_key", None) or getattr(model, "api_key", None)
        if not api_key:
            raise ValueError(
                f"No API key for provider: {getattr(model, 'provider', 'unknown')}"
            )

        client = _create_client(model, api_key, options)
        params = _build_params(model, context)

        # onPayload 回调
        on_payload = getattr(options, "on_payload", None)
        if on_payload is not None:
            next_params = await on_payload(params, model)
            if next_params is not None:
                params = cast(dict[str, Any], next_params)

        # 请求选项
        signal = getattr(options, "signal", None)
        request_options: dict[str, Any] = {"max_retries": 0}
        if signal is not None:
            request_options["signal"] = signal
        timeout_ms = getattr(options, "timeout_ms", None)
        if timeout_ms is not None:
            request_options["timeout"] = timeout_ms

        # 发起请求（带重试）
        raw_response = await retry_provider_request(
            lambda: client.chat.completions.with_raw_response.create(
                **params, **request_options
            ),
            ProviderRetryOptions(
                max_retries=getattr(options, "max_retries", 0) or 0,
                max_retry_delay_ms=getattr(options, "max_retry_delay_ms", None),
                signal=signal,
            ),
        )

        # 解析响应
        raw_response_obj = cast("Any", raw_response)
        response = raw_response_obj.parse()
        response_headers = dict(raw_response_obj.headers)

        # onResponse 回调
        on_response = getattr(options, "on_response", None)
        if on_response is not None:
            await on_response(
                {
                    "status": getattr(raw_response, "status_code", 0),
                    "headers": headers_to_record(response_headers),
                },
                model,
            )

        output["response_id"] = getattr(response, "id", None)

        # 解析 token 用量
        usage = getattr(response, "usage", None)
        if usage is not None:
            output["usage"] = _parse_usage(usage, model)

        # 解析响应内容
        choices = getattr(response, "choices", [])
        if choices:
            choice = choices[0]
            message = getattr(choice, "message", None)
            if message is not None:
                content = getattr(message, "content", None)
                if isinstance(content, str) and content:
                    output["output"].append({"type": "text", "text": content})

                images = getattr(message, "images", None) or []
                for image in images:
                    image_url = _extract_image_url(image)
                    if not image_url or not image_url.startswith("data:"):
                        continue
                    matches = re.match(r"^data:([^;]+);base64,(.+)$", image_url)
                    if not matches:
                        continue
                    output["output"].append(
                        {
                            "type": "image",
                            "mime_type": matches.group(1),
                            "data": matches.group(2),
                        }
                    )

        return output

    except Exception as error:
        signal = getattr(options, "signal", None)
        if signal is not None and getattr(signal, "aborted", False):
            output["stop_reason"] = "aborted"
        else:
            output["stop_reason"] = "error"
        output["error_message"] = format_provider_error(
            normalize_provider_error(error)
        )
        return output


def _extract_image_url(image: Any) -> str | None:
    """从图片对象中提取 URL。"""
    if isinstance(image, dict):
        raw = image.get("image_url")
    else:
        raw = getattr(image, "image_url", None)
    if raw is None:
        return None
    if isinstance(raw, str):
        return raw
    if isinstance(raw, dict):
        url = raw.get("url")
        return url if isinstance(url, str) else None
    return None


def _create_client(
    model: Any,
    api_key: str,
    options: Any | None = None,
) -> AsyncOpenAI:
    """创建 OpenAI 客户端。"""
    headers: dict[str, str | None] = {}

    model_headers = getattr(model, "headers", None) or {}
    if isinstance(model_headers, dict):
        headers.update(model_headers)

    options_headers = getattr(options, "headers", None) or {}
    if isinstance(options_headers, dict):
        headers.update(options_headers)

    filtered_headers: dict[str, str] = {
        k: v for k, v in headers.items() if v is not None
    }

    return AsyncOpenAI(
        api_key=api_key,
        base_url=getattr(model, "base_url", ""),
        default_headers=filtered_headers or None,
        http_client=getattr(options, "fetch", None),
    )


def _build_params(model: Any, context: Any) -> dict[str, Any]:
    """构建请求参数。"""
    prompt = getattr(context, "prompt", "")
    content: list[dict[str, Any]] = [
        {"type": "text", "text": sanitize_surrogates(prompt)}
    ]

    model_output = getattr(model, "output", ["image"])
    modalities: list[str] = ["image"]
    if "text" in model_output:
        modalities.append("text")

    return {
        "model": getattr(model, "id", getattr(model, "model_id", "")),
        "messages": [{"role": "user", "content": content}],
        "stream": False,
        "modalities": modalities,
    }


def _parse_usage(raw_usage: Any, model: Any) -> dict[str, Any]:
    """解析 token 用量。"""
    prompt_tokens = getattr(raw_usage, "prompt_tokens", 0) or 0
    completion_tokens = getattr(raw_usage, "completion_tokens", 0) or 0

    prompt_tokens_details = getattr(raw_usage, "prompt_tokens_details", None)
    reported_cached_tokens = 0
    cache_write_tokens = 0
    if prompt_tokens_details is not None:
        reported_cached_tokens = (
            getattr(prompt_tokens_details, "cached_tokens", 0) or 0
        )
        cache_write_tokens = (
            getattr(prompt_tokens_details, "cache_write_tokens", 0) or 0
        )

    cache_read_tokens = (
        max(0, reported_cached_tokens - cache_write_tokens)
        if cache_write_tokens > 0
        else reported_cached_tokens
    )
    input_tokens = max(0, prompt_tokens - cache_read_tokens - cache_write_tokens)
    output_tokens = completion_tokens

    model_cost = getattr(model, "cost", {})
    if isinstance(model_cost, dict):
        cost_input = model_cost.get("input", 0)
        cost_output = model_cost.get("output", 0)
        cost_cache_read = model_cost.get("cache_read", 0)
        cost_cache_write = model_cost.get("cache_write", 0)
    else:
        cost_input = getattr(model_cost, "input", 0)
        cost_output = getattr(model_cost, "output", 0)
        cost_cache_read = getattr(model_cost, "cache_read", 0)
        cost_cache_write = getattr(model_cost, "cache_write", 0)

    cost_input_val = (cost_input / 1000000) * input_tokens
    cost_output_val = (cost_output / 1000000) * output_tokens
    cost_cache_read_val = (cost_cache_read / 1000000) * cache_read_tokens
    cost_cache_write_val = (cost_cache_write / 1000000) * cache_write_tokens
    cost_total = (
        cost_input_val
        + cost_output_val
        + cost_cache_read_val
        + cost_cache_write_val
    )

    return {
        "input": input_tokens,
        "output": output_tokens,
        "cache_read": cache_read_tokens,
        "cache_write": cache_write_tokens,
        "total_tokens": input_tokens
        + output_tokens
        + cache_read_tokens
        + cache_write_tokens,
        "cost": {
            "input": cost_input_val,
            "output": cost_output_val,
            "cache_read": cost_cache_read_val,
            "cache_write": cost_cache_write_val,
            "total": cost_total,
        },
    }