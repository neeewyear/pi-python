from __future__ import annotations

from typing import TYPE_CHECKING

from .telemetry import is_install_telemetry_enabled

if TYPE_CHECKING:
    from pi_ai.models import ModelRecord
    from pi_ai.types import ProviderHeaders

    from .settings_manager import SettingsManager

OPENROUTER_HOST = "openrouter.ai"
NVIDIA_NIM_HOST = "integrate.api.nvidia.com"
CLOUDFLARE_API_HOST = "api.cloudflare.com"
CLOUDFLARE_AI_GATEWAY_HOST = "gateway.ai.cloudflare.com"
OPENCODE_HOST = "opencode.ai"


def _matches_host(base_url: str, expected_host: str) -> bool:
    from urllib.parse import urlparse

    try:
        return urlparse(base_url).hostname == expected_host
    except Exception:
        return False


def _is_openrouter_model(model: ModelRecord) -> bool:
    return model.provider == "openrouter" or OPENROUTER_HOST in model.base_url


def _is_nvidia_nim_model(model: ModelRecord) -> bool:
    return model.provider == "nvidia" or _matches_host(model.base_url, NVIDIA_NIM_HOST)


def _is_cloudflare_model(model: ModelRecord) -> bool:
    return (
        model.provider == "cloudflare-workers-ai"
        or model.provider == "cloudflare-ai-gateway"
        or _matches_host(model.base_url, CLOUDFLARE_API_HOST)
        or _matches_host(model.base_url, CLOUDFLARE_AI_GATEWAY_HOST)
    )


def _get_default_attribution_headers(
    model: ModelRecord,
    settings_manager: SettingsManager,
) -> dict[str, str] | None:
    if not is_install_telemetry_enabled(settings_manager):
        return None

    if _is_openrouter_model(model):
        return {
            "HTTP-Referer": "https://pi.dev",
            "X-OpenRouter-Title": "pi",
            "X-OpenRouter-Categories": "cli-agent",
        }

    if _is_nvidia_nim_model(model):
        return {
            "X-BILLING-INVOKE-ORIGIN": "Pi",
        }

    if _is_cloudflare_model(model):
        return {
            "User-Agent": "pi-coding-agent",
        }

    return None


def _get_session_headers(
    model: ModelRecord,
    session_id: str | None,
) -> dict[str, str] | None:
    if not session_id:
        return None
    if (
        model.provider != "opencode"
        and model.provider != "opencode-go"
        and not _matches_host(model.base_url, OPENCODE_HOST)
    ):
        return None
    return {"x-opencode-session": session_id, "x-opencode-client": "pi"}


def merge_provider_attribution_headers(
    model: ModelRecord,
    settings_manager: SettingsManager,
    session_id: str | None,
    *header_sources: ProviderHeaders | None,
) -> ProviderHeaders | None:
    merged: dict[str, str | None] = {}

    session_headers = _get_session_headers(model, session_id)
    if session_headers:
        merged.update(session_headers)

    default_headers = _get_default_attribution_headers(model, settings_manager)
    if default_headers:
        merged.update(default_headers)

    for headers in header_sources:
        if headers:
            merged.update(headers)

    return merged if any(v is not None for v in merged.values()) else None
