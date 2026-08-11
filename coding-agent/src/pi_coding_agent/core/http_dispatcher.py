from __future__ import annotations

import math
import os
from typing import Any

DEFAULT_HTTP_IDLE_TIMEOUT_MS = 300_000
DEFAULT_AUTO_SELECT_FAMILY_ATTEMPT_TIMEOUT_MS = 2_000

HTTP_IDLE_TIMEOUT_CHOICES = [
    {"label": "30 sec", "timeout_ms": 30_000},
    {"label": "1 min", "timeout_ms": 60_000},
    {"label": "2 min", "timeout_ms": 120_000},
    {"label": "5 min", "timeout_ms": 300_000},
    {"label": "disabled", "timeout_ms": 0},
]


def parse_http_idle_timeout_ms(value: object) -> int | None:
    if isinstance(value, str):
        trimmed = value.strip()
        if trimmed.lower() == "disabled":
            return 0
        if len(trimmed) == 0:
            return None
        return parse_http_idle_timeout_ms(int(trimmed))
    if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0:
        return math.floor(value)
    return None


def format_http_idle_timeout_ms(timeout_ms: int) -> str:
    for choice in HTTP_IDLE_TIMEOUT_CHOICES:
        if choice["timeout_ms"] == timeout_ms:
            return str(choice["label"])
    return f"{timeout_ms / 1000} sec"


def apply_http_proxy_settings(http_proxy: str | None) -> None:
    proxy = http_proxy.strip() if http_proxy else None
    if not proxy:
        return
    os.environ.setdefault("HTTP_PROXY", proxy)
    os.environ.setdefault("HTTPS_PROXY", proxy)


def configure_http_dispatcher(timeout_ms: int = DEFAULT_HTTP_IDLE_TIMEOUT_MS) -> None:
    normalized_timeout_ms = parse_http_idle_timeout_ms(timeout_ms)
    if normalized_timeout_ms is None:
        raise ValueError(f"Invalid HTTP idle timeout: {timeout_ms!s}")
    # Configure httpx client with appropriate settings
    import httpx

    client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            connect=DEFAULT_AUTO_SELECT_FAMILY_ATTEMPT_TIMEOUT_MS / 1000,
            read=normalized_timeout_ms / 1000 if normalized_timeout_ms > 0 else None,
            write=normalized_timeout_ms / 1000 if normalized_timeout_ms > 0 else None,
            pool=normalized_timeout_ms / 1000 if normalized_timeout_ms > 0 else None,
        ),
        limits=httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=normalized_timeout_ms / 1000
            if normalized_timeout_ms > 0
            else 0,
        ),
    )
    # Store as global default
    global _default_http_client
    _default_http_client = client


_default_http_client = None


def get_default_http_client() -> Any:
    global _default_http_client
    if _default_http_client is None:
        configure_http_dispatcher()
    return _default_http_client
