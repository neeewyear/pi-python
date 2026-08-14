"""HTTP 代理解析工具。

提供 ``resolve_http_proxy_url_for_target`` 函数，用于解析 HTTP 代理 URL。
"""

from __future__ import annotations

import os
import re
from urllib.parse import urlparse, urlunparse

from ..types import ProviderEnv
from .provider_env import get_provider_env_value

DEFAULT_PROXY_PORTS: dict[str, int] = {
    "ftp": 21,
    "gopher": 70,
    "http": 80,
    "https": 443,
    "ws": 80,
    "wss": 443,
}

UNSUPPORTED_PROXY_PROTOCOL_MESSAGE = (
    "Unsupported proxy protocol. SOCKS and PAC proxy URLs are not supported;"
    " use an HTTP or HTTPS proxy URL."
)


def _get_proxy_env(key: str, env: ProviderEnv | None = None) -> str:
    """获取代理环境变量。"""
    lowercase_key = key.lower()
    uppercase_key = key.upper()
    value = (
        (env.get(lowercase_key) if env else None)
        or (env.get(uppercase_key) if env else None)
        or get_provider_env_value(lowercase_key)
        or get_provider_env_value(uppercase_key)
        or ""
    )
    return value


def _parse_proxy_target_url(target_url: str) -> str | None:
    """解析代理目标 URL。"""
    try:
        parsed = urlparse(target_url if "://" in target_url else f"http://{target_url}")
        if parsed.netloc or parsed.hostname:
            return target_url
        return None
    except Exception:
        return None


def _should_proxy_hostname(
    hostname: str, port: int, env: ProviderEnv | None = None
) -> bool:
    """检查 hostname 是否应通过代理。"""
    no_proxy = _get_proxy_env("no_proxy", env).lower()
    if not no_proxy:
        return True
    if no_proxy == "*":
        return False

    for proxy in re.split(r"[,\s]", no_proxy):
        if not proxy:
            continue
        parsed = re.match(r"^(.+):(\d+)$", proxy)
        proxy_hostname = parsed.group(1) if parsed else proxy
        proxy_port = int(parsed.group(2)) if parsed else 0
        if proxy_port and proxy_port != port:
            continue
        if not proxy_hostname.startswith(".") and not proxy_hostname.startswith("*"):
            if hostname == proxy_hostname:
                return False
        else:
            if proxy_hostname.startswith("*"):
                proxy_hostname = proxy_hostname[1:]
            if hostname.endswith(proxy_hostname):
                return False
    return True


def _get_proxy_for_url(target_url: str, env: ProviderEnv | None = None) -> str:
    """获取目标 URL 的代理 URL。"""
    parsed = _parse_proxy_target_url(target_url)
    if not parsed:
        return ""

    parsed_url = urlparse(parsed)
    if not parsed_url.scheme or not parsed_url.hostname:
        return ""

    protocol = parsed_url.scheme.split(":")[0]
    hostname = parsed_url.hostname
    port = parsed_url.port or DEFAULT_PROXY_PORTS.get(protocol, 0)
    if not _should_proxy_hostname(hostname, port, env):
        return ""

    proxy = _get_proxy_env(f"{protocol}_proxy", env) or _get_proxy_env("all_proxy", env)
    if proxy and "://" not in proxy:
        proxy = f"{protocol}://{proxy}"
    return proxy


def resolve_http_proxy_url_for_target(
    target_url: str, env: ProviderEnv | None = None
) -> str | None:
    """解析目标 URL 的 HTTP 代理 URL。

    返回代理 URL 字符串，无代理时返回 ``None``。
    """
    proxy = _get_proxy_for_url(target_url, env)
    if not proxy:
        return None

    try:
        proxy_url = urlparse(proxy)
    except Exception as exc:
        raise ValueError(
            f"Invalid proxy URL {proxy!r}: {exc}"
        ) from exc

    if proxy_url.scheme not in ("http", "https"):
        raise ValueError(
            f"{UNSUPPORTED_PROXY_PROTOCOL_MESSAGE} Got {proxy_url.scheme}"
        )

    return proxy