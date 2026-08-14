"""OAuth 类型兼容入口。

仅用于 coding-agent 扩展 OAuth 声明兼容。
"""

from __future__ import annotations

from .auth.types import OAuthCredentials
from .compat import (
    OAuthAuthInfo,
    OAuthDeviceCodeInfo,
    OAuthLoginCallbacks,
    OAuthPrompt,
    OAuthSelectOption,
    OAuthSelectPrompt,
)

__all__ = [
    "OAuthAuthInfo",
    "OAuthCredentials",
    "OAuthDeviceCodeInfo",
    "OAuthLoginCallbacks",
    "OAuthPrompt",
    "OAuthSelectOption",
    "OAuthSelectPrompt",
]
