"""AuthContext 实现"""

from __future__ import annotations

import os
from pathlib import Path

from .types import AuthContext


def default_provider_auth_context() -> AuthContext:
    """默认认证上下文：从 ``os.environ`` 读取环境变量，通过 ``pathlib.Path.exists`` 检查文件。"""
    class _DefaultAuthContext:
        async def env(self, name: str) -> str | None:
            value = os.environ.get(name)
            return value if value and value.strip() else None

        async def file_exists(self, path: str) -> bool:
            try:
                resolved = path
                if resolved.startswith("~"):
                    resolved = str(Path.home()) + resolved[1:]
                return Path(resolved).exists()
            except (OSError, ValueError):
                return False

    return _DefaultAuthContext()