"""Provider 环境变量解析。"""

from __future__ import annotations

import os

from ..types import ProviderEnv

_proc_env_cache: dict[str, str] | None = None


def _get_bun_sandbox_env_value(name: str) -> str | None:
    """Bun 沙箱环境变量回退。

    Python 侧无 Bun 沙箱问题，此函数保留接口兼容性。
    """
    return None


def get_provider_env_value(
    name: str, env: ProviderEnv | None = None
) -> str | None:
    """解析 provider 环境变量值。

    优先级：``env`` 显式覆盖 > ``os.environ`` 进程环境变量 > Bun 沙箱回退。
    """
    if env is not None and name in env:
        return env[name]
    value = os.environ.get(name)
    if value is not None:
        return value
    return _get_bun_sandbox_env_value(name)