from __future__ import annotations

import os
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .settings_manager import SettingsManager


def _is_truthy_env_flag(value: Optional[str]) -> bool:
    if not value:
        return False
    return value == "1" or value.lower() in ("true", "yes")


def is_install_telemetry_enabled(
    settings_manager: "SettingsManager",
    telemetry_env: Optional[str] = None,
) -> bool:
    if telemetry_env is None:
        telemetry_env = os.environ.get("PI_TELEMETRY")
    if telemetry_env is not None:
        return _is_truthy_env_flag(telemetry_env)
    return settings_manager.get_enable_install_telemetry()