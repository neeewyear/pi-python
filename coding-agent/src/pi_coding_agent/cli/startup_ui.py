"""启动 UI（对应 TS ``cli/startup-ui.ts``）。

非 TUI 版本：使用 stdin 文本交互。
由于 TUI 组件尚未移植到 Python，此版本提供简单的文本选择界面。
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import (
    APP_NAME,
    CONFIG_DIR_NAME,
    ENV_AGENT_DIR,
    PACKAGE_NAME,
    get_settings_path,
)
from ..core.experimental import are_experimental_features_enabled


@dataclass
class StartupUIOptions:
    """启动 UI 选项。"""

    resolved_paths: Any = None
    """已解析的资源路径。"""
    settings_manager: Any = None
    """设置管理器。"""
    cwd: str | None = None
    """当前工作目录。"""
    agent_dir: str | None = None
    """agent 配置目录。"""
    has_ui: bool = False
    """是否具有 UI 能力。"""
    should_run_first_time_setup: bool = False
    """是否应运行首次设置。"""


# 官方分发元数据常量
_OFFICIAL_PACKAGE_NAME = "@earendil-works/pi-coding-agent"
_OFFICIAL_APP_NAME = "pi"
_OFFICIAL_CONFIG_DIR_NAME = ".pi"


def _is_official_distribution(
    *,
    package_name: str = PACKAGE_NAME,
    app_name: str = APP_NAME,
    config_dir_name: str = CONFIG_DIR_NAME,
) -> bool:
    """检查是否为官方分发版。"""
    return (
        package_name == _OFFICIAL_PACKAGE_NAME
        and app_name == _OFFICIAL_APP_NAME
        and config_dir_name == _OFFICIAL_CONFIG_DIR_NAME
    )


def should_run_first_time_setup(
    settings_path: str | None = None,
) -> bool:
    """检查是否应运行首次设置。

    当以下条件全部满足时返回 True：
    - 这是官方 Pi 分发版（不是 fork/rebrand）
    - 启用了实验性功能（PI_EXPERIMENTAL=1）
    - 使用默认 agent 目录（没有自定义覆盖）
    - 设置尚未完成（settings.json 不存在）

    Args:
        settings_path: 设置文件路径，默认使用 ``get_settings_path()``。

    Returns:
        是否应运行首次设置。
    """
    resolved_settings_path = (
        Path(settings_path) if settings_path else get_settings_path()
    )

    if not _is_official_distribution():
        return False
    if not are_experimental_features_enabled():
        return False
    if os.environ.get(ENV_AGENT_DIR):
        return False
    return not resolved_settings_path.exists()


async def show_startup_selector(
    settings_manager: Any,
    title: str,
    options: list[dict[str, Any]],
) -> Any | None:
    """显示启动选择器（非 TUI 版本）。

    使用 stdin 文本列表让用户选择。

    Args:
        settings_manager: 设置管理器。
        title: 选择器标题。
        options: 选项列表，每个选项包含 ``label`` 和 ``value`` 键。

    Returns:
        选中的值，取消返回 None。
    """
    print(file=sys.stderr)
    print(f"=== {title} ===", file=sys.stderr)
    print(file=sys.stderr)

    for i, option in enumerate(options, start=1):
        label = option.get("label", f"Option {i}")
        print(f"  {i}. {label}", file=sys.stderr)

    print("  q. Cancel", file=sys.stderr)
    print(file=sys.stderr)

    if not sys.stdin.isatty():
        print("(non-interactive mode, selecting first option)", file=sys.stderr)
        return options[0].get("value") if options else None

    while True:
        choice = input(f"Select [1-{len(options)}] (default: 1): ").strip().lower()
        if choice == "q" or choice == "":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx].get("value")
        except ValueError:
            pass
        print(
            f"Invalid choice. Enter 1-{len(options)} or 'q' to cancel.", file=sys.stderr
        )


async def show_first_time_setup(settings_manager: Any) -> None:
    """显示首次设置对话框（非 TUI 版本）。

    使用 stdin 文本提示收集用户首选项（主题、分析共享等）。

    Args:
        settings_manager: 设置管理器。
    """
    print(file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(f"  Welcome to {APP_NAME} - First Time Setup", file=sys.stderr)
    print("=" * 50, file=sys.stderr)
    print(file=sys.stderr)

    if not sys.stdin.isatty():
        print("(non-interactive mode, skipping first time setup)", file=sys.stderr)
        return

    # 主题选择
    theme = input("Choose theme [dark/light] (default: dark): ").strip().lower()
    if theme not in ("dark", "light"):
        theme = "dark"
    print(f"  Selected theme: {theme}", file=sys.stderr)

    # 分析共享选择
    analytics = (
        input("Share anonymous usage data to improve the product? [y/N]: ")
        .strip()
        .lower()
    )
    share_analytics = analytics == "y"
    print(f"  Share analytics: {'Yes' if share_analytics else 'No'}", file=sys.stderr)

    print(file=sys.stderr)

    # 持久化设置
    if hasattr(settings_manager, "set_theme"):
        settings_manager.set_theme(theme)
    if hasattr(settings_manager, "set_enable_analytics"):
        settings_manager.set_enable_analytics(share_analytics)
    if hasattr(settings_manager, "flush"):
        settings_manager.flush()

    print("Setup complete! Configuration saved.", file=sys.stderr)
    print(file=sys.stderr)


async def show_startup_input(
    settings_manager: Any,
    title: str,
    placeholder: str | None = None,
) -> str | None:
    """显示启动输入框（非 TUI 版本）。

    使用 stdin 文本输入提示。

    Args:
        settings_manager: 设置管理器。
        title: 输入框标题。
        placeholder: 输入占位符文本。

    Returns:
        输入的文本，取消返回 None。
    """
    print(file=sys.stderr)
    print(f"=== {title} ===", file=sys.stderr)
    if placeholder:
        print(f"({placeholder})", file=sys.stderr)
    print(file=sys.stderr)

    if not sys.stdin.isatty():
        print("(non-interactive mode, returning None)", file=sys.stderr)
        return None

    value = input("Enter value (or 'q' to cancel): ").strip()
    if value.lower() == "q":
        return None
    return value


async def show_startup_ui(options: StartupUIOptions | None = None) -> None:
    """显示启动 UI（非 TUI 版本）。

    根据选项显示启动选择器、首次设置或输入框。

    Args:
        options: 启动 UI 选项。
    """
    opts = options or StartupUIOptions()
    settings_manager = opts.settings_manager

    if opts.should_run_first_time_setup:
        await show_first_time_setup(settings_manager)


__all__ = [
    "StartupUIOptions",
    "should_run_first_time_setup",
    "show_first_time_setup",
    "show_startup_input",
    "show_startup_selector",
    "show_startup_ui",
]
