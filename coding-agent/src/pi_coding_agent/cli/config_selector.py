"""配置选择器（对应 TS ``cli/config-selector.ts``）。

非 TUI 版本：使用 stdin 交互式选择配置。
由于 TUI 组件尚未移植到 Python，此版本提供简单的文本选择界面。
"""

from __future__ import annotations

import sys
from typing import Any

from ..config import CONFIG_DIR_NAME, get_agent_dir, get_settings_path


def show_config_selector(
    *,
    resolved_paths: Any = None,
    settings_manager: Any = None,
    cwd: str | None = None,
    agent_dir: str | None = None,
    write_scope: str = "global",
    project_mode_available: bool = False,
) -> None:
    """交互式配置选择（非 TUI 版本）。

    使用 stdin 提示用户选择要操作的配置范围。
    完整 TUI 实现需要 ``@earendil-works/pi-tui`` 支持。

    Args:
        resolved_paths: 已解析的资源路径。
        settings_manager: 设置管理器。
        cwd: 当前工作目录。
        agent_dir: agent 配置目录。
        write_scope: 写入范围（"global" 或 "project"）。
        project_mode_available: 是否可用项目模式。
    """
    # 确定配置目录
    resolved_agent_dir = agent_dir or str(get_agent_dir())
    settings_path = str(get_settings_path())

    print(f"Config directory: {resolved_agent_dir}", file=sys.stderr)
    print(f"Settings file: {settings_path}", file=sys.stderr)
    print(file=sys.stderr)

    # 如果可用项目模式，让用户选择范围
    if project_mode_available:
        print(f"Current write scope: {write_scope}", file=sys.stderr)
        print(file=sys.stderr)
        print("Available scopes:", file=sys.stderr)
        print("  1. global  - Apply to all projects", file=sys.stderr)
        print(
            f"  2. project - Apply only to this project ({CONFIG_DIR_NAME}/)",
            file=sys.stderr,
        )
        print(file=sys.stderr)
        if sys.stdin.isatty():
            choice = input("Select scope [1/2] (default: 1): ").strip()
            if choice == "2":
                print("Selected scope: project", file=sys.stderr)
            else:
                print("Selected scope: global", file=sys.stderr)
        else:
            print("(non-interactive mode, using default scope)", file=sys.stderr)
    else:
        print("Scope: global (project mode not available)", file=sys.stderr)

    print(file=sys.stderr)
    print(
        "To edit configuration, modify the settings.json file directly.",
        file=sys.stderr,
    )
    print("Or use: pi config --help", file=sys.stderr)


__all__ = [
    "show_config_selector",
]
