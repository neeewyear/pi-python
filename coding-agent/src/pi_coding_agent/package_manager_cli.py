"""包管理器 CLI。

提供安装/更新/卸载扩展包的 CLI 入口。
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import Literal

from .config import APP_NAME, get_agent_dir
from .core.package_manager import DefaultPackageManager
from .core.settings_manager import SettingsManager

# ---------------------------------------------------------------------------
# 类型定义
# ---------------------------------------------------------------------------

PackageCommand = Literal["install", "remove", "update", "list"]


# ---------------------------------------------------------------------------
# 入口函数
# ---------------------------------------------------------------------------


def run_package_manager_cli(args: list[str] | None = None) -> None:
    """包管理器 CLI 入口。

    解析命令行参数并执行对应的包管理操作。

    Args:
        args: 命令行参数列表。默认为 ``sys.argv[1:]``。
    """
    if args is None:
        args = sys.argv[1:]

    if not args:
        _print_usage()
        sys.exit(1)

    command = args[0]
    rest = args[1:]

    if command in ("install", "remove", "update", "list", "uninstall"):
        asyncio.run(_handle_package_command(command, rest))
    elif command == "config":
        asyncio.run(_handle_config_command(rest))
    elif command in ("-h", "--help"):
        _print_usage()
    else:
        print(f"Unknown command: {command}", file=sys.stderr)
        _print_usage()
        sys.exit(1)


# ---------------------------------------------------------------------------
# 内部实现
# ---------------------------------------------------------------------------


def _print_usage() -> None:
    """打印使用说明。"""
    print(f"Usage: {APP_NAME} <command> [options]")
    print()
    print("Commands:")
    print("  install <source>    Install a package")
    print("  remove <source>     Remove a package")
    print("  update [target]     Update pi, packages, or models")
    print("  list                List installed packages")
    print("  config              Open resource configuration TUI")
    print()
    print(f"Run `{APP_NAME} <command> --help` for detailed usage.")


async def _handle_package_command(command: str, args: list[str]) -> None:
    """处理包管理命令。

    Args:
        command: 命令名。
        args: 命令参数。
    """
    if command == "uninstall":
        command = "remove"

    if not args or args[0] in ("-h", "--help"):
        _print_command_help(command)
        return

    cwd = os.getcwd()
    agent_dir = get_agent_dir()

    # 解析通用选项
    local = False
    source: str | None = None
    for arg in args:
        if arg in ("-l", "--local"):
            local = True
        elif not arg.startswith("-"):
            source = arg

    if command in ("install", "remove") and not source:
        print(f"Missing {command} source.", file=sys.stderr)
        sys.exit(1)

    assert source is not None  # 已在上方检查，确保 source 不为 None

    settings_manager = SettingsManager.create(cwd, str(agent_dir))
    package_manager = DefaultPackageManager(
        {
            "cwd": cwd,
            "agent_dir": str(agent_dir),
            "settings_manager": settings_manager,
        }
    )

    try:
        if command == "install":
            await package_manager.install_and_persist(source, {"local": local})
            print(f"Installed {source}")
        elif command == "remove":
            removed = await package_manager.remove_and_persist(source, {"local": local})
            if not removed:
                print(f"No matching package found for {source}", file=sys.stderr)
                sys.exit(1)
            print(f"Removed {source}")
        elif command == "update":
            await package_manager.update(source)
            print(f"Updated {source or 'packages'}")
        elif command == "list":
            configured_packages = package_manager.list_configured_packages()
            if not configured_packages:
                print("No packages installed.")
                return

            for pkg in configured_packages:
                display = f"{pkg.source} (filtered)" if pkg.filtered else pkg.source
                print(f"  {display}")
                if pkg.installed_path:
                    print(f"    {pkg.installed_path}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


async def _handle_config_command(args: list[str]) -> None:
    """处理配置命令。

    Args:
        args: 命令参数。
    """
    if args and args[0] in ("-h", "--help"):
        print(f"Usage: {APP_NAME} config [-l]")
        print()
        print("Open the resource configuration TUI.")
        sys.exit(0)

    print("Config command TUI is not yet implemented in Python.", file=sys.stderr)
    sys.exit(1)


def _print_command_help(command: str) -> None:
    """打印命令帮助信息。

    Args:
        command: 命令名。
    """
    if command == "install":
        print(f"Usage: {APP_NAME} install <source> [-l]")
        print()
        print("Install a package and add it to settings.")
        print()
        print("Options:")
        print("  -l, --local    Install project-locally")
        print()
        print("Examples:")
        print(f"  {APP_NAME} install npm:@foo/bar")
        print(f"  {APP_NAME} install git:github.com/user/repo")
    elif command == "remove":
        print(f"Usage: {APP_NAME} remove <source> [-l]")
        print()
        print("Remove a package and its source from settings.")
        print()
        print("Options:")
        print("  -l, --local    Remove from project settings")
        print()
        print("Examples:")
        print(f"  {APP_NAME} remove npm:@foo/bar")
    elif command == "update":
        print(
            f"Usage: {APP_NAME} update [source|self|pi] "
            "[--all] [--extensions] [--models] [--force]"
        )
        print()
        print("Update pi, installed packages, or model catalogs.")
        print()
        print("Options:")
        print("  --self          Update pi only (default)")
        print("  --extensions    Update installed packages only")
        print("  --models        Refresh model catalogs only")
        print("  --all           Update pi and installed packages")
        print("  --force         Reinstall pi even if current version is latest")
    elif command == "list":
        print(f"Usage: {APP_NAME} list")
        print()
        print("List installed packages.")
    else:
        _print_usage()


__all__: list[str] = [
    "run_package_manager_cli",
]
