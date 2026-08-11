"""会话迁移（对应 TS ``migrations.ts``）。

提供启动时运行的一次性数据迁移函数，包括认证凭据迁移、
会话文件位置迁移、扩展目录重命名、二进制工具迁移等。
"""

from __future__ import annotations

import json
import os
import sys
import termios
import tty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MIGRATION_GUIDE_URL = (
    "https://github.com/earendil-works/pi-mono/blob/main/packages/"
    "coding-agent/CHANGELOG.md#extensions-migration"
)
EXTENSIONS_DOC_URL = (
    "https://github.com/earendil-works/pi-mono/blob/main/packages/"
    "coding-agent/docs/extensions.md"
)


# ---------------------------------------------------------------------------
# SessionMigration dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SessionMigration:
    """表示一次会话迁移操作的结果。

    Attributes:
        migrated_auth_providers: 已迁移的认证提供者名称列表。
        deprecation_warnings: 弃用警告列表。
    """

    migrated_auth_providers: list[str] = field(default_factory=list)
    deprecation_warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 认证迁移
# ---------------------------------------------------------------------------


def migrate_auth_to_auth_json() -> list[str]:
    """将遗留的 ``oauth.json`` 和 ``settings.json`` 中的 ``apiKeys`` 迁移到 ``auth.json``。

    Returns:
        已迁移的提供者名称列表。
    """
    from .config import get_agent_dir

    agent_dir = get_agent_dir()
    auth_path = agent_dir / "auth.json"
    oauth_path = agent_dir / "oauth.json"
    settings_path = agent_dir / "settings.json"

    if auth_path.exists():
        return []

    migrated: dict[str, Any] = {}
    providers: list[str] = []

    # 迁移 oauth.json
    if oauth_path.exists():
        try:
            oauth_data = json.loads(oauth_path.read_text("utf-8"))
            for provider, cred in oauth_data.items():
                migrated[provider] = {"type": "oauth", **cred}
                providers.append(provider)
            oauth_path.rename(oauth_path.with_suffix(".json.migrated"))
        except Exception:
            pass

    # 迁移 settings.json 中的 apiKeys
    if settings_path.exists():
        try:
            content = settings_path.read_text("utf-8")
            settings = json.loads(content)
            api_keys = settings.get("apiKeys")
            if isinstance(api_keys, dict):
                for provider, key in api_keys.items():
                    if provider not in migrated and isinstance(key, str):
                        migrated[provider] = {"type": "api_key", "key": key}
                        providers.append(provider)
                del settings["apiKeys"]
                settings_path.write_text(
                    json.dumps(settings, indent=2, ensure_ascii=False),
                    "utf-8",
                )
        except Exception:
            pass

    if migrated:
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        auth_path.write_text(
            json.dumps(migrated, indent=2, ensure_ascii=False),
            "utf-8",
        )
        # 设置权限为 0o600
        auth_path.chmod(0o600)

    return providers


# ---------------------------------------------------------------------------
# 会话位置迁移
# ---------------------------------------------------------------------------


def migrate_sessions_from_agent_root() -> None:
    """将 ``~/.pi/agent/*.jsonl`` 迁移到 ``~/.pi/agent/sessions/<encoded-cwd>/``。

    v0.30.0 的 bug 导致会话被保存到 ``~/.pi/agent/`` 而非正确的子目录。
    此函数根据会话头中的 ``cwd`` 将其移动到正确位置。
    """
    from .config import get_agent_dir

    agent_dir = get_agent_dir()

    # 查找 agentDir 下的所有 .jsonl 文件（非子目录）
    try:
        files = [p for p in agent_dir.iterdir() if p.suffix == ".jsonl" and p.is_file()]
    except Exception:
        return

    if not files:
        return

    for file in files:
        try:
            content = file.read_text("utf-8")
            first_line = content.split("\n")[0]
            if not first_line or not first_line.strip():
                continue

            header = json.loads(first_line)
            if header.get("type") != "session" or not header.get("cwd"):
                continue

            cwd: str = header["cwd"]

            # 计算正确的会话目录（与 session-manager.ts 编码一致）
            stripped = cwd.lstrip("/\\")
            safe_path = (
                "--"
                + stripped.replace("/", "-").replace("\\", "-").replace(":", "-")
                + "--"
            )
            correct_dir = agent_dir / "sessions" / safe_path

            # 创建目录
            correct_dir.mkdir(parents=True, exist_ok=True)

            # 移动文件
            new_path = correct_dir / file.name
            if new_path.exists():
                continue

            file.rename(new_path)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# 扩展目录迁移
# ---------------------------------------------------------------------------


def migrate_commands_to_prompts(base_dir: Path, label: str) -> bool:
    """将 ``commands/`` 目录重命名为 ``prompts/``。

    Args:
        base_dir: 基础目录。
        label: 用于日志的标签（如 "Global" / "Project"）。

    Returns:
        是否执行了迁移。
    """
    commands_dir = base_dir / "commands"
    prompts_dir = base_dir / "prompts"

    if commands_dir.exists() and not prompts_dir.exists():
        try:
            commands_dir.rename(prompts_dir)
            print(f"Migrated {label} commands/ → prompts/")
            return True
        except Exception as err:
            print(
                f"Warning: Could not migrate {label} commands/ to prompts/: {err}",
                file=sys.stderr,
            )
    return False


# ---------------------------------------------------------------------------
# 键绑定迁移
# ---------------------------------------------------------------------------


def migrate_keybindings_config_file() -> None:
    """迁移键绑定配置文件。"""
    from .config import get_agent_dir
    from .core.keybindings import migrate_keybindings_config

    config_path = get_agent_dir() / "keybindings.json"
    if not config_path.exists():
        return

    try:
        parsed = json.loads(config_path.read_text("utf-8"))
        if not isinstance(parsed, dict):
            return
        config, migrated = migrate_keybindings_config(parsed)
        if not migrated:
            return
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n",
            "utf-8",
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# 二进制工具迁移
# ---------------------------------------------------------------------------


def migrate_tools_to_bin() -> None:
    """将 ``fd``/``rg`` 二进制文件从 ``tools/`` 移动到 ``bin/``。"""
    from .config import get_agent_dir, get_bin_dir

    agent_dir = get_agent_dir()
    tools_dir = agent_dir / "tools"
    bin_dir = get_bin_dir()

    if not tools_dir.exists():
        return

    binaries = ["fd", "rg", "fd.exe", "rg.exe"]
    moved_any = False

    for bin_name in binaries:
        old_path = tools_dir / bin_name
        new_path = bin_dir / bin_name

        if old_path.exists():
            if not bin_dir.exists():
                bin_dir.mkdir(parents=True, exist_ok=True)
            if not new_path.exists():
                try:
                    old_path.rename(new_path)
                    moved_any = True
                except Exception:
                    pass
            else:
                try:
                    old_path.unlink()
                except Exception:
                    pass

    if moved_any:
        print("Migrated managed binaries tools/ → bin/")


# ---------------------------------------------------------------------------
# 弃用目录检查
# ---------------------------------------------------------------------------


def check_deprecated_extension_dirs(base_dir: Path, label: str) -> list[str]:
    """检查弃用的 ``hooks/`` 和 ``tools/`` 目录。

    Args:
        base_dir: 基础目录。
        label: 用于日志的标签（如 "Global" / "Project"）。

    Returns:
        警告列表。
    """
    warnings: list[str] = []
    hooks_dir = base_dir / "hooks"
    tools_dir = base_dir / "tools"

    if hooks_dir.exists():
        warnings.append(
            f"{label} hooks/ directory found. Hooks have been renamed to extensions."
        )

    if tools_dir.exists():
        try:
            entries = [p for p in tools_dir.iterdir()]
            custom_tools = [
                e.name
                for e in entries
                if e.name.lower() not in ("fd", "rg", "fd.exe", "rg.exe")
                and not e.name.startswith(".")
            ]
            if custom_tools:
                warnings.append(
                    f"{label} tools/ directory contains custom tools. "
                    "Custom tools have been merged into extensions."
                )
        except Exception:
            pass

    return warnings


# ---------------------------------------------------------------------------
# 扩展系统迁移
# ---------------------------------------------------------------------------


def migrate_extension_system(cwd: str) -> list[str]:
    """运行扩展系统迁移（commands → prompts）并收集弃用目录警告。

    Args:
        cwd: 当前工作目录。

    Returns:
        弃用警告列表。
    """
    from .config import CONFIG_DIR_NAME, get_agent_dir

    agent_dir = get_agent_dir()
    project_dir = Path(cwd) / CONFIG_DIR_NAME

    migrate_commands_to_prompts(agent_dir, "Global")
    migrate_commands_to_prompts(project_dir, "Project")

    warnings: list[str] = []
    warnings.extend(check_deprecated_extension_dirs(agent_dir, "Global"))
    warnings.extend(check_deprecated_extension_dirs(project_dir, "Project"))
    return warnings


# ---------------------------------------------------------------------------
# 弃用警告显示
# ---------------------------------------------------------------------------


def _wait_for_keypress() -> None:
    """等待用户按键。"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def show_deprecation_warnings(warnings: list[str]) -> None:
    """打印弃用警告并等待按键。

    Args:
        warnings: 弃用警告列表。
    """
    if not warnings:
        return

    for warning in warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    print(
        "\nMove your extensions to the extensions/ directory.",
        file=sys.stderr,
    )
    print(f"Migration guide: {MIGRATION_GUIDE_URL}", file=sys.stderr)
    print(f"Documentation: {EXTENSIONS_DOC_URL}", file=sys.stderr)
    print("Press any key to continue...", file=sys.stderr)

    _wait_for_keypress()
    print()


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def migrate_session() -> SessionMigration:
    """运行所有迁移。在启动时调用一次。

    Returns:
        迁移结果，包含已迁移的认证提供者和弃用警告。
    """
    migrated_auth_providers = migrate_auth_to_auth_json()
    migrate_sessions_from_agent_root()
    migrate_tools_to_bin()
    migrate_keybindings_config_file()

    cwd = os.getcwd()
    deprecation_warnings = migrate_extension_system(cwd)

    return SessionMigration(
        migrated_auth_providers=deprecation_warnings,
        deprecation_warnings=deprecation_warnings,
    )


__all__: list[str] = [
    "SessionMigration",
    "migrate_auth_to_auth_json",
    "migrate_session",
    "migrate_sessions_from_agent_root",
    "show_deprecation_warnings",
]
