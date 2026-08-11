"""应用配置与路径工具（对应 TS ``config.ts`` 的简化版）。

提供应用名称、版本号以及用户配置目录（``~/.pi/agent/``）下各子路径的
获取函数。所有路径使用 ``pathlib.Path`` 表示。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# 应用常量
# ---------------------------------------------------------------------------

CONFIG_DIR_NAME = ".pi"
"""用户配置目录名。"""

APP_NAME = "pi"
"""应用名称。"""

APP_TITLE = "π"
"""应用标题。"""

PACKAGE_NAME = "@earendil-works/pi-coding-agent"
"""包名称。"""

VERSION = "0.1.0"
"""应用版本号。"""

# ---------------------------------------------------------------------------
# 环境变量名
# ---------------------------------------------------------------------------

ENV_AGENT_DIR = f"{APP_NAME.upper()}_CODING_AGENT_DIR"
"""环境变量名，用于覆盖 agent 配置目录。"""

ENV_SESSION_DIR = f"{APP_NAME.upper()}_CODING_AGENT_SESSION_DIR"
"""环境变量名，用于覆盖 session 存储目录。"""

_DEFAULT_SHARE_VIEWER_URL = "https://pi.dev/session/"


# ---------------------------------------------------------------------------
# 路径工具
# ---------------------------------------------------------------------------


def get_agent_dir() -> Path:
    """获取 agent 配置目录。

    优先级：
    1. ``<APP_NAME>_CODING_AGENT_DIR`` 环境变量。
    2. 默认 ``~/.pi/agent/``。

    Returns:
        Agent 配置目录的绝对路径。
    """
    env_dir = os.environ.get(ENV_AGENT_DIR)
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / CONFIG_DIR_NAME / "agent"


def get_models_path() -> Path:
    """获取模型配置路径（``models.json``）。"""
    return get_agent_dir() / "models.json"


def get_auth_path() -> Path:
    """获取认证配置路径（``auth.json``）。"""
    return get_agent_dir() / "auth.json"


def get_settings_path() -> Path:
    """获取设置配置路径（``settings.json``）。"""
    return get_agent_dir() / "settings.json"


def get_sessions_dir() -> Path:
    """获取会话存储目录。"""
    return get_agent_dir() / "sessions"


def get_prompts_dir() -> Path:
    """获取提示词模板目录。"""
    return get_agent_dir() / "prompts"


def get_tools_dir() -> Path:
    """获取工具配置目录。"""
    return get_agent_dir() / "tools"


def get_bin_dir() -> Path:
    """获取托管二进制文件目录（``fd``、``rg``）。"""
    return get_agent_dir() / "bin"


def get_custom_themes_dir() -> Path:
    """获取用户自定义主题目录。"""
    return get_agent_dir() / "themes"


def get_debug_log_path() -> Path:
    """获取调试日志文件路径。"""
    return get_agent_dir() / f"{APP_NAME}-debug.log"


def get_package_dir() -> Path:
    """获取 pi-coding-agent 包根目录（包含 pyproject.toml 的目录）。"""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_package_json_path() -> Path:
    """获取 ``package.json`` 路径。"""
    return get_package_dir() / "package.json"


def get_readme_path() -> Path:
    """获取 README.md 路径。"""
    return get_package_dir() / "README.md"


def get_docs_path() -> Path:
    """获取 docs 目录路径。"""
    return get_package_dir() / "docs"


def get_examples_path() -> Path:
    """获取 examples 目录路径。"""
    return get_package_dir() / "examples"


def get_changelog_path() -> Path:
    """获取 CHANGELOG.md 路径。"""
    return get_package_dir() / "CHANGELOG.md"


def get_themes_dir() -> Path:
    """获取内置主题目录路径。

    Python 包中主题位于 ``modes/interactive/theme/``。
    """
    return (
        get_package_dir()
        / "src"
        / "pi_coding_agent"
        / "modes"
        / "interactive"
        / "theme"
    )


def get_export_template_dir() -> Path:
    """获取 HTML 导出模板目录路径。

    Python 包中模板位于 ``core/export_html/``。
    """
    return get_package_dir() / "src" / "pi_coding_agent" / "core" / "export_html"


def get_interactive_assets_dir() -> Path:
    """获取内置交互式资源目录路径。"""
    return (
        get_package_dir()
        / "src"
        / "pi_coding_agent"
        / "modes"
        / "interactive"
        / "assets"
    )


def get_bundled_interactive_asset_path(name: str) -> Path:
    """获取打包的交互式资源文件路径。

    Args:
        name: 资源文件名。

    Returns:
        资源文件的完整路径。
    """
    return get_interactive_assets_dir() / name


def get_share_viewer_url(gist_id: str) -> str:
    """获取分享查看器 URL。

    Args:
        gist_id: Gist ID。

    Returns:
        完整的分享查看器 URL。
    """
    base_url = os.environ.get("PI_SHARE_VIEWER_URL", _DEFAULT_SHARE_VIEWER_URL)
    return f"{base_url}#{gist_id}"
