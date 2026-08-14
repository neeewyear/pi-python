"""Pi Coding Agent 产品层。

依赖：``pi_ai``（LLM API 层）、``pi_agent``（通用 agent 引擎）、``pi_session``（会话数据层）。
"""

from __future__ import annotations

from .cli.args import (
    Args,
    Diagnostic,
    Mode,
    is_valid_thinking_level,
    parse_args,
    print_help,
)
from .migrations import (
    SessionMigration,
    migrate_auth_to_auth_json,
    migrate_session,
    migrate_sessions_from_agent_root,
    show_deprecation_warnings,
)
from .package_manager_cli import run_package_manager_cli
from .rpc_entry import run_rpc_entry

__all__: list[str] = [
    "Args",
    "Diagnostic",
    "Mode",
    "SessionMigration",
    "is_valid_thinking_level",
    "migrate_auth_to_auth_json",
    "migrate_session",
    "migrate_sessions_from_agent_root",
    "parse_args",
    "print_help",
    "run_package_manager_cli",
    "run_rpc_entry",
    "show_deprecation_warnings",
]
