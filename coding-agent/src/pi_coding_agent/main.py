"""主入口点（对应 TS ``main.ts``）。

处理 CLI 参数解析、创建 SessionManager → 运行时服务 → AgentSessionRuntime，
然后分派到对应模式（print / json / rpc / interactive）。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from .cli.args import Args, Diagnostic, Mode, parse_args, print_help
from .config import APP_NAME, VERSION, get_agent_dir, get_sessions_dir
from .core.agent_session_runtime import (
    CreateAgentSessionRuntimeResult,
    create_agent_session_runtime,
)
from .core.agent_session_services import (
    CreateAgentSessionFromServicesOptions,
    CreateAgentSessionServicesOptions,
    create_agent_session_from_services,
    create_agent_session_services,
)
from .core.http_dispatcher import apply_http_proxy_settings, configure_http_dispatcher
from .core.session_manager import NewSessionOptions, SessionManager
from .core.settings_manager import SettingsManager
from .core.timings import print_timings, reset_timings, time
from .modes import (
    PrintModeOptions,
    run_interactive_mode,
    run_print_mode,
    run_rpc_mode,
)

# ---------------------------------------------------------------------------
# AppMode 类型
# ---------------------------------------------------------------------------

AppMode = str
"""应用运行模式（interactive / print / json / rpc）。"""


# ---------------------------------------------------------------------------
# 内部辅助函数
# ---------------------------------------------------------------------------


def _is_truthy_env_flag(value: str | None) -> bool:
    if not value:
        return False
    return value in ("1", "true", "yes", "True", "Yes")


def _resolve_app_mode(parsed: Args, stdin_is_tty: bool, stdout_is_tty: bool) -> AppMode:
    if parsed.mode == "rpc":
        return "rpc"
    if parsed.mode == "json":
        return "json"
    if parsed.print or not stdin_is_tty or not stdout_is_tty:
        return "print"
    return "interactive"


def _to_print_output_mode(app_mode: AppMode) -> Mode:
    return "json" if app_mode == "json" else "text"


def _is_plain_runtime_metadata_command(parsed: Args) -> bool:
    return (
        not parsed.print
        and parsed.mode is None
        and (parsed.help is True or parsed.list_models is not None)
    )


def _report_diagnostics(diagnostics: list[Diagnostic]) -> None:
    for d in diagnostics:
        color = "red" if d.type == "error" else "yellow"
        prefix = "Error: " if d.type == "error" else "Warning: "
        print(f"{prefix}{d.message}", file=sys.stderr)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


class MainOptions:
    """主入口选项（对应 TS ``MainOptions``）。

    Attributes:
        extension_factories: 内联扩展工厂列表。
    """

    def __init__(self, extension_factories: list[object] | None = None) -> None:
        self.extension_factories = extension_factories or []


async def main(args: list[str], options: MainOptions | None = None) -> None:
    """主入口函数。

    Args:
        args: 命令行参数字符串列表（不含 ``sys.argv[0]``）。
        options: 可选的主入口选项。
    """
    reset_timings()

    offline_mode = "--offline" in args or _is_truthy_env_flag(
        os.environ.get("PI_OFFLINE")
    )
    if offline_mode:
        os.environ["PI_OFFLINE"] = "1"
        os.environ["PI_SKIP_VERSION_CHECK"] = "1"

    cwd = os.getcwd()
    agent_dir = get_agent_dir()

    # 初始化启动时的设置管理器
    bootstrap_settings_manager = SettingsManager.create(cwd, str(agent_dir))
    apply_http_proxy_settings(
        bootstrap_settings_manager.get_global_settings().http_proxy
    )
    configure_http_dispatcher()

    # 解析 CLI 参数
    parsed = parse_args(args)
    if parsed.diagnostics:
        _report_diagnostics(parsed.diagnostics)
        if any(d.type == "error" for d in parsed.diagnostics):
            sys.exit(1)
    time("parseArgs")

    # --version
    if parsed.version:
        print(VERSION)
        sys.exit(0)

    # --help
    if parsed.help:
        print_help()
        sys.exit(0)

    # 确定运行模式
    app_mode = _resolve_app_mode(parsed, sys.stdin.isatty(), sys.stdout.isatty())
    print(f"{APP_NAME} v{VERSION} - AI coding assistant", file=sys.stderr)
    print(f"Mode: {app_mode}", file=sys.stderr)

    # ------------------------------------------------------------------
    # 第一步：创建 SessionManager
    # ------------------------------------------------------------------
    session_dir = (
        str(Path(parsed.session_dir).resolve()) if parsed.session_dir else None
    )

    if parsed.session:
        # 指定会话文件
        session_path = str(Path(parsed.session).expanduser().resolve())
        session_manager = SessionManager.open(session_path, cwd if cwd else None)
    elif parsed.resume or parsed.continue_:
        # 恢复最近会话 — 扫描会话目录
        sessions_base = session_dir or str(get_sessions_dir())
        if not os.path.isdir(sessions_base):
            session_manager = _create_default_session_manager(cwd, session_dir, parsed)
        else:
            latest_session_path = _find_latest_session(sessions_base)
            if latest_session_path:
                session_manager = SessionManager.open(latest_session_path, cwd)
            else:
                session_manager = _create_default_session_manager(
                    cwd, session_dir, parsed
                )
    elif parsed.fork:
        # 分叉指定会话
        fork_path = str(Path(parsed.fork).expanduser().resolve())
        fork_mgr = SessionManager.open(fork_path, cwd)
        fork_leaf = fork_mgr.get_leaf_id()
        if fork_leaf is None:
            msg = f"Session {parsed.fork} has no entries to fork from"
            raise RuntimeError(msg)
        persisted = fork_mgr.create_branched_session(fork_leaf)
        if persisted:
            session_manager = SessionManager.open(persisted, cwd)
        else:
            session_manager = _create_default_session_manager(cwd, session_dir, parsed)
    elif parsed.no_session:
        session_manager = SessionManager.in_memory(cwd)
    else:
        session_manager = _create_default_session_manager(cwd, session_dir, parsed)

    time("createSessionManager")

    # 注意：--name 设置会话名称的功能暂未实现
    # 需要 SessionManager 支持 label 方法

    # ------------------------------------------------------------------
    # 第二步：创建运行时工厂
    # ------------------------------------------------------------------
    async def _create_runtime_factory(
        *,
        cwd: str,
        agent_dir: str,
        session_manager: SessionManager,
        session_start_event: Any = None,
        project_trust_context: Any = None,
    ) -> CreateAgentSessionRuntimeResult:
        """内部运行时工厂（对应 TS ``createRuntime``）。"""
        # 创建 cwd 绑定的运行时服务
        services = await create_agent_session_services(
            CreateAgentSessionServicesOptions(
                cwd=cwd,
                agent_dir=agent_dir,
                settings_manager=bootstrap_settings_manager,
                model_runtime=None,
                extension_flag_values=parsed.unknown_flags,
            )
        )
        # 从服务创建 AgentSession
        session_result = await create_agent_session_from_services(
            CreateAgentSessionFromServicesOptions(
                services=services,
                session_manager=session_manager,
                model=parsed.model,
                thinking_level=parsed.thinking if parsed.thinking else None,
                tools=parsed.tools,
                exclude_tools=parsed.exclude_tools,
                no_tools="all"
                if parsed.no_tools
                else ("builtin" if parsed.no_builtin_tools else None),
                session_start_event=session_start_event
                or {
                    "type": "session_start",
                    "reason": "startup",
                },
            )
        )
        session = session_result.session
        return CreateAgentSessionRuntimeResult(
            session=session,
            services=services,
            diagnostics=services.diagnostics,
            model_fallback_message=session_result.model_fallback_message,
        )

    # ------------------------------------------------------------------
    # 第三步：创建 AgentSessionRuntime
    # ------------------------------------------------------------------
    runtime = await create_agent_session_runtime(
        _create_runtime_factory,
        {
            "cwd": cwd,
            "agent_dir": str(agent_dir),
            "session_manager": session_manager,
            "session_start_event": {
                "type": "session_start",
                "reason": "startup",
            },
        },
    )
    time("createRuntime")

    # ------------------------------------------------------------------
    # 第四步：分派到模式
    # ------------------------------------------------------------------
    try:
        if app_mode == "rpc":
            await run_rpc_mode(runtime)
        elif app_mode in ("print", "json"):
            print_mode = _to_print_output_mode(app_mode)
            initial_message = _build_initial_message(parsed)
            exit_code = await run_print_mode(
                runtime,
                PrintModeOptions(
                    mode=print_mode,
                    messages=parsed.messages,
                    initial_message=initial_message,
                    initial_images=None,
                ),
            )
            sys.exit(exit_code)
        else:
            # interactive 模式 — Textual TUI
            exit_code = await run_interactive_mode(runtime)
            sys.exit(exit_code)
    finally:
        print_timings()


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _create_default_session_manager(
    cwd: str, session_dir: str | None, parsed: Args
) -> SessionManager:
    """创建默认会话管理器（新会话）。"""
    return SessionManager.create(
        cwd,
        session_dir,
        NewSessionOptions(id=parsed.session_id) if parsed.session_id else None,
    )


def _find_latest_session(sessions_base: str) -> str | None:
    """在会话目录中找到最新的 .jsonl 会话文件。"""
    base = Path(sessions_base)
    if not base.is_dir():
        return None
    jsonl_files = sorted(base.rglob("*.jsonl"), key=lambda p: p.stat().st_mtime)
    return str(jsonl_files[-1]) if jsonl_files else None


def _build_initial_message(parsed: Args) -> str | None:
    """从 CLI 参数构建初始消息。"""
    if parsed.messages:
        return parsed.messages[0]
    if parsed.print:
        return None
    return None
