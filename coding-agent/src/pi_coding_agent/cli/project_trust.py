"""CLI 项目信任（对应 TS ``cli/project-trust.ts``）。

提供 CLI 模式下的项目信任上下文工厂。
"""

from __future__ import annotations

import sys
from typing import Any, cast


def create_project_trust_context(
    *,
    cwd: str,
    mode: Any,
    settings_manager: Any = None,
    has_ui: bool = False,
) -> Any:
    """创建项目信任上下文。

    Args:
        cwd: 当前工作目录。
        mode: 应用模式。
        settings_manager: 设置管理器。
        has_ui: 是否具有 UI。

    Returns:
        ``ProjectTrustContext`` 实例。
    """
    from ..core.extensions.types import (
        ExtensionMode,
    )
    from ..core.extensions.types import (
        ProjectTrustContext as _ProjectTrustContext,
    )

    # 将 mode 转换为 ExtensionMode 兼容的值
    tui_mode: ExtensionMode
    if mode == "interactive":
        tui_mode = "tui"
    else:
        tui_mode = cast(ExtensionMode, str(mode))

    return _ProjectTrustContext(
        cwd=cwd,
        mode=tui_mode,
        has_ui=has_ui,
        ui=_create_ui_context(
            has_ui=has_ui,
            mode=mode,
            settings_manager=settings_manager,
        ),
    )


def _create_ui_context(
    *,
    has_ui: bool,
    mode: Any,
    settings_manager: Any = None,
) -> Any:
    """创建 UI 上下文。

    Args:
        has_ui: 是否具有 UI。
        mode: 应用模式。
        settings_manager: 设置管理器。

    Returns:
        ``ExtensionUIContext`` 实例或 None。
    """
    if not has_ui or mode != "interactive":
        return None

    from ..core.extensions.types import ExtensionUIContext as _ExtensionUIContext

    async def _select(
        title: str, options: list[str], dialog_options: Any = None
    ) -> str | None:
        _ = dialog_options
        if not has_ui or mode != "interactive":
            return None
        return _show_startup_selector(settings_manager, title, options)

    async def _confirm(title: str, message: str, dialog_options: Any = None) -> bool:
        _ = dialog_options
        if not has_ui or mode != "interactive":
            return False
        result = _show_startup_selector(
            settings_manager,
            f"{title}\n{message}",
            ["Yes", "No"],
        )
        return result == "Yes" if result else False

    async def _input(
        title: str, placeholder: str | None, dialog_options: Any = None
    ) -> str | None:
        _ = dialog_options
        if not has_ui or mode != "interactive":
            return None
        return _show_startup_input(settings_manager, title, placeholder)

    def _notify(message: str, type_: str | None = "info") -> None:
        if mode != "interactive":
            color = (
                "\033[31m"
                if type_ == "error"
                else "\033[33m"
                if type_ == "warning"
                else "\033[36m"
            )
            reset = "\033[0m"
            print(f"{color}{message}{reset}", file=sys.stderr)

    return _ExtensionUIContext(
        select=_select,
        confirm=_confirm,
        input=_input,
        notify=_notify,
        on_terminal_input=lambda handler: lambda: None,
        set_status=lambda text, status: None,
        set_working_message=lambda msg: None,
        set_working_visible=lambda visible: None,
        set_working_indicator=lambda options: None,
        set_hidden_thinking_label=lambda label: None,
        set_widget=lambda id_, component, options: None,
        set_footer=lambda footer: None,
        set_header=lambda header: None,
        set_title=lambda title: None,
        custom=cast(Any, lambda component, options: None),
        paste_to_editor=lambda text: None,
        set_editor_text=lambda text: None,
        get_editor_text=lambda: "",
        editor=cast(Any, lambda title, text: None),
        add_autocomplete_provider=lambda provider: None,
        set_editor_component=lambda factory: None,
        get_editor_component=lambda: None,
        theme=None,
        get_all_themes=list,
        get_theme=lambda name: None,
        set_theme=lambda theme: {"ok": True},
        get_tools_expanded=lambda: True,
        set_tools_expanded=lambda expanded: None,
    )


def _show_startup_selector(
    settings_manager: Any,
    title: str,
    options: list[str],
) -> str | None:
    """显示启动选择器（简化版）。

    Args:
        settings_manager: 设置管理器。
        title: 标题。
        options: 选项列表。

    Returns:
        选择的选项，取消返回 None。
    """
    _ = settings_manager
    print(f"\n{title}")
    for i, option in enumerate(options):
        print(f"  {i + 1}. {option}")
    try:
        choice = input("Enter number (or empty to cancel): ").strip()
        if choice and choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
    except (EOFError, KeyboardInterrupt):
        pass
    return None


def _show_startup_input(
    settings_manager: Any,
    title: str,
    placeholder: str | None,
) -> str | None:
    """显示启动输入（简化版）。

    Args:
        settings_manager: 设置管理器。
        title: 标题。
        placeholder: 占位符。

    Returns:
        输入的值，取消返回 None。
    """
    _ = settings_manager
    try:
        prompt = f"\n{title}"
        if placeholder:
            prompt += f" ({placeholder})"
        prompt += ": "
        value = input(prompt).strip()
        return value if value else None
    except (EOFError, KeyboardInterrupt):
        return None


__all__ = [
    "create_project_trust_context",
]
