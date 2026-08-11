"""TUI 内建命令处理器。

处理交互模式特有的 `/` 命令，与扩展命令（ExtensionRunner）互补。
非 TUI 内建命令会被透传到 `session.prompt()` 交由扩展命令处理。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Awaitable


# 内建命令定义
BUILTIN_COMMANDS: dict[str, tuple[str, str]] = {
    "/help": ("显示帮助信息", "Usage: /help [/command]"),
    "/clear": ("清屏", "Usage: /clear"),
    "/exit": ("退出应用", "Usage: /exit"),
    "/quit": ("退出应用", "Usage: /quit"),
    "/new": ("新建会话", "Usage: /new"),
    "/history": ("显示命令历史", "Usage: /history"),
    "/save": ("保存会话到文件", "Usage: /save [filename]"),
}

# 需要交互模式自行处理，不应透传到 session.prompt() 的 TUI 命令
TUI_HANDLED_COMMANDS = frozenset(
    {"help", "clear", "exit", "quit", "new", "history", "save"}
)


def get_builtin_command_names() -> list[str]:
    """获取所有内建命令名列表（不含 / 前缀，按字母排序）。"""
    return sorted(TUI_HANDLED_COMMANDS)


# 命令处理函数签名
CommandHandler = Callable[[str], "Awaitable[bool]"]
"""处理函数接收参数字符串，返回 True 表示已处理。"""


def is_tui_command(text: str) -> bool:
    """判断是否为 TUI 内建命令。"""
    command_name = _extract_command_name(text)
    return command_name in TUI_HANDLED_COMMANDS


def is_extension_command(text: str) -> bool:
    """判断是否为扩展命令（以 / 开头但不是 TUI 内建）。"""
    command_name = _extract_command_name(text)
    return bool(command_name) and command_name not in TUI_HANDLED_COMMANDS


def _extract_command_name(text: str) -> str | None:
    """从文本中提取命令名（不含 / 前缀）。"""
    text = text.strip()
    if not text or not text.startswith("/"):
        return None
    space_index = text.find(" ")
    if space_index != -1:
        return text[1:space_index]
    return text[1:]


def format_help_text(
    extra_commands: list[dict[str, str]] | None = None,
) -> str:
    """格式化帮助文本。"""
    lines = ["[bold cyan]Available Commands[/]", ""]
    # TUI 内建命令
    lines.append("[bold]TUI Commands:[/]")
    for name, (desc, _) in sorted(BUILTIN_COMMANDS.items()):
        lines.append(f"  [green]{name:<15}[/] {desc}")
    # 扩展命令
    if extra_commands:
        lines.append("")
        lines.append("[bold]Extension Commands:[/]")
        for cmd in extra_commands:
            name = cmd.get("invocation_name", cmd.get("name", "?"))
            desc = cmd.get("description", "")
            lines.append(f"  [yellow]/{name:<15}[/] {desc}")
    lines.append("")
    lines.append("[dim]Type your message and press Ctrl+D to submit.[/]")
    return "\n".join(lines)
