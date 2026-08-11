"""会话管理 UI（TUI 弹窗）。

提供会话选择、切换、新建等操作的 Textual Screen 组件。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import Binding, BindingType
from textual.screen import ModalScreen
from textual.widgets import Label, ListItem, ListView, Static

from pi_coding_agent.config import get_sessions_dir

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime

# ---------------------------------------------------------------------------
# 会话信息
# ---------------------------------------------------------------------------


@dataclass
class SessionInfo:
    """会话文件元信息。"""

    path: str
    """会话文件绝对路径。"""
    file_name: str
    """文件名（如 ``20260304_abc123.jsonl``）。"""
    timestamp: str = ""
    """时间戳部分（如 ``20260304_abc123`` 截取前 8 位）。"""
    summary: str = ""
    """会话摘要（取自文件头部，如有）。"""


# ---------------------------------------------------------------------------
# 会话选择器 Screen
# ---------------------------------------------------------------------------


class SessionPickerScreen(ModalScreen[str | None]):
    """会话选择器弹窗（TUI 版）。

    列出可用会话文件，支持：
    - 上下箭头选择
    - Enter 切换会话
    - Ctrl+N 新建会话
    - Escape 取消
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        Binding("up", "cursor_up", "Up", show=False),
        Binding("down", "cursor_down", "Down", show=False),
        Binding("enter", "select", "Select"),
        Binding("ctrl+n", "new_session", "New Session"),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

    DEFAULT_CSS = """
    SessionPickerScreen {
        align: center middle;
    }

    #session-picker-box {
        width: 60;
        height: 70%;
        min-height: 10;
        border: thick $primary;
        background: $surface;
    }

    #session-picker-title {
        padding: 1;
        text-style: bold;
        background: $primary;
        color: $text;
    }

    #session-picker-hint {
        padding: 0 1;
        color: $text-muted;
        height: 1;
    }

    #session-list {
        height: 1fr;
    }

    #session-picker-footer {
        padding: 0 1;
        height: 1;
        color: $text-muted;
        background: $panel;
    }
    """

    def __init__(self, runtime_host: AgentSessionRuntime) -> None:
        super().__init__()
        self._runtime_host = runtime_host
        self._sessions: list[SessionInfo] = []
        self._session_dir: str = ""

    def compose(self) -> ComposeResult:
        """组合会话选择器 UI。"""
        with Static(id="session-picker-box"):
            yield Static("[bold]Session Picker[/]", id="session-picker-title")
            yield Static(
                "Select a session or create a new one.",
                id="session-picker-hint",
            )
            yield ListView(id="session-list")
            yield Static(
                "Enter: switch | Ctrl+N: new | Esc: cancel",
                id="session-picker-footer",
            )

    async def on_mount(self) -> None:
        """挂载后加载会话列表。"""
        self._session_dir = str(get_sessions_dir())
        self._load_sessions()
        self._render_sessions()

    # ------------------------------------------------------------------
    # 会话加载
    # ------------------------------------------------------------------

    def _load_sessions(self) -> None:
        """扫描会话目录，加载会话文件列表。"""
        self._sessions.clear()
        session_dir = Path(self._session_dir)
        if not session_dir.is_dir():
            return

        # 扫描 .jsonl 文件，按修改时间倒序排列
        jsonl_files: list[Path] = sorted(
            session_dir.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )

        for f in jsonl_files:
            name = f.name
            # 从文件名提取时间戳（前 8 位日期）
            ts = name[:8] if len(name) >= 8 and name[:8].isdigit() else ""
            # 读取文件头部作为摘要（首行前 80 字符）
            summary = self._read_summary(f)
            self._sessions.append(
                SessionInfo(
                    path=str(f),
                    file_name=name,
                    timestamp=ts,
                    summary=summary,
                )
            )

    def _read_summary(self, path: Path) -> str:
        """读取会话文件首行作为摘要。"""
        try:
            first_line = path.read_text(encoding="utf-8", errors="replace").split("\n")[
                0
            ]
            # 提取摘要（去掉 JSON 结构）
            if len(first_line) > 120:
                first_line = first_line[:120] + "..."
            return first_line.strip()
        except (OSError, UnicodeDecodeError):
            return ""

    def _render_sessions(self) -> None:
        """将会话列表渲染到 ListView。"""
        list_view = self.query_one("#session-list", ListView)
        list_view.clear()

        if not self._sessions:
            list_view.append(ListItem(Label("[dim]No sessions found[/]")))
            return

        for info in self._sessions:
            # 显示：日期 + 文件名
            label_parts = []
            if info.timestamp:
                label_parts.append(f"[bold]{info.timestamp}[/]")
            label_parts.append(f"[dim]{info.file_name}[/]")
            if info.summary:
                label_parts.append(f"[dim]{info.summary[:60]}[/]")
            label = Label("  ".join(label_parts))
            list_view.append(ListItem(label))

    # ------------------------------------------------------------------
    # 动作
    # ------------------------------------------------------------------

    async def action_select(self) -> None:
        """选择当前高亮会话并切换。"""
        list_view = self.query_one("#session-list", ListView)
        if list_view.index is None:
            return
        idx = list_view.index
        if idx < 0 or idx >= len(self._sessions):
            return
        selected = self._sessions[idx]
        self.dismiss(selected.path)

    async def action_new_session(self) -> None:
        """新建会话并返回新建标记。"""
        self.dismiss("__new__")

    async def action_cancel(self) -> None:
        """取消选择。"""
        self.dismiss(None)

    async def action_cursor_up(self) -> None:
        """上移光标。"""
        list_view = self.query_one("#session-list", ListView)
        if list_view.index is not None and list_view.index > 0:
            list_view.index = list_view.index - 1

    async def action_cursor_down(self) -> None:
        """下移光标。"""
        list_view = self.query_one("#session-list", ListView)
        max_idx = len(self._sessions) - 1
        if list_view.index is None:
            list_view.index = 0
        elif list_view.index < max_idx:
            list_view.index = list_view.index + 1


# ---------------------------------------------------------------------------
# 便捷函数
# ---------------------------------------------------------------------------


async def show_session_picker(
    runtime_host: AgentSessionRuntime,
) -> str | None:
    """弹出会话选择器并等待用户选择。

    Args:
        runtime_host: AgentSessionRuntime 实例。

    Returns:
        选择的会话路径，``__new__`` 表示新建，None 表示取消。
    """
    from .app import InteractiveModeApp

    app = InteractiveModeApp(runtime_host)
    result = await app.push_screen_wait(SessionPickerScreen(runtime_host))
    return result
