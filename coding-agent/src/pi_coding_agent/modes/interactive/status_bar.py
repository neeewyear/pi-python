"""状态栏组件（Footer 状态栏）。

显示模型名称、会话 ID、处理状态、工具使用统计等信息。
替代默认的 ``Footer`` 和 ``#status-label`` Static 组件。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from textual.containers import Horizontal
from textual.widgets import Footer, Static

if TYPE_CHECKING:
    from textual.app import ComposeResult


class StatusBar(Footer):
    """自定义状态栏，显示模型/会话/处理状态/工具统计。

    用法：在 ``compose()`` 中 yield ``StatusBar()``，
    然后通过 ``update_status()`` 方法更新状态。
    """

    DEFAULT_CSS = """
    StatusBar {
        height: 1;
        padding: 0 1;
        background: $panel;
        color: $text-muted;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._model_name: str = ""
        self._session_id: str = ""
        self._status_text: str = "Ready"
        self._tools_text: str = ""

    def compose(self) -> ComposeResult:
        """组合状态栏内容。"""
        with Horizontal():
            yield Static(id="statusbar-left", expand=True)
            yield Static(id="statusbar-center")
            yield Static(id="statusbar-right", expand=True)

    def on_mount(self) -> None:
        """挂载后初始化。"""
        self._refresh()

    # ------------------------------------------------------------------
    # 公共更新方法
    # ------------------------------------------------------------------

    def set_model(self, name: str) -> None:
        """设置模型名称。"""
        self._model_name = name
        self._refresh()

    def set_session_id(self, sid: str) -> None:
        """设置会话 ID。"""
        if sid and len(sid) > 8:
            self._session_id = sid[:8]
        else:
            self._session_id = sid or ""
        self._refresh()

    def set_status(self, status: str) -> None:
        """设置状态文本（如 Ready / Processing...）。"""
        self._status_text = status
        self._refresh()

    def set_tools(self, running: int, total: int) -> None:
        """设置工具使用统计。"""
        if total > 0:
            self._tools_text = f"Tools: {running}r/{total}t"
        else:
            self._tools_text = ""
        self._refresh()

    # ------------------------------------------------------------------
    # 内部渲染
    # ------------------------------------------------------------------

    def _refresh(self) -> None:
        """渲染状态栏（更新各个 Static 子组件）。"""
        try:
            left = self.query_one("#statusbar-left", Static)
            center = self.query_one("#statusbar-center", Static)
            right = self.query_one("#statusbar-right", Static)
        except Exception:
            return  # 尚未挂载

        # 左侧：模型 + 会话
        left_parts: list[str] = []
        if self._model_name:
            left_parts.append(f"Model: {self._model_name}")
        if self._session_id:
            left_parts.append(f"Session: {self._session_id}")
        left.update(" | ".join(left_parts) if left_parts else "")

        # 中间：状态
        center.update(f"[bold]{self._status_text}[/]")

        # 右侧：工具统计
        right.update(f"[dim]{self._tools_text}[/]" if self._tools_text else "")
