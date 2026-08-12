"""Textual TUI 主应用（交互模式）。

使用 Textual 框架实现终端交互式对话界面。
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from pi_agent.types import AgentEvent, CancellationToken
from pi_ai.utils.text import content_text
from textual.app import App, ComposeResult
from textual.binding import Binding, BindingType
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    Header,
    RichLog,
    Static,
    TextArea,
)

from .commands import (
    format_help_text,
    is_tui_command,
)
from .completer import TabCompleter
from .history import CommandHistory
from .output_renderer import OutputRenderer
from .session_ui import SessionPickerScreen
from .signal_handler import register_signal_handlers, restore_signal_handlers
from .status_bar import StatusBar

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime


class InteractiveModeApp(App[None]):
    """交互式 TUI 应用主类。

    Layout:
        ┌─────────────────────────────────────┐
        │  Header (title + model/session)     │
        ├─────────────────┬───────────────────┤
        │  Output Panel   │  Info Panel       │
        │  (RichLog)      │  (Static)         │
        │                 │                   │
        ├─────────────────┴───────────────────┤
        │  Input Area (TextArea)              │
        ├─────────────────────────────────────┤
        │  Footer (keybindings)               │
        └─────────────────────────────────────┘
    """

    # 响应式状态
    session_id = reactive("")
    model_name = reactive("")
    is_processing = reactive(False)

    BINDINGS: list[BindingType] = cast(  # type: ignore[misc]
        "list[BindingType]",
        [
            Binding("ctrl+c", "cancel", "Cancel", priority=True),
            Binding("ctrl+d", "submit", "Submit", priority=True),
            Binding("ctrl+q", "quit", "Quit"),
            Binding("ctrl+n", "new_session", "New Session"),
            Binding("ctrl+o", "pick_session", "Open Session"),
            Binding("ctrl+l", "clear", "Clear"),
            Binding("up", "history_up", "History Up", show=False),
            Binding("down", "history_down", "History Down", show=False),
            Binding("tab", "tab_complete", "Complete", show=False),
            Binding("escape", "focus_input", "Input"),
        ],
    )

    CSS = """
    Screen {
        background: $surface;
    }

    #main-container {
        height: 100%;
    }

    #output-panel {
        border: solid $primary;
        height: 1fr;
        min-height: 10;
    }

    #output-log {
        height: 100%;
    }

    #info-panel {
        border: solid $primary;
        width: 30;
        min-width: 20;
        height: 1fr;
    }

    #info-text {
        height: 100%;
        padding: 1;
    }

    #input-textarea {
        border: solid $secondary;
        height: 8;
        min-height: 3;
        max-height: 12;
    }
    """

    def __init__(self, runtime_host: AgentSessionRuntime) -> None:
        super().__init__()
        self._runtime_host = runtime_host
        self._session = runtime_host.session
        self._event_listener: Callable[[], None] | None = None
        self._pending_tool_calls: dict[str, dict[str, object]] = {}
        # 命令历史（持久化到 ~/.pi/agent/history.json）
        history_path = Path.home() / ".pi" / "agent" / "history.json"
        self._history = CommandHistory(file_path=history_path)
        # 缓存导航前输入，用于下箭头恢复
        self._history_stash: str = ""
        # Tab 补全器
        self._completer = TabCompleter(
            get_extension_commands=self._get_extension_commands,
        )
        # 消息缓冲区：在流式传输期间累积文本 delta，用于 message_end 时渲染语法高亮
        self._message_buffer: list[str] = []
        # 输出渲染器：代码块检测 + 语法高亮
        self._message_renderer = OutputRenderer()

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="main-container"):
            with Vertical(id="output-panel"):
                yield RichLog(
                    id="output-log",
                    highlight=True,
                    markup=True,
                    max_lines=10_000,
                )
            with Vertical(id="info-panel"):
                yield Static(id="info-text")
        yield TextArea(
            id="input-textarea",
            text="",
            language=None,
            soft_wrap=True,
        )
        yield StatusBar()

    def on_mount(self) -> None:
        """应用启动时初始化。"""
        self.title = "Pi Coding Agent"
        self.sub_title = "Interactive Mode"
        # 初始化状态栏
        self._update_status_bar()
        self._update_info_panel()
        self._subscribe_to_events()
        # 输出欢迎信息
        output_log = self.query_one("#output-log", RichLog)
        output_log.write("[bold cyan]Pi Coding Agent v1.0[/]")
        output_log.write(
            "Type your message and press Ctrl+D to submit. "
            "Use /help for available commands."
        )
        output_log.write("")

    def on_unmount(self) -> None:
        """应用关闭时清理。"""
        self._unsubscribe_from_events()

    # ------------------------------------------------------------------
    # 事件订阅
    # ------------------------------------------------------------------

    def _subscribe_to_events(self) -> None:
        """订阅 Agent 事件（流式输出、工具调用等）。"""
        self._event_listener = self._session.agent.subscribe(self._on_agent_event)

    def _unsubscribe_from_events(self) -> None:
        """取消订阅事件。"""
        if self._event_listener is not None:
            self._event_listener()
            self._event_listener = None

    def _on_agent_event(
        self, event: AgentEvent, token: CancellationToken | None = None
    ) -> None:
        """处理 Agent 事件（同步回调 -> 异步 UI 更新）。

        注意：Agent 回调运行在 UI 的同一事件循环线程中，
        因此直接调用 _handle_event 即可，无需 call_from_thread。
        """
        self._handle_event(event)

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _handle_event(self, event: object) -> None:
        """在 UI 线程中处理事件（同步，被 call_from_thread 调用）。"""
        event_type = getattr(event, "type", str(type(event)))
        output_log = self.query_one("#output-log", RichLog)

        if event_type == "message_update":
            self._handle_message_update(event, output_log)
        elif event_type == "message_start":
            self._handle_message_start(event, output_log)
        elif event_type == "message_end":
            self._handle_message_end(event, output_log)
        elif event_type == "tool_execution_start":
            self._handle_tool_start(event, output_log)
        elif event_type == "tool_execution_update":
            self._handle_tool_update(event, output_log)
        elif event_type == "tool_execution_end":
            self._handle_tool_end(event, output_log)
        elif event_type == "agent_session_agent_settled":
            self.is_processing = False
            self._update_status_bar()

    def _handle_message_start(self, event: object, log: RichLog) -> None:
        """消息开始。"""
        self.is_processing = True
        self._message_buffer.clear()
        self._update_status_bar()

    def _handle_message_update(self, event: object, log: RichLog) -> None:
        """消息更新（含流式文本 delta）。"""
        assistant_event = getattr(event, "assistant_message_event", None)
        if assistant_event is None:
            return
        event_type = getattr(assistant_event, "type", "")
        if event_type == "text_delta":
            delta = getattr(assistant_event, "delta", "")
            if delta:
                self._message_buffer.append(delta)
        elif event_type == "thinking_delta":
            delta = getattr(assistant_event, "delta", "")
            if delta:
                log.write(f"[dim]{delta}[/]")

    def _handle_message_end(self, event: object, log: RichLog) -> None:
        """消息结束 — 渲染完整消息（含代码块语法高亮）。"""
        full_text = "".join(self._message_buffer)
        # 如果缓冲区为空（非流式响应），从 event.message 提取完整文本
        error_message: str | None = None
        if not full_text.strip():
            msg = getattr(event, "message", None)
            if msg is not None:
                content = getattr(msg, "content", None)
                with open("message_end.txt", "a") as f:
                    f.write(
                        f"[dim]debug: message_end msg={msg!r}, content={content!r}[/]\n"
                    )
                if content is not None:
                    full_text = content_text(content, "")
                # 检查 error_message 属性（助理消息失败时使用）
                if not full_text.strip():
                    error_message = getattr(msg, "error_message", None)
        if full_text.strip():
            for item in self._message_renderer.render_message(full_text):
                log.write(item)
        elif error_message:
            log.write(f"[bold red]Error: {error_message}[/]")
        self._message_buffer.clear()
        self.is_processing = False
        self._update_info_panel()
        self._update_status_bar()

    def _handle_tool_start(self, event: object, log: RichLog) -> None:
        """工具调用开始。"""
        tool_name = getattr(event, "tool_name", "unknown")
        tool_call_id = getattr(event, "tool_call_id", "")
        log.write(f"[bold yellow]🔧 {tool_name}[/] [dim]({tool_call_id})[/]")
        self._pending_tool_calls[tool_call_id] = {
            "name": tool_name,
            "status": "running",
        }
        self._update_status_bar()

    def _handle_tool_update(self, event: object, log: RichLog) -> None:
        """工具调用更新。"""
        tool_name = getattr(event, "tool_name", "unknown")
        partial = getattr(event, "partial_result", None)
        if partial is not None:
            log.write(f"  [dim]{partial}[/]")

    def _handle_tool_end(self, event: object, log: RichLog) -> None:
        """工具调用结束。"""
        tool_call_id = getattr(event, "tool_call_id", "")
        result = getattr(event, "result", None)
        is_error = getattr(event, "is_error", False)
        if tool_call_id in self._pending_tool_calls:
            self._pending_tool_calls[tool_call_id]["status"] = (
                "error" if is_error else "done"
            )
        if result is not None:
            result_str = str(result)
            if len(result_str) > 200:
                result_str = result_str[:200] + "..."
            log.write(f"  [dim]{'❌' if is_error else '✅'} {result_str}[/]")
        self._update_status_bar()

    # ------------------------------------------------------------------
    # UI 更新
    # ------------------------------------------------------------------

    def _update_info_panel(self) -> None:
        """更新信息面板。"""
        info = self.query_one("#info-text", Static)
        text = ""
        if self._session:
            meta = self._session.session_manager
            session_id = meta.get_session_id()
            if session_id:
                text += f"[bold]Session:[/] {session_id[:8]}...\n"
            else:
                text += "[bold]Session:[/] (in-memory)\n"
            cwd = getattr(meta, "cwd", None)
            if cwd:
                text += f"[bold]CWD:[/] {cwd}\n"
        if self.model_name:
            text += f"[bold]Model:[/] {self.model_name}\n"
        if not text:
            text = "[dim]No session info[/]"
        info.update(text)

    def _update_status_bar(self) -> None:
        """更新状态栏。"""
        try:
            status_bar = self.query_one(StatusBar)
        except Exception:
            return

        if self.is_processing:
            buf_len = sum(len(s) for s in self._message_buffer)
            status_bar.set_status(f"Processing... ({buf_len} chars)")
        else:
            status_bar.set_status("Ready")

        tool_count = len(self._pending_tool_calls)
        if tool_count > 0:
            running = sum(
                1
                for t in self._pending_tool_calls.values()
                if t.get("status") == "running"
            )
            status_bar.set_tools(running, tool_count)
        else:
            status_bar.set_tools(0, 0)

        status_bar.set_model(self.model_name)

    # ------------------------------------------------------------------
    # 输入 & 命令
    # ------------------------------------------------------------------

    def _get_input_text(self) -> str:
        """获取当前输入框文本。"""
        return self.query_one("#input-textarea", TextArea).text

    def _set_input_text(self, text: str) -> None:
        """设置输入框文本。"""
        text_area = self.query_one("#input-textarea", TextArea)
        text_area.text = text

    # ------------------------------------------------------------------
    # Tab 补全
    # ------------------------------------------------------------------

    def _get_extension_commands(self) -> list[str]:
        """获取扩展命令列表（供 Tab 补全器使用）。"""
        try:
            if self._session._extension_runner:
                registered = self._session._extension_runner.get_registered_commands()
                return [
                    f"/{c.invocation_name}" for c in registered if c.invocation_name
                ]
        except Exception:
            pass
        return []

    async def action_tab_complete(self) -> None:
        """Tab 补全（命令补全循环）。"""
        if self.is_processing:
            return
        text_area = self.query_one("#input-textarea", TextArea)
        text = text_area.text
        cursor_pos = len(text)  # 从末尾补全
        result, start, end = self._completer.complete(text, cursor_pos)
        if result is not None:
            text_area.text = result
            # 光标移到补全后末尾
            text_area.cursor = (  # type: ignore[attr-defined]
                len(result.split("\n")) - 1,
                len(result.split("\n")[-1]),
            )
        # 如果没有匹配，不做任何操作（保留 Tab 原有行为）

    async def _handle_tui_command(self, text: str) -> bool:
        """处理 TUI 内建命令。返回 True 表示已处理。"""
        output_log = self.query_one("#output-log", RichLog)
        text_stripped = text.strip()

        if text_stripped == "/help":
            # 尝试获取扩展命令列表
            extra_commands = None
            try:
                if self._session._extension_runner:
                    registered = (
                        self._session._extension_runner.get_registered_commands()
                    )
                    extra_commands = [
                        {
                            "invocation_name": c.invocation_name,
                            "description": c.description or "",
                        }
                        for c in registered
                    ]
            except Exception:
                pass
            output_log.write(format_help_text(extra_commands))
            return True

        if text_stripped in ("/exit", "/quit"):
            output_log.write("[bold cyan]Goodbye![/]")
            self.exit(return_code=0)
            return True

        if text_stripped.startswith("/save"):
            # /save [filename] — 将会话对话保存到文件
            self._save_session(text_stripped, output_log)
            return True

        if text_stripped == "/clear":
            output_log.clear()
            return True

        if text_stripped == "/new":
            await self.action_new_session()
            return True

        if text_stripped == "/history":
            all_entries = self._history.all()
            if not all_entries:
                output_log.write("[dim]No command history.[/]")
            else:
                lines = ["[bold cyan]Command History:[/]"]
                for i, entry in enumerate(all_entries[-50:], 1):  # 最多显示 50 条
                    lines.append(f"  {i:3d}. {entry}")
                output_log.write("\n".join(lines))
            return True

        return False

    def _format_message_for_save(self, msg: Any) -> str | None:
        """将单条消息格式化为 Markdown 文本。

        Args:
            msg: AgentMessage 对象。

        Returns:
            Markdown 格式的文本，如果消息无内容则返回 None。
        """
        role = getattr(msg, "role", "unknown")

        if role == "user":
            content = getattr(msg, "content", [])
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                    elif isinstance(block, dict):
                        text += f"[{block.get('type', 'unknown')} content]\n"
            elif isinstance(content, str):
                text = content
            if not text.strip():
                return None
            return f"## User\n\n{text.strip()}\n"

        if role == "assistant":
            content = getattr(msg, "content", [])
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                    elif isinstance(block, dict):
                        text += f"[{block.get('type', 'unknown')} content]\n"
            elif isinstance(content, str):
                text = content
            if not text.strip():
                return None
            return f"## Assistant\n\n{text.strip()}\n"

        if role == "tool_result" or role == "toolResult":
            content = getattr(msg, "content", [])
            text = ""
            if isinstance(content, list):
                for block in content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        text += block.get("text", "")
                    elif isinstance(block, dict):
                        text += f"[{block.get('type', 'unknown')} content]\n"
            elif isinstance(content, str):
                text = content
            tool_name = getattr(msg, "tool_name", "tool")
            if not text.strip():
                return None
            return f"### Tool Result ({tool_name})\n\n```\n{text.strip()}\n```\n"

        return None

    def _save_session(self, text: str, output_log: RichLog) -> None:
        """将会话对话保存到 Markdown 文件。

        Args:
            text: 用户输入的命令（可能包含文件名）。
            output_log: 输出日志组件。
        """
        # 解析文件名
        parts = text.split(maxsplit=1)
        if len(parts) > 1:
            filename = parts[1].strip()
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_output_{timestamp}.md"

        # 获取消息列表
        messages = self._session.messages

        # 格式化为 Markdown
        lines: list[str] = [
            f"# Session Output ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})\n",
        ]

        for msg in messages:
            formatted = self._format_message_for_save(msg)
            if formatted:
                lines.append(formatted)

        content = "\n".join(lines)

        # 写入文件
        try:
            filepath = Path(filename).resolve()
            filepath.write_text(content, encoding="utf-8")
            output_log.write(
                f"[bold green]✓ 会话已保存到[/] [cyan]{filepath}[/] "
                f"[dim]({len(content)} 字节, {len(messages)} 条消息)[/]"
            )
        except OSError as e:
            output_log.write(f"[bold red]✗ 保存失败:[/] {e}")

    async def action_submit(self) -> None:
        """提交输入（Ctrl+D）。"""
        if self.is_processing:
            return
        input_area = self.query_one("#input-textarea", TextArea)
        text = input_area.text
        # 保留空行用于多行编辑，但以非空行触发
        stripped = text.strip()
        if not stripped:
            return
        input_area.text = ""
        output_log = self.query_one("#output-log", RichLog)

        # 处理 TUI 内建命令
        if is_tui_command(stripped):
            handled = await self._handle_tui_command(stripped)
            if handled:
                return

        # 记录到历史
        self._history.add(stripped)
        # 重置补全状态
        self._completer.reset()

        # 显示用户输入
        output_log.write(f"[bold green]> {stripped}[/]")

        # 扩展命令（/xxx）由 session.prompt() 内部路由到 ExtensionRunner
        self.is_processing = True
        self._update_status_bar()
        try:
            await self._session.prompt(stripped)
        except Exception as exc:
            output_log.write(f"[bold red]Error: {exc}[/]")
            self.is_processing = False
            self._update_status_bar()

    async def action_history_up(self) -> None:
        """上箭头：历史导航。"""
        if self.is_processing:
            return
        current = self._get_input_text()
        if current.strip():
            self._history_stash = current
        entry = self._history.navigate_up()
        if entry is not None:
            self._set_input_text(entry)
            self._completer.reset()

    async def action_history_down(self) -> None:
        """下箭头：历史导航。"""
        if self.is_processing:
            return
        entry = self._history.navigate_down()
        if entry is not None:
            self._set_input_text(entry)
            self._completer.reset()
        else:
            # 回到历史导航前保存的输入
            stash = self._history_stash
            self._history_stash = ""
            self._set_input_text(stash)

    async def action_cancel(self) -> None:
        """取消当前操作（Ctrl+C）。"""
        if self.is_processing:
            self._session.agent.abort()
            self.is_processing = False
            output_log = self.query_one("#output-log", RichLog)
            output_log.write("[bold red]Cancelled[/]")
            self._update_status_bar()

    async def action_quit(self) -> None:
        """退出应用（Ctrl+Q）。"""
        self.exit(return_code=0)

    async def action_new_session(self) -> None:
        """新建会话（Ctrl+N）。"""
        await self._runtime_host.new_session()
        self._session = self._runtime_host.session
        self._subscribe_to_events()
        output_log = self.query_one("#output-log", RichLog)
        output_log.write("[bold cyan]--- New session created ---[/]")
        self._update_info_panel()
        self._update_status_bar()
        # 更新状态栏会话 ID
        try:
            status_bar = self.query_one(StatusBar)
            meta = self._session.session_manager
            sid = meta.get_session_id()
            if sid:
                status_bar.set_session_id(sid)
        except Exception:
            pass

    async def action_pick_session(self) -> None:
        """打开会话选择器（Ctrl+O）。"""
        if self.is_processing:
            return
        picker = SessionPickerScreen(self._runtime_host)
        result = await self.push_screen_wait(picker)

        if result is None:
            return  # 用户取消

        if result == "__new__":
            await self.action_new_session()
            return

        # 切换到选中的会话
        try:
            await self._runtime_host.switch_session(result)
            self._session = self._runtime_host.session
            self._subscribe_to_events()
            output_log = self.query_one("#output-log", RichLog)
            output_log.write(f"[bold cyan]--- Switched to session: {result} ---[/]")
            self._update_info_panel()
            self._update_status_bar()
        except Exception as exc:
            output_log = self.query_one("#output-log", RichLog)
            output_log.write(f"[bold red]Error switching session: {exc}[/]")

    async def action_clear(self) -> None:
        """清屏（Ctrl+L）。"""
        output_log = self.query_one("#output-log", RichLog)
        output_log.clear()

    async def action_focus_input(self) -> None:
        """聚焦输入框（Escape）。"""
        self.query_one("#input-textarea", TextArea).focus()


async def run_interactive_mode(runtime_host: AgentSessionRuntime) -> int:
    """运行交互模式。

    启动 Textual TUI，注册信号处理器，管理运行时生命周期。

    Args:
        runtime_host: AgentSessionRuntime 实例。

    Returns:
        退出码（0 表示正常退出）。
    """
    exit_code = 0
    disposed = False
    signal_cleanup_handlers: list[Callable[[], None]] = []

    async def dispose_runtime() -> None:
        """清理运行时资源（幂等）。"""
        nonlocal disposed
        if disposed:
            return
        disposed = True
        await runtime_host.dispose()

    # 启动 TUI 应用
    app = InteractiveModeApp(runtime_host)

    # 注册信号处理器（必须在 app 创建之后，app.run_async 之前/之后均可）
    signal_cleanup_handlers = register_signal_handlers(
        dispose_runtime=dispose_runtime,
        app_exit=lambda: app.exit(return_code=0) if app._running else None,
    )

    # 注册会话重绑定（用于 session_ui.py 中的会话切换）
    session = runtime_host.session

    async def rebind_session(new_session: AgentSession) -> None:
        nonlocal session
        session = new_session

    runtime_host.set_rebind_session(rebind_session)

    # 启动 TUI 应用
    app = InteractiveModeApp(runtime_host)
    try:
        await app.run_async()
        return exit_code
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        restore_signal_handlers(signal_cleanup_handlers)
        await dispose_runtime()
