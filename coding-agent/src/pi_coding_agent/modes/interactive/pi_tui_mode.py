"""基于 pi_tui 的交互模式（替代 Textual 版）。

对标 TS `packages/coding-agent/src/modes/interactive/interactive-mode.ts` 的
main-screen 交互层，使用 pi_tui（差分渲染 TUI 库）构建：
- 流式消息（snapshot + text_delta）
- 工具调用事件展示
- 斜杠命令（/help /model /compact /thinking /session /tools /clear /exit）
- 编辑器自动补全（斜杠命令 + fd 路径）
- footer 状态栏（model | thinking | ctx）
- Ctrl+P 切换模型
- busy 期间 follow-up 排队

事件桥接（两个订阅通道）：
- ``session.agent.subscribe`` → pi_agent AgentEvent（message/tool/agent_end/turn_end）
- ``session.subscribe`` → AgentSessionEvent（compaction/retry/settled/queue）
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING

from pi_agent.types import AgentEvent, CancellationToken
from pi_tui import (
    TUI,
    AutocompleteItem,
    CombinedAutocompleteProvider,
    Editor,
    Markdown,
    ProcessTerminal,
    RgbColor,
    SlashCommand,
    Spacer,
    Text,
)

from pi_coding_agent.core.agent_session import PromptOptions
from pi_coding_agent.modes.interactive.theme.theme import (
    _bold,
    _cyan,
    _dim,
    _green,
    _red,
    _yellow,
    get_editor_theme,
    get_markdown_theme,
)

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime


# ---------------------------------------------------------------------------
# 终端主题探测（对齐 origin_pi theme.ts 的 detectTerminalThemeForAuto）
# ---------------------------------------------------------------------------


def _theme_for_rgb(rgb: RgbColor) -> str:
    """按相对亮度判断主题：亮度 >= 0.5 视为 light。

    使用与 origin_pi ``getThemeForRgbColor`` 一致的加权公式。
    """
    luminance = (0.299 * rgb.r + 0.587 * rgb.g + 0.114 * rgb.b) / 255.0
    return "light" if luminance >= 0.5 else "dark"


def _detect_theme_from_env() -> str | None:
    """从 ``COLORFGBG`` 环境变量探测终端明暗。

    COLORFGBG 形如 "fg;bg"，bg 为 ANSI 色号（0-15）；色号 >= 8 为亮色系。
    """
    raw = os.environ.get("COLORFGBG", "")
    if not raw:
        return None
    parts = raw.split(";")
    bg = parts[-1] if parts else ""
    if not bg.isdigit():
        return None
    return "light" if int(bg) >= 8 else "dark"


# ---------------------------------------------------------------------------
# 消息内容提取
# ---------------------------------------------------------------------------


def assistant_text_from_message(message: object) -> str:
    """从 assistant 消息快照中提取完整文本。

    兼容 content 元素为 dict（``{"type": "text", "text": ...}``）或 Pydantic
    对象（``.type`` / ``.text`` 属性）两种形态。
    """
    content = getattr(message, "content", None)
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                text = item.get("text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
        else:
            if getattr(item, "type", None) == "text":
                text = getattr(item, "text", "")
                if isinstance(text, str) and text:
                    parts.append(text)
    return "".join(parts)


def assistant_error_from_message(message: object) -> str | None:
    """提取消息的 error_message（assistant 失败时使用）。"""
    err = getattr(message, "error_message", None)
    if isinstance(err, str) and err.strip():
        return err.strip()
    return None


# ---------------------------------------------------------------------------
# 交互模式主实现
# ---------------------------------------------------------------------------


class PiTuiInteractiveMode:
    """pi_tui 交互模式。

    管理组件树（历史区 / 流式区 / 编辑器 / footer）、事件桥接与主循环。
    """

    def __init__(self, session: AgentSession) -> None:
        self._session = session
        # 追踪日志：设置 PI_INTERACTIVE_TRACE_LOG 可写入调用链（便于排查）
        self._trace_path = os.environ.get("PI_INTERACTIVE_TRACE_LOG", "").strip()

        self._terminal = ProcessTerminal()
        self._tui = TUI(self._terminal)

        # 输出区组件
        self._history_text = Text("", padding_x=1, padding_y=0)
        self._stream_text = Text("", padding_x=1, padding_y=0)
        self._footer_text = Text("", padding_x=1, padding_y=0)
        self._editor: Editor | None = None

        # 主题（Markdown 主题含 highlight_code 语法高亮钩子；启动时按终端配色探测切换）
        self._markdown_theme = get_markdown_theme()
        self._theme_dark = True

        # 流式渲染状态
        self._collected: list[str] = []
        self._rendered_response = ""
        # turn_end 已把回复转正到历史区（防止 agent_end 兜底逻辑重复显示）
        self._promoted_to_history = False
        self._pending_tools: dict[str, str] = {}
        self._is_busy = False
        self._is_compacting = False

        # 事件退订函数
        self._unsub_agent: Callable[[], None] | None = None
        self._unsub_session: Callable[[], None] | None = None
        # 主事件循环（run() 中获取，供终端线程回调跨线程调度）
        self._main_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------
    # 追踪日志
    # ------------------------------------------------------------------

    def _trace(self, msg: str) -> None:
        if not self._trace_path:
            return
        try:
            with open(self._trace_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """构建组件树与主题。

        主题统一由 ``theme.theme`` 工厂提供：
        - ``get_editor_theme``：编辑器 / 自动补全配色
        - ``get_markdown_theme``：Markdown 渲染配色 + ``highlight_code`` 语法高亮钩子
        """
        editor_theme = get_editor_theme()

        self._tui.add_child(self._history_text)
        self._tui.add_child(self._stream_text)
        self._tui.add_child(Spacer(1))
        self._tui.add_child(self._footer_text)

        editor = Editor(self._tui, editor_theme)
        self._editor = editor

        # 斜杠命令定义（供补全器使用）
        slash_commands: list[SlashCommand | AutocompleteItem] = [
            SlashCommand(name="exit", description="Exit the agent"),
            SlashCommand(name="clear", description="Clear conversation history"),
            SlashCommand(name="help", description="Show help"),
            SlashCommand(name="model", description="List or switch models"),
            SlashCommand(name="compact", description="Compact conversation context"),
            SlashCommand(
                name="thinking",
                description="Cycle thinking level (off/minimal/low/medium/high)",
            ),
            SlashCommand(name="session", description="Show session statistics"),
            SlashCommand(name="tools", description="List active tools"),
        ]
        autocomplete = CombinedAutocompleteProvider(
            commands=slash_commands,
            base_path=os.getcwd(),
        )
        editor.set_autocomplete_provider(autocomplete)

        self._tui.add_child(editor)
        self._tui.set_focus(editor)

        # 提交回调（pi_tui 在终端线程同步调用 on_submit）
        editor.on_submit = self._on_submit_sync

        self._update_footer()

    # ------------------------------------------------------------------
    # 历史 / 流式区操作
    # ------------------------------------------------------------------

    def _append_history(self, line: str) -> None:
        """追加一行到历史区。"""
        self._trace(f"append_history: {line[:120]!r}")
        current = self._history_text._text  # type: ignore[attr-defined]
        self._history_text.set_text((current + "\n" + line).lstrip("\n"))
        self._history_text.invalidate()

    def _set_stream(self, text: str) -> None:
        """更新当前流式响应行。"""
        self._trace(f"set_stream: {text[:120]!r}")
        self._stream_text.set_text(text)
        self._stream_text.invalidate()
        self._tui.request_render()

    def _render_markdown(self, text: str) -> str:
        """将 markdown 文本渲染为 ANSI 字符串（含代码块语法高亮）。

        用 pi_tui 的 Markdown 组件（注入 ``highlight_code`` 钩子）渲染，
        输出交给历史区 Text 组件显示（ANSI 直通、自动换行与 padding）。

        Args:
            text: markdown 源文本（用户消息 / assistant 回复）。

        Returns:
            渲染后的多行 ANSI 字符串；空输入返回空串。
        """
        if not text:
            return ""
        md = Markdown(
            text,
            padding_x=0,
            padding_y=0,
            theme=self._markdown_theme,
        )
        # 减去历史区 Text 的左右 padding（padding_x=1），避免行宽溢出
        width = max(20, self._terminal.columns - 2)
        return "\n".join(md.render(width))

    async def _detect_and_apply_theme(self) -> None:
        """探测终端配色并应用对应主题（OSC 11 联动）。

        对齐 origin_pi theme.ts 的 ``detectTerminalThemeForAuto``，探测顺序：
        1. DSR 996 配色方案报告（``query_terminal_color_scheme``）
        2. OSC 11 背景色亮度（``query_terminal_background_color``）
        3. ``COLORFGBG`` 环境变量
        4. 默认暗色

        探测成功后按 dark/light 重建 Markdown 主题（含语法高亮配色），
        并触发整体重绘。
        """
        theme_name: str | None = None
        try:
            scheme = await self._tui.query_terminal_color_scheme(timeout_ms=120)
            if scheme is not None:
                theme_name = scheme
        except Exception:
            self._trace("theme: DSR 996 query failed")
        if theme_name is None:
            try:
                bg = await self._tui.query_terminal_background_color(timeout_ms=120)
                if bg is not None:
                    theme_name = _theme_for_rgb(bg)
            except Exception:
                self._trace("theme: OSC 11 query failed")
        if theme_name is None:
            theme_name = _detect_theme_from_env()

        dark = theme_name != "light"
        if dark == self._theme_dark:
            return
        self._theme_dark = dark
        self._markdown_theme = get_markdown_theme(dark)
        self._trace(f"terminal theme detected: {theme_name or 'dark (fallback)'}")
        self._tui.invalidate()
        self._tui.request_render()

    # ------------------------------------------------------------------
    # footer 更新
    # ------------------------------------------------------------------

    def _fmt_tokens(self, n: int) -> str:
        if n >= 1000:
            return f"{n // 1000}k"
        return str(n)

    def _update_footer(self) -> None:
        """刷新 footer：model | thinking: off | ctx: 12% (8k/128k)。"""
        session = self._session
        model = session.model
        model_str = model.model_id if model else "no model"
        thinking = getattr(session, "thinking_level", "off") or "off"
        parts = [model_str, f"thinking: {thinking}"]

        ctx = session.get_context_usage()
        if ctx is not None and ctx.percent is not None:
            tokens = ctx.tokens or 0
            window = ctx.context_window
            parts.append(
                f"ctx: {ctx.percent:.0f}% ({self._fmt_tokens(tokens)}/{self._fmt_tokens(window)})"
            )
        self._footer_text.set_text(_dim("  " + " | ".join(parts)))
        self._footer_text.invalidate()

    # ------------------------------------------------------------------
    # 事件桥接（agent 事件 → UI）
    # ------------------------------------------------------------------

    def _handle_agent_event(
        self, event: AgentEvent, token: CancellationToken | None = None
    ) -> None:
        """处理 pi_agent 事件（message / tool / agent_end / turn_end）。

        Args:
            event: Agent 事件。
            token: 取消令牌（Agent.subscribe 回调签名要求，本模式不使用）。
        """
        etype = getattr(event, "type", "") or ""
        self._trace(f"agent_event: {etype}")
        try:
            if etype == "message_start":
                msg = getattr(event, "message", None)
                if getattr(msg, "role", None) == "assistant":
                    if not self._stream_text._text:  # type: ignore[attr-defined]
                        self._set_stream(f"{_bold('Assistant:')} ")

            elif etype == "message_update":
                msg = getattr(event, "message", None)
                if getattr(msg, "role", None) != "assistant":
                    return
                # 优先用完整快照（流式时模型可能逐 token 更新 content）
                snapshot = assistant_text_from_message(msg)
                if snapshot and snapshot != self._rendered_response:
                    self._rendered_response = snapshot
                    self._set_stream(f"{_bold('Assistant:')} {snapshot}")
                    return
                ae = getattr(event, "assistant_message_event", None)
                if getattr(ae, "type", None) == "text_delta":
                    delta = getattr(ae, "delta", "")
                    if delta:
                        self._collected.append(delta)
                        so_far = "".join(self._collected)
                        if so_far != self._rendered_response:
                            self._rendered_response = so_far
                            self._set_stream(f"{_bold('Assistant:')} {so_far}")

            elif etype == "message_end":
                msg = getattr(event, "message", None)
                if getattr(msg, "role", None) == "assistant":
                    final_text = assistant_text_from_message(msg)
                    if final_text and final_text != self._rendered_response:
                        self._rendered_response = final_text
                        self._set_stream(f"{_bold('Assistant:')} {final_text}")

            elif etype == "tool_execution_start":
                tool_call_id = getattr(event, "tool_call_id", "")
                tool_name = getattr(event, "tool_name", "tool")
                self._pending_tools[tool_call_id] = tool_name
                self._append_history(f"{_yellow('Tool start:')} {tool_name}")
                self._tui.request_render()

            elif etype == "tool_execution_update":
                partial = getattr(event, "partial_result", None)
                if partial is not None:
                    text = str(partial)
                    if len(text) > 200:
                        text = text[:200] + "..."
                    self._append_history(f"  {_dim(text)}")
                    self._tui.request_render()

            elif etype == "tool_execution_end":
                tool_call_id = getattr(event, "tool_call_id", "")
                tool_name = self._pending_tools.pop(
                    tool_call_id, getattr(event, "tool_name", "tool")
                )
                is_error = bool(getattr(event, "is_error", False))
                result = getattr(event, "result", None)
                status = _red("error") if is_error else _green("ok")
                line = f"{_yellow('Tool end:')} {tool_name} ({status})"
                snippet = (
                    assistant_text_from_message(result) if result is not None else ""
                )
                if snippet:
                    compact = " ".join(snippet.split())
                    if len(compact) > 160:
                        compact = compact[:157] + "..."
                    line += f" - {compact}"
                self._append_history(line)
                self._tui.request_render()

            elif etype == "agent_end":
                # 仅当整个 turn 从未渲染过内容时兜底显示（turn_end 已转正时跳过）
                if not self._rendered_response and not self._promoted_to_history:
                    messages = getattr(event, "messages", None)
                    if isinstance(messages, list):
                        for msg in messages:
                            if getattr(msg, "role", None) != "assistant":
                                continue
                            err = assistant_error_from_message(msg)
                            if err:
                                self._set_stream(f"{_red('Error:')} {err}")
                                self._rendered_response = err
                                break
                            fallback = assistant_text_from_message(msg)
                            if fallback:
                                self._set_stream(f"{_bold('Assistant:')} {fallback}")
                                self._rendered_response = fallback
                                break

            elif etype == "turn_end":
                msg = getattr(event, "message", None)
                err = assistant_error_from_message(msg)
                if err:
                    self._set_stream(f"{_red('Error:')} {err}")
                # 将最终回复转正到历史区：Markdown 渲染（含代码块语法高亮）
                final_text = self._rendered_response
                if final_text and not err:
                    rendered = self._render_markdown(final_text)
                    if rendered:
                        self._append_history(rendered)
                    self._set_stream("")
                    self._rendered_response = ""
                    self._collected = []
                    self._promoted_to_history = True
                    self._tui.request_render()
        except Exception as exc:
            self._trace(f"agent_event exception: {exc!r}")

    def _handle_session_event(self, event: object) -> None:
        """处理 AgentSessionEvent（compaction / retry / settled / queue）。"""
        etype = getattr(event, "type", "") or ""
        self._trace(f"session_event: {etype}")
        try:
            if etype == "agent_settled":
                self._is_busy = False
                self._update_footer()

            elif etype == "compaction_start":
                self._is_compacting = True
                self._append_history(_dim("Compacting context..."))
                self._tui.request_render()

            elif etype == "compaction_end":
                self._is_compacting = False
                err = getattr(event, "error_message", None)
                if err:
                    self._append_history(f"{_red('Compaction error:')} {err}")
                else:
                    self._append_history(_dim("Context compacted."))
                self._update_footer()
                self._tui.request_render()

            elif etype == "auto_retry_start":
                attempt = getattr(event, "attempt", 0)
                max_a = getattr(event, "max_attempts", 3)
                delay = getattr(event, "delay_ms", 0)
                err = getattr(event, "error_message", "") or ""
                self._append_history(
                    f"{_yellow(f'Retry {attempt}/{max_a}:')} {_dim(str(err))} "
                    f"(wait {delay // 1000}s)"
                )
                self._tui.request_render()

            elif etype == "auto_retry_end":
                success = getattr(event, "success", True)
                if not success:
                    err = getattr(event, "final_error", "") or ""
                    self._append_history(f"{_red('Retry failed:')} {err}")
                    self._tui.request_render()

            elif etype == "queue_update":
                queued = getattr(event, "follow_up", None) or []
                if queued:
                    self._append_history(
                        f"{_dim('Queued:')} {queued[0][:80] if isinstance(queued[0], str) else queued[0]}"
                    )
                    self._tui.request_render()
        except Exception as exc:
            self._trace(f"session_event exception: {exc!r}")

    # ------------------------------------------------------------------
    # 提交处理（斜杠命令 + prompt）
    # ------------------------------------------------------------------

    async def _handle_submit(self, text: str) -> None:
        """处理用户提交（斜杠命令或普通消息）。"""
        session = self._session
        stripped = text.strip()
        if not stripped:
            return

        # ── 斜杠命令 ────────────────────────────────────────────────
        if stripped in ("/exit", "exit", "quit"):
            self._tui.stop()
            return

        if stripped == "/clear":
            self._history_text.set_text("")
            self._history_text.invalidate()
            self._set_stream("")
            return

        if stripped == "/help":
            lines = [
                _bold("Available commands:"),
                f"  {_cyan('/exit')}     — Exit the agent",
                f"  {_cyan('/clear')}    — Clear conversation history",
                f"  {_cyan('/model')}    — List available models / switch model",
                f"  {_cyan('/compact')}  — Compact context to free tokens",
                f"  {_cyan('/thinking')} — Cycle thinking level",
                f"  {_cyan('/session')}  — Show session statistics",
                f"  {_cyan('/tools')}    — List active tools",
                f"  {_cyan('Ctrl+P')}    — Cycle to next model",
            ]
            self._append_history("\n".join(lines))
            self._tui.request_render()
            return

        if stripped == "/tools":
            names = session.get_active_tool_names()
            if names:
                self._append_history(
                    _bold("Active tools:") + "\n" + "\n".join(f"  - {n}" for n in names)
                )
            else:
                self._append_history(_dim("No active tools."))
            self._tui.request_render()
            return

        if stripped == "/session":
            stats = session.get_session_stats()
            lines = [
                _bold("Session stats:"),
                f"  Session ID:   {stats.session_id or '?'}",
                f"  User msgs:    {stats.user_messages}",
                f"  Asst msgs:    {stats.assistant_messages}",
                f"  Tool calls:   {stats.tool_calls}",
                f"  Total tokens: {self._fmt_tokens(stats.tokens.get('total', 0))}",
                f"  Cost:         ${stats.cost:.4f}",
            ]
            self._append_history("\n".join(lines))
            self._tui.request_render()
            return

        if stripped == "/thinking":
            new_level = session.cycle_thinking_level()
            if new_level:
                self._append_history(f"{_cyan('Thinking level:')} {new_level}")
            else:
                self._append_history(_dim("Thinking not supported by current model."))
            self._update_footer()
            self._tui.request_render()
            return

        if stripped == "/compact":
            self._append_history(_dim("Compacting context..."))
            self._tui.request_render()
            try:
                result = await session.compact()
                summary = result.summary if result else ""
                if summary:
                    short = summary[:400] + "..." if len(summary) > 400 else summary
                    self._append_history(
                        f"{_green('Compaction complete.')}\n{_dim(short)}"
                    )
                else:
                    self._append_history(
                        _dim("Compaction complete (nothing to summarize).")
                    )
            except Exception as exc:
                self._append_history(f"{_red('Compaction error:')} {exc}")
            self._update_footer()
            self._tui.request_render()
            return

        if stripped == "/model" or stripped.startswith("/model "):
            await self._handle_model_command(stripped)
            return

        # ── busy 保护：流式中输入 → 排队 follow-up ─────────────────
        if self._is_busy:
            await session.follow_up(stripped)
            self._append_history(f"{_dim('Queued follow-up:')} {stripped}")
            self._tui.request_render()
            return

        self._is_busy = True
        self._collected = []
        self._rendered_response = ""
        self._promoted_to_history = False

        # 用户消息也走 Markdown 渲染（含代码块语法高亮），对齐 origin_pi 消息列表
        user_rendered = self._render_markdown(f"**You:** {stripped}")
        self._append_history(user_rendered or f"{_bold('You:')} {stripped}")
        self._tui.request_render()

        try:
            await session.prompt(stripped, PromptOptions(source="interactive"))
        except asyncio.TimeoutError:
            await session.abort()
            self._set_stream(_yellow("Response timed out and was aborted"))
        except Exception as exc:
            self._trace(f"prompt exception: {exc!r}")
            self._set_stream(f"{_red('Error:')} {exc}")
        finally:
            # 流式结果收尾：从 stream 区移入历史区
            final = self._stream_text._text  # type: ignore[attr-defined]
            if final:
                self._append_history(final)
                self._set_stream("")
            self._is_busy = False
            self._update_footer()
            self._tui.request_render()

    async def _handle_model_command(self, stripped: str) -> None:
        """处理 /model 与 /model <id>。"""
        session = self._session
        parts = stripped.split(None, 1)
        model_arg = parts[1].strip() if len(parts) > 1 else None

        available = session.model_runtime.get_available_snapshot()
        current = session.model

        if model_arg:
            target = next(
                (
                    m
                    for m in available
                    if m.model_id == model_arg
                    or m.model_id.lower() == model_arg.lower()
                ),
                None,
            )
            if target is None:
                self._append_history(f"{_red('Unknown model:')} {model_arg}")
            else:
                try:
                    await session.set_model(target)
                    self._append_history(
                        f"{_cyan('Switched to model:')} {target.model_id} ({target.provider})"
                    )
                except Exception as exc:
                    self._append_history(f"{_red('Model switch failed:')} {exc}")
            self._update_footer()
            self._tui.request_render()
            return

        if not available:
            self._append_history(_dim("No models available."))
            self._tui.request_render()
            return

        lines = [_bold("Available models:")]
        for m in available:
            marker = _cyan("→") if (current and m.model_id == current.model_id) else " "
            lines.append(f"  {marker} {m.model_id} ({m.provider})")
        lines.append(_dim("Use /model <id> to switch."))
        self._append_history("\n".join(lines))
        self._tui.request_render()

    async def _cycle_model_async(self) -> None:
        """Ctrl+P：切换到下一个模型。"""
        try:
            result = await self._session.cycle_model("forward")
            if result and result.model:
                m = result.model
                self._append_history(f"{_cyan('Model:')} {m.model_id} ({m.provider})")
            else:
                self._append_history(_dim("Only one model available."))
            self._update_footer()
            self._tui.request_render()
        except Exception as exc:
            self._append_history(f"{_red('Model switch failed:')} {exc}")
            self._tui.request_render()

    # ------------------------------------------------------------------
    # 同步回调（终端线程 → 事件循环）
    # ------------------------------------------------------------------

    def _on_submit_sync(self, text: str) -> None:
        """Editor.on_submit 回调（pi_tui 在终端线程调用）。"""
        if self._main_loop is None:
            return
        asyncio.run_coroutine_threadsafe(self._handle_submit(text), self._main_loop)

    def _on_keydown_sync(self, key: str) -> None:
        """处理特殊按键（Ctrl+P = \\x10）。"""
        if key == "\x10" and self._main_loop is not None:
            asyncio.run_coroutine_threadsafe(self._cycle_model_async(), self._main_loop)

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def _subscribe(self) -> None:
        """订阅 agent 事件与 session 事件。"""
        self._unsub_agent = self._session.agent.subscribe(self._handle_agent_event)
        self._unsub_session = self._session.subscribe(self._handle_session_event)

    def _unsubscribe(self) -> None:
        if self._unsub_agent is not None:
            self._unsub_agent()
            self._unsub_agent = None
        if self._unsub_session is not None:
            self._unsub_session()
            self._unsub_session = None

    async def run(self) -> int:
        """启动 TUI 并进入主循环。"""
        self._main_loop = asyncio.get_running_loop()
        self._build_ui()
        self._subscribe()

        # Ctrl+P 通过 TUI 输入监听器捕获（与 pi-mono 的 editor.on_keydown 等价）
        self._tui.add_input_listener(self._on_keydown_sync)

        self._trace("tui: start")
        self._tui.start()

        # OSC 11 联动：启动时探测终端配色并按需切换主题（不阻塞主循环）
        await self._detect_and_apply_theme()

        try:
            while not self._tui.stopped:
                await asyncio.sleep(0.05)
        except (KeyboardInterrupt, asyncio.CancelledError):
            self._trace("tui: keyboard/cancelled")
        finally:
            if not self._tui.stopped:
                self._trace("tui: stop in finally")
                self._tui.stop()
            self._unsubscribe()
        return 0


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


async def run_interactive_mode(runtime_host: AgentSessionRuntime) -> int:
    """运行 pi_tui 交互模式。

    Args:
        runtime_host: AgentSessionRuntime 实例（含 session 与 dispose）。

    Returns:
        退出码（0 表示正常退出）。
    """
    exit_code = 0
    disposed = False

    async def dispose_runtime() -> None:
        """清理运行时资源（幂等）。"""
        nonlocal disposed
        if disposed:
            return
        disposed = True
        await runtime_host.dispose()

    session = runtime_host.session
    mode = PiTuiInteractiveMode(session)
    try:
        exit_code = await mode.run()
    except Exception as error:
        print(str(error), file=sys.stderr)
        exit_code = 1
    finally:
        await dispose_runtime()
    return exit_code
