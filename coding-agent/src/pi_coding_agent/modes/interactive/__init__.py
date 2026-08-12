"""Interactive mode - pi_tui 实现。

提供终端交互式对话界面，支持流式输出、代码高亮、工具调用面板、
斜杠命令、自动补全等功能（基于 pi_tui 差分渲染 TUI 库，替代原 Textual 版）。
"""

from __future__ import annotations

from .pi_tui_mode import PiTuiInteractiveMode, run_interactive_mode

__all__ = [
    "PiTuiInteractiveMode",
    "run_interactive_mode",
]
