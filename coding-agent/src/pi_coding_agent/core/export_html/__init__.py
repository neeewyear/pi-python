"""导出 HTML 模块入口。

将按舒适话条目导出为独立的 HTML 文件，包含自定义主题、
工具渲染器支持，以及语法高亮等。
"""

from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import orjson

from ...config import APP_NAME
from ..session_manager import SessionEntry, SessionManager
from .ansi_to_html import ansi_lines_to_html
from .templates import HIGHLIGHT_JS, MARKED_JS, TEMPLATE_CSS, TEMPLATE_HTML, TEMPLATE_JS
from .tool_renderer import ToolHtmlRenderer

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------


class ToolHtmlRendererProtocol(Protocol):
    """工具 HTML 渲染器协议。"""

    def render_call(
        self, tool_call_id: str, tool_name: str, args: object
    ) -> str | None:
        """将工具调用渲染为 HTML。如果工具没有自定义渲染器，返回 None。"""
        ...

    def render_result(
        self,
        tool_call_id: str,
        tool_name: str,
        result: list[dict[str, object]],
        details: object,
        is_error: bool,
    ) -> dict[str, str] | None:
        """将工具结果渲染为折叠/展开 HTML。如果工具没有自定义渲染器，返回 None。"""
        ...


@dataclass
class ExportOptions:
    """导出选项。"""

    output_path: str | None = None
    """输出文件路径。"""
    theme_name: str | None = None
    """主题名称。"""
    tool_renderer: ToolHtmlRenderer | None = None
    """可选的自定义工具渲染器。"""


@dataclass
class _RenderedToolHtml:
    """预渲染的自定义工具调用和结果 HTML。"""

    call_html: str | None = None
    result_html_collapsed: str | None = None
    result_html_expanded: str | None = None


# ---------------------------------------------------------------------------
# 颜色工具
# ---------------------------------------------------------------------------


def _parse_color(color: str) -> dict[str, int] | None:
    """将颜色字符串解析为 RGB 值。支持 hex（#RRGGBB）和 rgb(r,g,b) 格式。"""
    hex_match = re.match(r"^#([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", color)
    if hex_match:
        return {
            "r": int(hex_match.group(1), 16),
            "g": int(hex_match.group(2), 16),
            "b": int(hex_match.group(3), 16),
        }
    rgb_match = re.match(r"^rgb\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$", color)
    if rgb_match:
        return {
            "r": int(rgb_match.group(1)),
            "g": int(rgb_match.group(2)),
            "b": int(rgb_match.group(3)),
        }
    return None


def _get_luminance(r: int, g: int, b: int) -> float:
    """计算颜色的相对亮度（0-1，越高越亮）。"""

    def to_linear(c: int) -> float:
        s = c / 255
        return s / 12.92 if s <= 0.03928 else ((s + 0.055) / 1.055) ** 2.4

    return 0.2126 * to_linear(r) + 0.7152 * to_linear(g) + 0.0722 * to_linear(b)


def _adjust_brightness(color: str, factor: float) -> str:
    """调整颜色亮度。factor > 1 变亮，< 1 变暗。"""
    parsed = _parse_color(color)
    if parsed is None:
        return color

    def adjust(c: int) -> int:
        return max(0, min(255, round(c * factor)))

    return f"rgb({adjust(parsed['r'])}, {adjust(parsed['g'])}, {adjust(parsed['b'])})"


def _derive_export_colors(
    base_color: str,
) -> dict[str, str]:
    """从基础颜色（如 userMessageBg）派生导出背景色。"""
    parsed = _parse_color(base_color)
    if parsed is None:
        return {
            "page_bg": "rgb(24, 24, 30)",
            "card_bg": "rgb(30, 30, 36)",
            "info_bg": "rgb(60, 55, 40)",
        }

    luminance = _get_luminance(parsed["r"], parsed["g"], parsed["b"])
    is_light = luminance > 0.5

    if is_light:
        return {
            "page_bg": _adjust_brightness(base_color, 0.96),
            "card_bg": base_color,
            "info_bg": f"rgb({min(255, parsed['r'] + 10)}, {min(255, parsed['g'] + 5)}, {max(0, parsed['b'] - 20)})",
        }
    return {
        "page_bg": _adjust_brightness(base_color, 0.7),
        "card_bg": _adjust_brightness(base_color, 0.85),
        "info_bg": f"rgb({min(255, parsed['r'] + 20)}, {min(255, parsed['g'] + 15)}, {parsed['b']})",
    }


# ---------------------------------------------------------------------------
# 主题变量生成
# ---------------------------------------------------------------------------


def _get_default_theme_colors() -> dict[str, str]:
    """获取默认主题颜色。"""
    return {
        "userMessageBg": "#343541",
        "text": "#e0e0e0",
        "muted": "#888888",
        "dim": "#666666",
        "accent": "#4a9eff",
        "success": "#22c55e",
        "error": "#ef4444",
        "warning": "#e8a838",
        "border": "#444444",
        "borderAccent": "#4a9eff",
        "selectedBg": "rgba(74, 158, 255, 0.15)",
        "hover": "rgba(255, 255, 255, 0.05)",
        "thinkingText": "#888888",
        "toolOutput": "#cccccc",
        "toolPendingBg": "rgba(255, 255, 255, 0.03)",
        "toolSuccessBg": "rgba(34, 197, 94, 0.08)",
        "toolErrorBg": "rgba(239, 68, 68, 0.08)",
        "toolDiffAdded": "#22c55e",
        "toolDiffRemoved": "#ef4444",
        "toolDiffContext": "#888888",
        "customMessageBg": "rgba(255, 255, 255, 0.03)",
        "customMessageText": "#cccccc",
        "customMessageLabel": "#e8a838",
        "mdHeading": "#e0e0e0",
        "mdLink": "#4a9eff",
        "mdCode": "#e8a838",
        "mdQuote": "#888888",
        "mdQuoteBorder": "#444444",
        "mdListBullet": "#4a9eff",
        "mdHr": "#444444",
        "mdCodeBlockBorder": "#444444",
        "syntaxComment": "#6a9955",
        "syntaxKeyword": "#569cd6",
        "syntaxNumber": "#b5cea8",
        "syntaxString": "#ce9178",
        "syntaxFunction": "#dcdcaa",
        "syntaxType": "#4ec9b0",
        "syntaxVariable": "#9cdcfe",
        "syntaxOperator": "#d4d4d4",
        "syntaxPunctuation": "#d4d4d4",
        "userMessageText": "#e0e0e0",
    }


def _get_theme_export_colors(
    theme_name: str | None = None,
) -> dict[str, str | None]:
    """获取主题导出颜色（如果可用）。

    当前返回空字典，表示回退到派生颜色。
    """
    return {"pageBg": None, "cardBg": None, "infoBg": None}


def _generate_theme_vars(theme_name: str | None = None) -> str:
    """从主题颜色生成 CSS 自定义属性声明。"""
    colors = _get_default_theme_colors()
    lines: list[str] = []
    for key, value in colors.items():
        lines.append(f"      --{key}: {value};")

    # 使用显式主题导出颜色（如果可用），否则从 userMessageBg 派生
    theme_export = _get_theme_export_colors(theme_name)
    user_message_bg = colors.get("userMessageBg", "#343541")
    derived_colors = _derive_export_colors(user_message_bg)

    page_bg = theme_export.get("pageBg") or derived_colors["page_bg"]
    card_bg = theme_export.get("cardBg") or derived_colors["card_bg"]
    info_bg = theme_export.get("infoBg") or derived_colors["info_bg"]

    lines.append(f"      --exportPageBg: {page_bg};")
    lines.append(f"      --exportCardBg: {card_bg};")
    lines.append(f"      --exportInfoBg: {info_bg};")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 由 HTML 模板直接渲染的工具（不通过 TUI→ANSI→HTML 管道预渲染）
TEMPLATE_RENDERED_TOOLS = frozenset(["bash", "read", "write", "edit", "ls"])


# ---------------------------------------------------------------------------
# 预渲染工具
# ---------------------------------------------------------------------------


def _pre_render_custom_tools(
    entries: list[SessionEntry],
    tool_renderer: ToolHtmlRenderer,
) -> dict[str, _RenderedToolHtml]:
    """使用 TUI 渲染器预渲染自定义工具到 HTML。"""
    rendered_tools: dict[str, _RenderedToolHtml] = {}

    for entry in entries:
        if entry.type != "message":
            continue
        msg = entry.message

        # 在 assistant 消息中查找工具调用
        if msg.role == "assistant" and isinstance(msg.content, list):
            for block in msg.content:
                if (
                    hasattr(block, "type")
                    and block.type == "toolCall"
                    and hasattr(block, "name")
                    and block.name not in TEMPLATE_RENDERED_TOOLS
                ):
                    call_html = tool_renderer.render_call(
                        block.tool_call_id, block.name, block.args
                    )
                    if call_html is not None:
                        rendered_tools[block.tool_call_id] = _RenderedToolHtml(
                            call_html=call_html
                        )

        # 查找工具结果
        if (
            msg.role == "toolResult"
            and hasattr(msg, "tool_call_id")
            and msg.tool_call_id
        ):
            tool_name = getattr(msg, "tool_name", "") or ""
            # 仅当我们有预渲染的调用或它不是模板渲染的工具时才渲染
            existing = rendered_tools.get(msg.tool_call_id)
            if existing is not None or tool_name not in TEMPLATE_RENDERED_TOOLS:
                content_list = (
                    [
                        {"type": c.type, "text": getattr(c, "text", None)}
                        for c in msg.content
                    ]
                    if isinstance(msg.content, list)
                    else []
                )
                rendered = tool_renderer.render_result(
                    msg.tool_call_id,
                    tool_name,
                    content_list,
                    getattr(msg, "details", None),
                    getattr(msg, "is_error", False) or False,
                )
                if rendered is not None:
                    rendered_tools[msg.tool_call_id] = _RenderedToolHtml(
                        call_html=existing.call_html if existing else None,
                        result_html_collapsed=rendered.get("collapsed"),
                        result_html_expanded=rendered.get("expanded"),
                    )

    return rendered_tools


# ---------------------------------------------------------------------------
# HTML 生成
# ---------------------------------------------------------------------------


def _generate_html(
    session_data: dict[str, object],
    theme_name: str | None = None,
) -> str:
    """核心 HTML 生成逻辑。"""
    theme_vars = _generate_theme_vars(theme_name)
    colors = _get_default_theme_colors()
    theme_export = _get_theme_export_colors(theme_name)
    derived_export_colors = _derive_export_colors(
        colors.get("userMessageBg", "#343541")
    )
    body_bg = theme_export.get("pageBg") or derived_export_colors["page_bg"]
    container_bg = theme_export.get("cardBg") or derived_export_colors["card_bg"]
    info_bg = theme_export.get("infoBg") or derived_export_colors["info_bg"]

    # Base64 编码会话数据以避免转义问题
    session_data_json = orjson.dumps(session_data)
    session_data_base64 = base64.b64encode(session_data_json).decode("ascii")

    # 构建 CSS，注入主题变量
    css = (
        TEMPLATE_CSS.replace("{{THEME_VARS}}", theme_vars)
        .replace("{{BODY_BG}}", body_bg)
        .replace("{{CONTAINER_BG}}", container_bg)
        .replace("{{INFO_BG}}", info_bg)
    )

    return (
        TEMPLATE_HTML.replace("{{CSS}}", css)
        .replace("{{JS}}", TEMPLATE_JS)
        .replace("{{SESSION_DATA}}", session_data_base64)
        .replace("{{MARKED_JS}}", MARKED_JS)
        .replace("{{HIGHLIGHT_JS}}", HIGHLIGHT_JS)
    )


# ---------------------------------------------------------------------------
# 导出函数
# ---------------------------------------------------------------------------


async def export_session_to_html(
    session_manager: SessionManager,
    state: Any = None,
    options: ExportOptions | str | None = None,
) -> str:
    """将会话导出为 HTML。

    用于 TUI 的 /export 命令。
    """
    if isinstance(options, str):
        opts = ExportOptions(output_path=options)
    elif options is None:
        opts = ExportOptions()
    else:
        opts = options

    session_file = session_manager.get_session_file()
    if session_file is None:
        raise ValueError("Cannot export in-memory session to HTML")
    session_file_path = Path(session_file)
    if not session_file_path.exists():
        raise ValueError("Nothing to export yet - start a conversation first")

    entries = session_manager.get_entries()

    # 如果提供了工具渲染器，预渲染自定义工具
    rendered_tools: dict[str, _RenderedToolHtml] | None = None
    if opts.tool_renderer is not None:
        rendered_tools = _pre_render_custom_tools(entries, opts.tool_renderer)
        if len(rendered_tools) == 0:
            rendered_tools = None

    # 构建会话数据
    header = session_manager.get_header()
    session_data: dict[str, object] = {
        "header": header.model_dump() if header is not None else None,
        "entries": [e.model_dump() for e in entries],
        "leafId": session_manager.get_leaf_id(),
        "systemPrompt": getattr(state, "system_prompt", None)
        if state is not None
        else None,
        "tools": (
            [
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.parameters,
                }
                for t in state.tools
            ]
            if state is not None and hasattr(state, "tools") and state.tools
            else None
        ),
        "renderedTools": rendered_tools,
    }

    html = _generate_html(session_data, opts.theme_name)

    # 确定输出路径
    output_path: str | None = None
    if opts.output_path is not None:
        output_path = opts.output_path
    else:
        session_basename = session_file_path.stem
        output_path = f"{APP_NAME}-session-{session_basename}.html"

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


async def export_from_file(
    input_path: str,
    options: ExportOptions | str | None = None,
) -> str:
    """将会话文件导出为 HTML（独立使用，无需 AgentState）。

    用于 CLI 导出任意会话文件。
    """
    if isinstance(options, str):
        opts = ExportOptions(output_path=options)
    elif options is None:
        opts = ExportOptions()
    else:
        opts = options

    resolved_input_path = Path(input_path).resolve()
    if not resolved_input_path.exists():
        raise FileNotFoundError(f"File not found: {resolved_input_path}")

    session_manager = SessionManager.open(str(resolved_input_path))

    header = session_manager.get_header()
    session_data: dict[str, object] = {
        "header": header.model_dump() if header is not None else None,
        "entries": [e.model_dump() for e in session_manager.get_entries()],
        "leafId": session_manager.get_leaf_id(),
        "systemPrompt": None,
        "tools": None,
    }

    html = _generate_html(session_data, opts.theme_name)

    # 确定输出路径
    output_path: str | None = None
    if opts.output_path is not None:
        output_path = opts.output_path
    else:
        input_basename = resolved_input_path.stem
        output_path = f"{APP_NAME}-session-{input_basename}.html"

    Path(output_path).write_text(html, encoding="utf-8")
    return output_path


__all__ = [
    "ExportOptions",
    "ToolHtmlRenderer",
    "ToolHtmlRendererProtocol",
    "export_from_file",
    "export_session_to_html",
]
