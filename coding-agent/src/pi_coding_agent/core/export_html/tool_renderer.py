"""工具 HTML 渲染器。

将自定义工具调用和结果渲染为 HTML，通过调用其 TUI 渲染器
并将 ANSI 输出转换为 HTML。
"""

from __future__ import annotations

from typing import Any, Protocol

from ..extensions.types import ToolDefinition, ToolRenderContext

from .ansi_to_html import ansi_lines_to_html

# 匹配 ANSI 转义序列：ESC[ 后跟参数，以 'm' 结尾
ANSI_ESCAPE_REGEX = r"\x1b\[[\d;]*m"


class ToolHtmlRendererDeps(Protocol):
    """工具 HTML 渲染器依赖。"""

    get_tool_definition: Any  # Callable[[str], ToolDefinition | None]
    """按名称查找工具定义的函数。"""
    theme: Any
    """用于样式设置的主题。"""
    cwd: str
    """渲染上下文的工作目录。"""
    width: int
    """渲染的终端宽度（默认：100）。"""


class ToolHtmlRenderer(Protocol):
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


import re


def _is_blank_rendered_line(line: str) -> bool:
    """检查行的渲染后内容是否为空。"""
    return len(re.sub(ANSI_ESCAPE_REGEX, "", line).strip()) == 0


def _trim_rendered_result_lines(lines: list[str]) -> list[str]:
    """修剪渲染结果的前后空白行。"""
    start = 0
    end = len(lines)
    while start < end and _is_blank_rendered_line(lines[start]):
        start += 1
    while end > start and _is_blank_rendered_line(lines[end - 1]):
        end -= 1
    return lines[start:end]


def create_tool_html_renderer(
    deps: ToolHtmlRendererDeps,
) -> ToolHtmlRenderer:
    """创建工具 HTML 渲染器。

    渲染器查找工具定义并调用其 renderCall/renderResult 方法，
    将生成的 TUI Component 输出（ANSI）转换为 HTML。
    """
    get_tool_definition = deps.get_tool_definition
    theme = deps.theme
    cwd = deps.cwd
    width = getattr(deps, "width", 100)

    rendered_call_components: dict[str, Any] = {}
    rendered_result_components: dict[str, Any] = {}
    rendered_states: dict[str, Any] = {}
    rendered_args: dict[str, object] = {}

    def get_state(tool_call_id: str) -> Any:
        state = rendered_states.get(tool_call_id)
        if state is None:
            state = {}
            rendered_states[tool_call_id] = state
        return state

    def create_render_context(
        tool_call_id: str,
        last_component: Any,
        expanded: bool,
        is_partial: bool,
        is_error: bool,
    ) -> ToolRenderContext:
        return ToolRenderContext(
            args=rendered_args.get(tool_call_id),
            tool_call_id=tool_call_id,
            invalidate=lambda: None,
            last_component=last_component,
            state=get_state(tool_call_id),
            cwd=cwd,
            execution_started=True,
            args_complete=True,
            is_partial=is_partial,
            expanded=expanded,
            show_images=False,
            is_error=is_error,
        )

    class _Renderer:
        """内部渲染器实现。"""

        def render_call(
            self, tool_call_id: str, tool_name: str, args: object
        ) -> str | None:
            try:
                rendered_args[tool_call_id] = args
                tool_def = get_tool_definition(tool_name)
                if tool_def is None or not hasattr(tool_def, "render_call"):
                    return None
                render_call_fn = getattr(tool_def, "render_call")
                if render_call_fn is None:
                    return None

                component = render_call_fn(
                    args,
                    theme,
                    create_render_context(
                        tool_call_id,
                        rendered_call_components.get(tool_call_id),
                        False,
                        True,
                        False,
                    ),
                )
                rendered_call_components[tool_call_id] = component
                lines = component.render(width)
                return ansi_lines_to_html(lines)
            except Exception:
                # 出错时返回 None，以便 HTML 导出可回退到结构化结果渲染
                return None

        def render_result(
            self,
            tool_call_id: str,
            tool_name: str,
            result: list[dict[str, object]],
            details: object,
            is_error: bool,
        ) -> dict[str, str] | None:
            try:
                tool_def = get_tool_definition(tool_name)
                if tool_def is None or not hasattr(tool_def, "render_result"):
                    return None
                render_result_fn = getattr(tool_def, "render_result")
                if render_result_fn is None:
                    return None

                # 从内容数组构建 AgentToolResult
                agent_tool_result = {
                    "content": result,
                    "details": details,
                    "is_error": is_error,
                }

                # 渲染折叠视图
                collapsed_component = render_result_fn(
                    agent_tool_result,
                    {"expanded": False, "is_partial": False},
                    theme,
                    create_render_context(
                        tool_call_id,
                        rendered_result_components.get(tool_call_id),
                        False,
                        False,
                        is_error,
                    ),
                )
                rendered_result_components[tool_call_id] = collapsed_component
                collapsed = ansi_lines_to_html(
                    _trim_rendered_result_lines(collapsed_component.render(width))
                )

                # 渲染展开视图
                expanded_component = render_result_fn(
                    agent_tool_result,
                    {"expanded": True, "is_partial": False},
                    theme,
                    create_render_context(
                        tool_call_id,
                        rendered_result_components.get(tool_call_id),
                        True,
                        False,
                        is_error,
                    ),
                )
                rendered_result_components[tool_call_id] = expanded_component
                expanded = ansi_lines_to_html(
                    _trim_rendered_result_lines(expanded_component.render(width))
                )

                result_dict: dict[str, str] = {}
                if collapsed and collapsed != expanded:
                    result_dict["collapsed"] = collapsed
                result_dict["expanded"] = expanded
                return result_dict
            except Exception:
                # 出错时返回 None，以便 HTML 导出可回退到结构化结果渲染
                return None

    return _Renderer()