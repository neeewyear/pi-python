"""系统提示词格式化。"""

from __future__ import annotations

from .types import Skill


def _escape_xml(value: str) -> str:
    """转义 XML 特殊字符。"""
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def format_skills_for_system_prompt(skills: list[Skill]) -> str:
    """把技能列表格式化为规范兼容的 ``<available_skills>`` 系统提示词块。

    被 ``disable_model_invocation`` 标记的技能不展示给模型。
    """
    visible = [skill for skill in skills if not skill.disable_model_invocation]
    if not visible:
        return ""

    lines = [
        "The following skills provide specialized instructions for specific tasks.",
        "Read the full skill file when the task matches its description.",
        "When a skill file references a relative path, resolve it against the skill directory (parent of SKILL.md / dirname of the path) and use that absolute path in tool commands.",
        "",
        "<available_skills>",
    ]
    for skill in visible:
        lines.append("  <skill>")
        lines.append(f"    <name>{_escape_xml(skill.name)}</name>")
        lines.append(f"    <description>{_escape_xml(skill.description)}</description>")
        lines.append(f"    <location>{_escape_xml(skill.file_path)}</location>")
        lines.append("  </skill>")
    lines.append("</available_skills>")
    return "\n".join(lines)
