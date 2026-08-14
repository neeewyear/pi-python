"""系统提示词构建。

提供 ``build_system_prompt`` 函数，用于构建包含工具列表、指南和项目上下文的
系统提示词。
"""

from __future__ import annotations

from pydantic import BaseModel

from ..config import get_docs_path, get_examples_path, get_readme_path
from .skills import Skill, format_skills_for_prompt


class BuildSystemPromptOptions(BaseModel):
    """系统提示词构建选项。"""

    custom_prompt: str | None = None
    selected_tools: list[str] | None = None
    tool_snippets: dict[str, str] | None = None
    prompt_guidelines: list[str] | None = None
    append_system_prompt: str | None = None
    cwd: str
    context_files: list[dict[str, str]] | None = None
    skills: list[Skill] | None = None


def build_system_prompt(options: BuildSystemPromptOptions) -> str:
    """构建系统提示词。

    包含工具列表、操作指南、Pi 文档路径、项目上下文和技能信息。
    """
    prompt_cwd = options.cwd.replace("\\", "/")

    append_section = (
        f"\n\n{options.append_system_prompt}" if options.append_system_prompt else ""
    )

    context_files = options.context_files or []
    skills = options.skills or []

    if options.custom_prompt:
        prompt = options.custom_prompt

        if append_section:
            prompt += append_section

        # Append project context files
        if context_files:
            prompt += "\n\n<project_context>\n\n"
            prompt += "Project-specific instructions and guidelines:\n\n"
            for cf in context_files:
                prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
            prompt += "</project_context>\n"

        # Append skills section (only if read tool is available)
        custom_prompt_has_read = (
            not options.selected_tools or "read" in options.selected_tools
        )
        if custom_prompt_has_read and skills:
            prompt += format_skills_for_prompt(skills)

        prompt += f"\nCurrent working directory: {prompt_cwd}"
        return prompt

    # Get absolute paths to documentation and examples
    readme_path = str(get_readme_path())
    docs_path = str(get_docs_path())
    examples_path = str(get_examples_path())

    # Build tools list based on selected tools
    tools = options.selected_tools or ["read", "bash", "edit", "write"]
    tool_snippets = options.tool_snippets or {}
    visible_tools = [name for name in tools if name in tool_snippets]
    if visible_tools:
        tools_list = "\n".join(
            f"- {name}: {tool_snippets[name]}" for name in visible_tools
        )
    else:
        tools_list = "(none)"

    # Build guidelines
    guidelines_list: list[str] = []
    guidelines_set: set[str] = set()

    def add_guideline(guideline: str) -> None:
        if guideline not in guidelines_set:
            guidelines_set.add(guideline)
            guidelines_list.append(guideline)

    has_bash = "bash" in tools
    has_grep = "grep" in tools
    has_find = "find" in tools
    has_ls = "ls" in tools
    has_read = "read" in tools

    # File exploration guidelines
    if has_bash and not has_grep and not has_find and not has_ls:
        add_guideline("Use bash for file operations like ls, rg, find")

    for guideline in options.prompt_guidelines or []:
        normalized = guideline.strip()
        if normalized:
            add_guideline(normalized)

    # Always include these
    add_guideline("Be concise in your responses")
    add_guideline("Show file paths clearly when working with files")

    guidelines = "\n".join(f"- {g}" for g in guidelines_list)

    prompt = (
        f"You are an expert coding assistant operating inside pi, a coding agent harness. "
        f"You help users by reading files, executing commands, editing code, and writing new files.\n"
        f"\n"
        f"Available tools:\n"
        f"{tools_list}\n"
        f"\n"
        f"In addition to the tools above, you may have access to other custom tools depending on the project.\n"
        f"\n"
        f"Guidelines:\n"
        f"{guidelines}\n"
        f"\n"
        f"Pi documentation (read only when the user asks about pi itself, its SDK, extensions, themes, skills, or TUI):\n"
        f"- Main documentation: {readme_path}\n"
        f"- Additional docs: {docs_path}\n"
        f"- Examples: {examples_path} (extensions, custom tools, SDK)\n"
        f"- When reading pi docs or examples, resolve docs/... under Additional docs and examples/... under Examples, "
        f"not the current working directory\n"
        f"- When asked about: extensions (docs/extensions.md, examples/extensions/), themes (docs/themes.md), "
        f"skills (docs/skills.md), prompt templates (docs/prompt-templates.md), TUI components (docs/tui.md), "
        f"keybindings (docs/keybindings.md), SDK integrations (docs/sdk.md), custom providers (docs/custom-provider.md), "
        f"adding models (docs/models.md), pi packages (docs/packages.md), environment variables "
        f"(docs/environment-variables.md)\n"
        f"- When working on pi topics, read the docs and examples, and follow .md cross-references before implementing\n"
        f"- Always read pi .md files completely and follow links to related docs "
        f"(e.g., tui.md for TUI API details)"
    )

    if append_section:
        prompt += append_section

    # Append project context files
    if context_files:
        prompt += "\n\n<project_context>\n\n"
        prompt += "Project-specific instructions and guidelines:\n\n"
        for cf in context_files:
            prompt += f'<project_instructions path="{cf["path"]}">\n{cf["content"]}\n</project_instructions>\n\n'
        prompt += "</project_context>\n"

    # Append skills section (only if read tool is available)
    if has_read and skills:
        prompt += format_skills_for_prompt(skills)

    prompt += f"\nCurrent working directory: {prompt_cwd}"

    return prompt


__all__ = [
    "BuildSystemPromptOptions",
    "build_system_prompt",
]
