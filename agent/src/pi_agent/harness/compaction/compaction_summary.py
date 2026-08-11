"""摘要生成（对应 ``compaction.ts`` 的摘要生成 + LLM 提示词）。

包含：
- 4 组内置提示词模板
- ``generate_summary`` / ``generate_summary_with_usage``
- ``complete_simple_with_retries``（LLM 调用占位）
"""

from __future__ import annotations

from ...result import Result, err, ok
from ...types import (
    AgentMessage,
    AssistantMessage,
    Message,
    Model,
    TextContent,
    ThinkingLevel,
    Usage,
    UserMessage,
)
from ..messages import convert_to_llm
from ..types import CompactionError
from .utils import serialize_conversation

# ---------------------------------------------------------------------------
# 提示词模板
# ---------------------------------------------------------------------------

SUMMARIZATION_SYSTEM_PROMPT = """You are a context summarization assistant. Your task is to read a conversation between a user and an AI assistant, then produce a structured summary following the exact format specified.

Do NOT continue the conversation. Do NOT respond to any questions in the conversation. ONLY output the structured summary."""

SUMMARIZATION_PROMPT = """The messages above are a conversation to summarize. Create a structured context checkpoint summary that another LLM will use to continue the work.

Use this EXACT format:

## Goal
[What is the user trying to accomplish? Can be multiple items if the session covers different tasks.]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned by user]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Current work]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [Ordered list of what should happen next]

## Critical Context
- [Any data, examples, or references needed to continue]
- [Or "(none)" if not applicable]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

UPDATE_SUMMARIZATION_PROMPT = """The messages above are NEW conversation messages to incorporate into the existing summary provided in <previous-summary> tags.

Update the existing structured summary with new information. RULES:
- PRESERVE all existing information from the previous summary
- ADD new progress, decisions, and context from the new messages
- UPDATE the Progress section: move items from "In Progress" to "Done" when completed
- UPDATE "Next Steps" based on what was accomplished
- PRESERVE exact file paths, function names, and error messages
- If something is no longer relevant, you may remove it

Use this EXACT format:

## Goal
[Preserve existing goals, add new ones if the task expanded]

## Constraints & Preferences
- [Preserve existing, add new ones discovered]

## Progress
### Done
- [x] [Include previously done items AND newly completed items]

### In Progress
- [ ] [Current work - update based on progress]

### Blocked
- [Current blockers - remove if resolved]

## Key Decisions
- **[Decision]**: [Brief rationale] (preserve all previous, add new)

## Next Steps
1. [Update based on current state]

## Critical Context
- [Preserve important context, add new if needed]

Keep each section concise. Preserve exact file paths, function names, and error messages."""

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


# ---------------------------------------------------------------------------
# LLM 调用占位
# ---------------------------------------------------------------------------


async def complete_simple_with_retries(
    model: Model,
    messages: list[Message],
    system_prompt: str,
    max_tokens: int,
) -> AssistantMessage:
    """LLM 调用包装（对应 TS ``completeSimpleWithRetries``）。

    TODO：当前为占位实现，待 pi-ai Python 层就绪后对接真实的
    ``models.completeSimple`` + ``retryAssistantCall``。
    """
    raise NotImplementedError(
        "complete_simple_with_retries requires pi-ai backend. "
        "This is a placeholder for the compaction module."
    )


# ---------------------------------------------------------------------------
# 摘要生成
# ---------------------------------------------------------------------------


def _content_text(content: object) -> str:
    """提取 content 列表中的纯文本。"""
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for block in content:
        if isinstance(block, dict):
            if block.get("type") == "text" and isinstance(block.get("text"), str):
                parts.append(block["text"])
        elif hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return "".join(parts)


async def generate_summary_with_usage(
    current_messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    """生成摘要并返回 usage（对应 TS ``generateSummaryWithUsage``）。

    TODO：当前为占位实现，待 pi-ai Python 层就绪后对接。
    """
    max_tokens = min(int(0.8 * reserve_tokens), 4096)

    base_prompt = (
        UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)

    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += base_prompt

    import time

    summarization_messages: list[Message] = [
        UserMessage(
            content=[TextContent(text=prompt_text)], timestamp=int(time.time() * 1000)
        ),
    ]

    try:
        response = await complete_simple_with_retries(
            model,
            summarization_messages,
            SUMMARIZATION_SYSTEM_PROMPT,
            max_tokens,
        )
    except NotImplementedError:
        return err(CompactionError("pi-ai backend not available"))

    if response.stop_reason == "aborted":
        return err(CompactionError(response.error_message or "Summarization aborted"))
    if response.stop_reason == "error":
        return err(
            CompactionError(
                f"Summarization failed: {response.error_message or 'Unknown error'}",
            )
        )

    text = _content_text(response.content)
    return ok((text, response.usage or Usage()))


async def generate_summary(
    current_messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    previous_summary: str | None = None,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> Result[str, CompactionError]:
    """生成摘要文本（对应 TS ``generateSummary``）。"""
    result = await generate_summary_with_usage(
        current_messages,
        model,
        reserve_tokens,
        previous_summary=previous_summary,
        custom_instructions=custom_instructions,
        thinking_level=thinking_level,
    )
    if not result.is_ok():
        return err(result.error)
    return ok(result.value[0])
