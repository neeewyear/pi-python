"""自定义消息与 LLM 转换（对应 ``harness/messages.ts``）。

将四种自定义消息（bashExecution / custom / branchSummary / compactionSummary）
转换为 LLM 可理解的 ``user`` 消息；标准消息原样透传。
"""

from __future__ import annotations

from typing import TypeAlias

from ..types import (
    AgentMessage,
    AssistantMessage,
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
    ImageContent,
    Message,
    TextContent,
    ToolResultMessage,
    UserMessage,
)

COMPACTION_SUMMARY_PREFIX = (
    "The conversation history before this point was compacted into the following summary:\n\n<summary>\n"
)
COMPACTION_SUMMARY_SUFFIX = "\n</summary>"

BRANCH_SUMMARY_PREFIX = (
    "The following is a summary of a branch that this conversation came back from:\n\n<summary>\n"
)
BRANCH_SUMMARY_SUFFIX = "</summary>"

CustomContent: TypeAlias = str | list[TextContent | ImageContent]
"""自定义消息内容（字符串或内容块数组）。"""


def bash_execution_to_text(msg: BashExecutionMessage) -> str:
    """把 bash 执行消息渲染为纯文本（用于转换进 LLM 上下文）。"""
    text = f"Ran `{msg.command}`\n"
    if msg.output:
        text += f"```\n{msg.output}\n```"
    else:
        text += "(no output)"
    if msg.cancelled:
        text += "\n\n(command cancelled)"
    elif msg.exit_code is not None and msg.exit_code != 0:
        text += f"\n\nCommand exited with code {msg.exit_code}"
    if msg.truncated and msg.full_output_path:
        text += f"\n\n[Output truncated. Full output: {msg.full_output_path}]"
    return text


def _normalize_timestamp(timestamp: str | int) -> int:
    """把 ISO 字符串或毫秒时间戳统一为毫秒时间戳。"""
    if isinstance(timestamp, int):
        return timestamp
    from datetime import datetime

    return int(datetime.fromisoformat(timestamp.replace("Z", "+00:00")).timestamp() * 1000)


def create_branch_summary_message(summary: str, from_id: str, timestamp: str | int) -> BranchSummaryMessage:
    """创建分支摘要消息。"""
    return BranchSummaryMessage(summary=summary, from_id=from_id, timestamp=_normalize_timestamp(timestamp))


def create_compaction_summary_message(summary: str, tokens_before: int, timestamp: str | int) -> CompactionSummaryMessage:
    """创建上下文压缩摘要消息。"""
    return CompactionSummaryMessage(
        summary=summary, tokens_before=tokens_before, timestamp=_normalize_timestamp(timestamp)
    )


def create_custom_message(
    custom_type: str,
    content: CustomContent,
    display: bool,
    details: object | None,
    timestamp: str | int,
) -> CustomMessage:
    """创建应用自定义消息。"""
    return CustomMessage(
        custom_type=custom_type,
        content=content,
        display=display,
        details=details,
        timestamp=_normalize_timestamp(timestamp),
    )


def convert_to_llm(messages: list[AgentMessage]) -> list[Message]:
    """把 ``AgentMessage[]`` 转换为 LLM 兼容的 ``Message[]``。

    契约：**不抛异常**；无法转换的消息被过滤掉。
    """
    converted: list[Message] = []
    for msg in messages:
        if isinstance(msg, BashExecutionMessage):
            if msg.exclude_from_context:
                continue
            converted.append(
                UserMessage(content=[TextContent(text=bash_execution_to_text(msg))], timestamp=msg.timestamp)
            )
        elif isinstance(msg, CustomMessage):
            content = (
                [TextContent(text=msg.content)]
                if isinstance(msg.content, str)
                else msg.content
            )
            converted.append(UserMessage(content=content, timestamp=msg.timestamp))
        elif isinstance(msg, BranchSummaryMessage):
            converted.append(
                UserMessage(
                    content=[TextContent(text=BRANCH_SUMMARY_PREFIX + msg.summary + BRANCH_SUMMARY_SUFFIX)],
                    timestamp=msg.timestamp,
                )
            )
        elif isinstance(msg, CompactionSummaryMessage):
            converted.append(
                UserMessage(
                    content=[TextContent(text=COMPACTION_SUMMARY_PREFIX + msg.summary + COMPACTION_SUMMARY_SUFFIX)],
                    timestamp=msg.timestamp,
                )
            )
        elif isinstance(msg, (UserMessage, AssistantMessage, ToolResultMessage)):
            converted.append(msg)
    return converted
