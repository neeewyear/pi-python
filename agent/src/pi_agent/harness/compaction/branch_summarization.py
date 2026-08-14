"""分支摘要。

包含：
- ``collect_entries_for_branch_summary`` / ``prepare_branch_entries``
- ``generate_branch_summary``
- 相关数据模型：``BranchSummaryResult`` / ``BranchPreparation`` / ``CollectEntriesResult`` 等
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

from ...result import Result, err, ok
from ...types import AgentMessage, TextContent, Usage, UserMessage
from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
)
from pi_session.types import BranchSummaryEntry, Entry, SessionError
from ..types import BranchSummaryError

if TYPE_CHECKING:
    from pi_session.session import Session
from .compaction_summary import (
    SUMMARIZATION_SYSTEM_PROMPT,
    complete_simple_with_retries,
)
from .compaction_token import estimate_tokens
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class BranchSummaryDetails(BaseModel):
    """分支摘要详情"""

    read_files: list[str] = []
    modified_files: list[str] = []


class BranchSummaryResult(BaseModel):
    """分支摘要结果"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    usage: Usage | None = None
    read_files: list[str] = []
    modified_files: list[str] = []


class BranchPreparation(BaseModel):
    """分支准备数据"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[AgentMessage]
    file_ops: FileOperations = FileOperations()
    total_tokens: int = 0


class CollectEntriesResult(BaseModel):
    """收集条目结果"""

    entries: list[Entry] = []
    common_ancestor_id: str | None = None


class GenerateBranchSummaryOptions(BaseModel):
    """分支摘要选项"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: object = None  # Model | None（避免 Pydantic isinstance 校验）
    reserve_tokens: int = 16384
    custom_instructions: str | None = None
    replace_instructions: bool = False


# ---------------------------------------------------------------------------
# 提示词
# ---------------------------------------------------------------------------

BRANCH_SUMMARY_PREAMBLE = """The user explored a different conversation branch before returning here.
Summary of that exploration:

"""

BRANCH_SUMMARY_PROMPT = """Create a structured summary of this conversation branch for context when returning later.

Use this EXACT format:

## Goal
[What was the user trying to accomplish in this branch?]

## Constraints & Preferences
- [Any constraints, preferences, or requirements mentioned]
- [Or "(none)" if none were mentioned]

## Progress
### Done
- [x] [Completed tasks/changes]

### In Progress
- [ ] [Work that was started but not finished]

### Blocked
- [Issues preventing progress, if any]

## Key Decisions
- **[Decision]**: [Brief rationale]

## Next Steps
1. [What should happen next to continue this work]

Keep each section concise. Preserve exact file paths, function names, and error messages."""


# ---------------------------------------------------------------------------
# collectEntriesForBranchSummary
# ---------------------------------------------------------------------------


async def collect_entries_for_branch_summary(
    session: Session,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    """收集分支摘要条目"""
    if old_leaf_id is None:
        return CollectEntriesResult()

    old_path = set(
        await session.find_entries_on_branch({"start": old_leaf_id})  # type: ignore[arg-type]
    )
    old_path_ids = {e.id for e in old_path}
    target_path = await session.find_entries_on_branch({"start": target_id})  # type: ignore[arg-type]

    common_ancestor_id: str | None = None
    for entry in target_path:
        if entry.id in old_path_ids:
            common_ancestor_id = entry.id
            break

    entries: list[Entry] = []
    current: str | None = old_leaf_id

    while current is not None and current != common_ancestor_id:
        entry = await session.get_entry(current)  # type: ignore[assignment]
        if entry is None:
            raise SessionError("invalid_entry", f"Entry {current} not found")
        entries.append(entry)
        current = entry.parent_id

    entries.reverse()
    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


# ---------------------------------------------------------------------------
# prepareBranchEntries
# ---------------------------------------------------------------------------


def _get_message_from_entry(entry: Entry) -> AgentMessage | None:
    """从条目获取消息"""
    if entry.type == "message":
        if entry.message.role == "toolResult":
            return None
        return entry.message
    if entry.type == "branch_summary":
        if isinstance(entry, BranchSummaryEntry):
            return create_branch_summary_message(
                entry.summary, entry.from_id, entry.timestamp
            )
        return None
    if entry.type == "compaction":
        return create_compaction_summary_message(
            entry.summary, entry.tokens_before, entry.timestamp
        )
    return None


def prepare_branch_entries(
    entries: list[Entry],
    token_budget: int = 0,
) -> BranchPreparation:
    """准备分支条目"""
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0

    # 先收集已有 branch_summary 的文件操作
    for entry in entries:
        if entry.type == "branch_summary" and entry.details is not None:
            details = entry.details
            if isinstance(details, dict):
                read_files = details.get("read_files", [])
                modified_files = details.get("modified_files", [])
                if isinstance(read_files, list):
                    for f in read_files:
                        if isinstance(f, str):
                            file_ops.read.add(f)
                if isinstance(modified_files, list):
                    for f in modified_files:
                        if isinstance(f, str):
                            file_ops.edited.add(f)

    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = _get_message_from_entry(entry)
        if message is None:
            continue

        extract_file_ops_from_message(message, file_ops)
        tokens = estimate_tokens(message)

        if token_budget > 0 and total_tokens + tokens > token_budget:
            if entry.type in ("compaction", "branch_summary"):
                if total_tokens < token_budget * 0.9:
                    messages.insert(0, message)
                    total_tokens += tokens
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(
        messages=messages,
        file_ops=file_ops,
        total_tokens=total_tokens,
    )


# ---------------------------------------------------------------------------
# generateBranchSummary
# ---------------------------------------------------------------------------


def _content_text(content: list[Any]) -> str:
    parts: list[str] = []
    for block in content:
        if hasattr(block, "text"):
            parts.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            parts.append(str(block.get("text", "")))
    return "".join(parts)


async def generate_branch_summary(
    entries: list[Entry],
    options: GenerateBranchSummaryOptions,
) -> Result[BranchSummaryResult, BranchSummaryError]:
    """生成分支摘要"""
    model = options.model
    if model is None:
        return err(BranchSummaryError("No model provided"))

    context_window = 128000
    token_budget = context_window - options.reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)

    if not preparation.messages:
        return ok(
            BranchSummaryResult(
                summary="No content to summarize",
                read_files=[],
                modified_files=[],
            )
        )

    llm_messages = convert_to_llm(preparation.messages)
    conversation_text = serialize_conversation(llm_messages)

    if options.replace_instructions and options.custom_instructions:
        instructions = options.custom_instructions
    elif options.custom_instructions:
        instructions = f"{BRANCH_SUMMARY_PROMPT}\n\nAdditional focus: {options.custom_instructions}"
    else:
        instructions = BRANCH_SUMMARY_PROMPT

    prompt_text = (
        f"<conversation>\n{conversation_text}\n</conversation>\n\n{instructions}"
    )

    summarization_messages = [
        UserMessage(
            content=[TextContent(text=prompt_text)],
            timestamp=int(time.time() * 1000),
        )
    ]

    try:
        response = await complete_simple_with_retries(
            model,  # type: ignore[arg-type]
            summarization_messages,  # type: ignore[arg-type]
            SUMMARIZATION_SYSTEM_PROMPT,
            2048,
        )
    except NotImplementedError:
        return err(BranchSummaryError("pi-ai backend not available"))

    if response.stop_reason == "aborted":
        return err(
            BranchSummaryError(
                response.error_message or "Branch summary aborted",
            )
        )
    if response.stop_reason == "error":
        return err(
            BranchSummaryError(
                f"Branch summary failed: {response.error_message or 'Unknown error'}",
            )
        )

    summary = BRANCH_SUMMARY_PREAMBLE + _content_text(response.content)
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        BranchSummaryResult(
            summary=summary or "No summary generated",
            usage=response.usage,
            read_files=read_files,
            modified_files=modified_files,
        )
    )
