"""Branch summarization for tree navigation.

When navigating to a different point in the session tree, this generates
a summary of the branch being left so context isn't lost.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, cast

from pi_agent.types import (
    AgentMessage,
    StreamFn,
    Usage,
)
from pi_ai.types import (
    Context,
    Model,
    SimpleStreamOptions,
    TextContent,
    UserMessage,
)
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy
from pi_ai.utils.text import content_text
from pydantic import BaseModel, ConfigDict
from pydantic.functional_validators import SkipValidation

from ..messages import (
    convert_to_llm,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
)
from ..session_manager import SessionManager
from .compaction import complete_summarization, estimate_tokens
from .utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

# ============================================================================
# Types
# ============================================================================


class BranchSummaryResult(BaseModel):
    """Branch summary result."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str | None = None
    usage: Usage | None = None
    read_files: list[str] = []
    modified_files: list[str] = []
    aborted: bool = False
    error: str | None = None


class BranchSummaryDetails(BaseModel):
    """Details stored in BranchSummaryEntry.details for file tracking."""

    read_files: list[str] = []
    modified_files: list[str] = []


class BranchPreparation(BaseModel):
    """Branch preparation data."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    messages: list[AgentMessage]
    file_ops: FileOperations = FileOperations()
    total_tokens: int = 0


class CollectEntriesResult(BaseModel):
    """Result of collecting entries for branch summary."""

    entries: list[Any] = []  # SessionEntry
    common_ancestor_id: str | None = None


class GenerateBranchSummaryOptions(BaseModel):
    """Options for generating a branch summary."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: Model
    api_key: str | None = None
    headers: dict[str, str] | None = None
    env: dict[str, str] | None = None
    signal: object = None
    custom_instructions: str | None = None
    replace_instructions: bool = False
    reserve_tokens: int = 16384
    stream_fn: Annotated[StreamFn | None, SkipValidation()] = None
    retry: RetryPolicy | None = None
    callbacks: RetryCallbacks | None = None


# ============================================================================
# Entry Collection
# ============================================================================


def collect_entries_for_branch_summary(
    session: SessionManager,
    old_leaf_id: str | None,
    target_id: str,
) -> CollectEntriesResult:
    """Collect entries that should be summarized when navigating from one position to another.

    Walks from oldLeafId back to the common ancestor with targetId, collecting entries
    along the way. Does NOT stop at compaction boundaries - those are included and their
    summaries become context.

    Args:
        session: Session manager (read-only access)
        old_leaf_id: Current position (where we're navigating from)
        target_id: Target position (where we're navigating to)

    Returns:
        Entries to summarize and the common ancestor
    """
    # If no old position, nothing to summarize
    if old_leaf_id is None:
        return CollectEntriesResult()

    # Find common ancestor (deepest node that's on both paths)
    old_path_ids = {e.id for e in session.get_branch(old_leaf_id)}
    target_path = session.get_branch(target_id)

    # target_path is root-first, so iterate backwards to find deepest common ancestor
    common_ancestor_id: str | None = None
    for i in range(len(target_path) - 1, -1, -1):
        if target_path[i].id in old_path_ids:
            common_ancestor_id = target_path[i].id
            break

    # Collect entries from old leaf back to common ancestor
    entries: list[Any] = []
    current: str | None = old_leaf_id

    while current is not None and current != common_ancestor_id:
        entry = session.get_entry(current)
        if entry is None:
            break
        entries.append(entry)
        current = entry.parent_id

    # Reverse to get chronological order
    entries.reverse()

    return CollectEntriesResult(entries=entries, common_ancestor_id=common_ancestor_id)


# ============================================================================
# Entry to Message Conversion
# ============================================================================


def _get_message_from_entry(entry: Any) -> AgentMessage | None:
    """Extract AgentMessage from a session entry.

    Similar to getMessageFromEntry in compaction.py but also handles compaction entries.
    """
    entry_type = getattr(entry, "type", None)

    if entry_type == "message":
        # Skip tool results - context is in assistant's tool call
        msg = getattr(entry, "message", None)
        if msg is not None and getattr(msg, "role", None) == "toolResult":
            return None
        return msg

    if entry_type == "custom_message":
        return create_custom_message(
            getattr(entry, "custom_type", ""),
            getattr(entry, "content", ""),
            getattr(entry, "display", True),
            getattr(entry, "details", None),
            getattr(entry, "timestamp", 0),
        )

    if entry_type == "branch_summary":
        return create_branch_summary_message(
            getattr(entry, "summary", ""),
            getattr(entry, "from_id", ""),
            getattr(entry, "timestamp", 0),
        )

    if entry_type == "compaction":
        return create_compaction_summary_message(
            getattr(entry, "summary", ""),
            getattr(entry, "tokens_before", 0),
            getattr(entry, "timestamp", 0),
        )

    # These don't contribute to conversation content
    return None


def prepare_branch_entries(
    entries: list[Any],
    token_budget: int = 0,
) -> BranchPreparation:
    """Prepare entries for summarization with token budget. 

    Walks entries from NEWEST to OLDEST, adding messages until we hit the token budget.
    This ensures we keep the most recent context when the branch is too long.

    Also collects file operations from:
    - Tool calls in assistant messages
    - Existing branch_summary entries' details (for cumulative tracking)

    Args:
        entries: Entries in chronological order
        token_budget: Maximum tokens to include (0 = no limit)
    """
    messages: list[AgentMessage] = []
    file_ops = create_file_ops()
    total_tokens = 0

    # First pass: collect file ops from ALL entries (even if they don't fit in token budget)
    # This ensures we capture cumulative file tracking from nested branch summaries
    # Only extract from pi-generated summaries (fromHook !== true), not extension-generated ones
    for entry in entries:
        entry_type = getattr(entry, "type", None)
        if entry_type == "branch_summary":
            from_hook = getattr(entry, "from_hook", None)
            details = getattr(entry, "details", None)
            if not from_hook and details is not None:
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

    # Second pass: walk from newest to oldest, adding messages until token budget
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        message = _get_message_from_entry(entry)
        if message is None:
            continue

        # Extract file ops from assistant messages (tool calls)
        extract_file_ops_from_message(message, file_ops)

        tokens = estimate_tokens(message)

        # Check budget before adding
        if token_budget > 0 and total_tokens + tokens > token_budget:
            # If this is a summary entry, try to fit it anyway as it's important context
            entry_type = getattr(entry, "type", None)
            if entry_type in ("compaction", "branch_summary"):
                if total_tokens < token_budget * 0.9:
                    messages.insert(0, message)
                    total_tokens += tokens
            # Stop - we've hit the budget
            break

        messages.insert(0, message)
        total_tokens += tokens

    return BranchPreparation(
        messages=messages,
        file_ops=file_ops,
        total_tokens=total_tokens,
    )


# ============================================================================
# Summary Generation
# ============================================================================

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


async def generate_branch_summary(
    entries: list[Any],
    options: GenerateBranchSummaryOptions,
) -> BranchSummaryResult:
    """Generate a summary of abandoned branch entries.

    Args:
        entries: Session entries to summarize (chronological order)
        options: Generation options
    """
    model = options.model

    # Token budget = context window minus reserved space for prompt + response
    context_window = getattr(model, "context_window", 0) or 128000
    token_budget = context_window - options.reserve_tokens

    preparation = prepare_branch_entries(entries, token_budget)

    if not preparation.messages:
        return BranchSummaryResult(summary="No content to summarize")

    # Transform to LLM-compatible messages, then serialize to text
    # Serialization prevents the model from treating it as a conversation to continue
    llm_messages = convert_to_llm(preparation.messages)
    conversation_text = serialize_conversation(llm_messages)

    # Build prompt
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

    # Call LLM for summarization. Prefer the session stream function so SDK
    # request behavior (timeouts, retries, attribution headers) stays consistent
    # without running through agent state/events. Retried via completeSummarization
    # so transient stream drops reuse the configured retry policy.
    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=cast("list[Any]", summarization_messages),
    )
    request_options = SimpleStreamOptions()
    if options.api_key:
        object.__setattr__(request_options, "api_key", options.api_key)
    if options.headers:
        request_options.headers = options.headers
    if options.env:
        object.__setattr__(request_options, "env", options.env)
    if options.signal is not None:
        object.__setattr__(request_options, "signal", options.signal)
    object.__setattr__(request_options, "max_tokens", 2048)

    response = await complete_summarization(
        model,
        context,
        request_options,
        stream_fn=options.stream_fn,
        retry=options.retry,
        callbacks=options.callbacks,
    )

    # Check if aborted or errored
    if response.stop_reason == "aborted":
        return BranchSummaryResult(aborted=True)
    if response.stop_reason == "error":
        return BranchSummaryResult(
            error=response.error_message or "Summarization failed"
        )

    summary = content_text(response.content)  # type: ignore[arg-type]

    # Prepend preamble to provide context about the branch summary
    summary = BRANCH_SUMMARY_PREAMBLE + summary

    # Compute file lists and append to summary
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return BranchSummaryResult(
        summary=summary or "No summary generated",
        usage=response.usage,
        read_files=read_files,
        modified_files=modified_files,
    )


__all__ = [
    "BranchPreparation",
    "BranchSummaryDetails",
    "BranchSummaryResult",
    "CollectEntriesResult",
    "GenerateBranchSummaryOptions",
    "collect_entries_for_branch_summary",
    "generate_branch_summary",
    "prepare_branch_entries",
]
