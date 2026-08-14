"""Context compaction for long sessions.

Pure functions for compaction logic. The session manager handles I/O,
and after compaction the session is reloaded.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

from pi_agent.types import (
    AgentMessage,
    AssistantMessage,
    StreamFn,
    ThinkingLevel,
    Usage,
)
from pi_ai.compat import complete_simple
from pi_ai.types import Context, Model, SimpleStreamOptions
from pi_ai.utils.retry import RetryCallbacks, RetryPolicy, retry_assistant_call
from pi_ai.utils.text import content_text
from pi_ai.utils.uuid import uuidv7
from pydantic import BaseModel, ConfigDict

from ..messages import convert_to_llm
from ..session_manager import CompactionEntry, SessionEntry
from .utils import (
    SUMMARIZATION_SYSTEM_PROMPT,
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

if TYPE_CHECKING:
    from pi_session.context import (
        build_session_context as _build_session_context,
    )
    from pi_session.context import (
        session_entry_to_context_messages as _session_entry_to_context_messages,
    )
else:
    from pi_session.context import (
        build_session_context as _build_session_context,  # type: ignore[import-not-found]
    )
    from pi_session.context import (  # type: ignore[import-not-found]
        session_entry_to_context_messages as _session_entry_to_context_messages,
    )


# ============================================================================
# File Operation Tracking
# ============================================================================


class CompactionDetails(BaseModel):
    """Details stored in CompactionEntry.details for file tracking."""

    read_files: list[str] = []
    modified_files: list[str] = []


def _extract_file_operations(
    messages: list[AgentMessage],
    entries: list[SessionEntry],
    prev_compaction_index: int,
) -> FileOperations:
    """Extract file operations from messages and previous compaction entries."""
    file_ops = create_file_ops()

    # Collect from previous compaction's details (if pi-generated)
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if isinstance(prev_compaction, CompactionEntry):
            if not prev_compaction.from_hook and prev_compaction.details is not None:
                details = prev_compaction.details
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

    # Extract from tool calls in messages
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)

    return file_ops


# ============================================================================
# Message Extraction
# ============================================================================


def _get_message_from_entry_for_compaction(entry: SessionEntry) -> AgentMessage | None:
    """Extract AgentMessage from an entry if it produces one.

    Returns None for entries that don't contribute to LLM context.
    """
    if isinstance(entry, CompactionEntry):
        return None
    messages = _session_entry_to_context_messages(entry, 0, [entry])  # type: ignore[list-item, arg-type]
    return messages[0] if messages else None


# ============================================================================
# Types
# ============================================================================


@dataclass
class SummaryResult:
    """Result from generate_summary_with_usage."""

    text: str
    usage: Usage | None


class CompactionResult(BaseModel):
    """Result from compact() - SessionManager adds uuid/parentUuid when saving."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    first_kept_entry_id: str
    tokens_before: int
    estimated_tokens_after: int | None = None
    usage: Usage | None = None
    details: object | None = None


def _combine_usage(first: Usage, second: Usage) -> Usage:
    """Combine two Usage objects."""
    total_tokens = first.total_tokens + second.total_tokens
    cost = first.cost.model_copy(
        update={
            "input": first.cost.input + second.cost.input,
            "output": first.cost.output + second.cost.output,
            "cache_read": first.cost.cache_read + second.cost.cache_read,
            "cache_write": first.cost.cache_write + second.cost.cache_write,
            "total": first.cost.total + second.cost.total,
        }
    )
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        total_tokens=total_tokens,
        cost=cost,
    )


class CompactionSettings(BaseModel):
    """Compaction threshold settings."""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# ============================================================================
# Token calculation
# ============================================================================


def calculate_context_tokens(usage: Usage) -> int:
    """Calculate total context tokens from usage.

    Uses the native totalTokens field when available, falls back to computing from components.
    """
    return (
        usage.total_tokens
        or usage.input + usage.output + usage.cache_read + usage.cache_write
    )


def _get_assistant_usage(msg: AgentMessage) -> Usage | None:
    """Get usage from an assistant message if available.

    Skips aborted, error, and all-zero usage messages as they don't have valid usage data.
    """
    if msg.role == "assistant" and isinstance(msg, AssistantMessage):
        if (
            msg.stop_reason not in ("aborted", "error")
            and msg.usage is not None
            and calculate_context_tokens(msg.usage) > 0
        ):
            return msg.usage
    return None


def get_last_assistant_usage(entries: list[SessionEntry]) -> Usage | None:
    """Find the last valid assistant message usage from session entries."""
    for i in range(len(entries) - 1, -1, -1):
        entry = entries[i]
        if isinstance(entry, CompactionEntry):
            continue
        messages = _session_entry_to_context_messages(entry, 0, [entry])  # type: ignore[list-item, arg-type]
        if messages:
            usage = _get_assistant_usage(messages[0])
            if usage:
                return usage
    return None


class ContextUsageEstimate(BaseModel):
    """Context usage estimate."""

    tokens: int
    usage_tokens: int
    trailing_tokens: int
    last_usage_index: int | None = None


def _get_last_assistant_usage_info(
    messages: list[AgentMessage],
) -> tuple[Usage, int] | None:
    """Get last assistant usage info from messages."""
    for i in range(len(messages) - 1, -1, -1):
        usage = _get_assistant_usage(messages[i])
        if usage:
            return usage, i
    return None


def estimate_context_tokens(messages: list[AgentMessage]) -> ContextUsageEstimate:
    """Estimate context tokens from messages.

    Uses the last assistant usage when available.
    If there are messages after the last usage, estimate their tokens with estimate_tokens.
    """
    usage_info = _get_last_assistant_usage_info(messages)

    if usage_info is None:
        estimated = 0
        for message in messages:
            estimated += estimate_tokens(message)
        return ContextUsageEstimate(
            tokens=estimated,
            usage_tokens=0,
            trailing_tokens=estimated,
            last_usage_index=None,
        )

    usage, usage_index = usage_info
    usage_tokens = calculate_context_tokens(usage)
    trailing_tokens = 0
    for i in range(usage_index + 1, len(messages)):
        trailing_tokens += estimate_tokens(messages[i])

    return ContextUsageEstimate(
        tokens=usage_tokens + trailing_tokens,
        usage_tokens=usage_tokens,
        trailing_tokens=trailing_tokens,
        last_usage_index=usage_index,
    )


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings,
) -> bool:
    """Check if compaction should trigger based on context usage."""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ============================================================================
# Cut point detection
# ============================================================================

ESTIMATED_IMAGE_CHARS = 4800


def _estimate_text_and_image_content_chars(
    content: str | list[dict[str, object]],
) -> int:
    """Estimate character count for text and image content."""
    if isinstance(content, str):
        return len(content)

    chars = 0
    for block in content:
        if isinstance(block, dict):
            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if isinstance(text, str):
                    chars += len(text)
            elif block_type == "image":
                chars += ESTIMATED_IMAGE_CHARS
    return chars


def estimate_tokens(message: AgentMessage) -> int:
    """Estimate token count for a message using chars/4 heuristic.

    This is conservative (overestimates tokens).
    """
    chars = 0

    if message.role == "user":
        # UserMessage has content field
        content = getattr(message, "content", "")
        if isinstance(content, (str, list)):
            chars = _estimate_text_and_image_content_chars(content)
        return max(1, math.ceil(chars / 4))

    if message.role == "assistant" and isinstance(message, AssistantMessage):
        # Thinking blocks are stored in a separate field, not in content
        if message.thinking:
            for tb in message.thinking:
                chars += len(tb.text)
        for block in message.content:
            if block.type == "text":
                chars += len(block.text)
            elif block.type == "toolCall":
                import json as _json

                chars += len(block.name) + len(_json.dumps(block.args))
        return max(1, math.ceil(chars / 4))

    if message.role in ("custom", "toolResult"):
        content = getattr(message, "content", "")
        if isinstance(content, (str, list)):
            chars = _estimate_text_and_image_content_chars(content)
        return max(1, math.ceil(chars / 4))

    if message.role == "bashExecution":
        chars = len(getattr(message, "command", "")) + len(
            getattr(message, "output", "")
        )
        return max(1, math.ceil(chars / 4))

    if message.role in ("branchSummary", "compactionSummary"):
        chars = len(getattr(message, "summary", ""))
        return max(1, math.ceil(chars / 4))

    return 0


def _is_cut_point_message(message: AgentMessage) -> bool:
    """Check if message is a valid cut point."""
    return message.role in (
        "user",
        "assistant",
        "bashExecution",
        "custom",
        "branchSummary",
        "compactionSummary",
    )


def _is_turn_start_message(message: AgentMessage) -> bool:
    """Check if message starts a turn."""
    return message.role in (
        "user",
        "bashExecution",
        "custom",
        "branchSummary",
        "compactionSummary",
    )


def _is_turn_start_entry(entry: SessionEntry) -> bool:
    """Check if entry starts a turn."""
    if isinstance(entry, CompactionEntry):
        return False
    messages = _session_entry_to_context_messages(entry, 0, [entry])  # type: ignore[list-item, arg-type]
    return any(_is_turn_start_message(m) for m in messages)


def _find_valid_cut_points(
    entries: list[SessionEntry],
    start_index: int,
    end_index: int,
) -> list[int]:
    """Find valid cut points.

    Never cut at tool results (they must follow their tool call).
    When we cut at an assistant message with tool calls, its tool results follow it
    and will be kept.
    """
    cut_points: list[int] = []
    for i in range(start_index, end_index):
        entry = entries[i]
        if isinstance(entry, CompactionEntry):
            continue
        messages = _session_entry_to_context_messages(entry, 0, [entry])  # type: ignore[list-item, arg-type]
        if any(_is_cut_point_message(m) for m in messages):
            cut_points.append(i)
    return cut_points


def find_turn_start_index(
    entries: list[SessionEntry],
    entry_index: int,
    start_index: int,
) -> int:
    """Find the context-visible user-role message that starts the turn containing the given entry index.

    Returns -1 if no turn start found before the index.
    """
    for i in range(entry_index, start_index - 1, -1):
        if _is_turn_start_entry(entries[i]):
            return i
    return -1


class CutPointResult(BaseModel):
    """Cut point result."""

    first_kept_entry_index: int
    """Index of first entry to keep."""
    turn_start_index: int = -1
    """Index of user message that starts the turn being split, or -1 if not splitting."""
    is_split_turn: bool = False
    """Whether this cut splits a turn (cut point is not a user message)."""


def find_cut_point(
    entries: list[SessionEntry],
    start_index: int,
    end_index: int,
    keep_recent_tokens: int,
    ) -> CutPointResult:
    """Find the cut point in session entries that keeps approximately keepRecentTokens.

    Algorithm: Walk backwards from newest, accumulating estimated message sizes.
    Stop when we've accumulated >= keepRecentTokens. Cut at that point.

    Can cut at user OR assistant messages (never tool results). When cutting at an
    assistant message with tool calls, its tool results come after and will be kept.

    Only considers entries between startIndex and endIndex (exclusive).
    """
    cut_points = _find_valid_cut_points(entries, start_index, end_index)

    if not cut_points:
        return CutPointResult(
            first_kept_entry_index=start_index, turn_start_index=-1, is_split_turn=False
        )

    # Walk backwards from newest, accumulating estimated message sizes
    accumulated_tokens = 0
    cut_index = cut_points[0]  # Default: keep from first message (not header)

    for i in range(end_index - 1, start_index - 1, -1):
        entry = entries[i]
        messages = _session_entry_to_context_messages(entry, 0, [entry])  # type: ignore[list-item, arg-type]
        message_tokens = sum(estimate_tokens(m) for m in messages)
        if message_tokens == 0:
            continue
        accumulated_tokens += message_tokens

        # Check if we've exceeded the budget
        if accumulated_tokens >= keep_recent_tokens:
            # Find the closest valid cut point at or after this entry
            for c in cut_points:
                if c >= i:
                    cut_index = c
                    break
            break

    # Scan backwards from cutIndex to include adjacent metadata entries that do not affect context.
    while cut_index > start_index:
        prev_entry = entries[cut_index - 1]
        # Stop at compaction boundaries or context-visible entries.
        if isinstance(prev_entry, CompactionEntry):
            break
        prev_messages = _session_entry_to_context_messages(prev_entry, 0, [prev_entry])  # type: ignore[list-item, arg-type]
        if prev_messages:
            break
        cut_index -= 1

    # Determine if this is a split turn
    cut_entry = entries[cut_index]
    starts_turn = _is_turn_start_entry(cut_entry)
    turn_start_idx = (
        -1 if starts_turn else find_turn_start_index(entries, cut_index, start_index)
    )

    return CutPointResult(
        first_kept_entry_index=cut_index,
        turn_start_index=turn_start_idx,
        is_split_turn=not starts_turn and turn_start_idx != -1,
    )


# ============================================================================
# Summarization
# ============================================================================

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


def _create_summarization_options(
    model: Model,
    max_tokens: int,
    *,
    headers: dict[str, str] | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> SimpleStreamOptions:
    """Create summarization options."""
    options = SimpleStreamOptions()
    # SimpleStreamOptions doesn't have max_tokens, so we use a workaround
    # by setting extra fields that the compat layer can access via getattr
    object.__setattr__(options, "max_tokens", max_tokens)
    if headers:
        options.headers = headers
    return options


async def _complete_with_stream_fn(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    stream_fn: StreamFn | None,
) -> AssistantMessage:
    """Complete with optional streamFn."""
    if stream_fn is not None:
        # StreamFn returns AsyncIterator[AssistantMessageEvent]
        # We need to collect events to build the final message
        from pi_ai.types import (
            AssistantAbortedEvent,
            AssistantErrorEvent,
            AssistantMessageSnapshot,
            AssistantStreamEnd,
        )

        last_message: AssistantMessage | None = None
        stream = stream_fn(model, context, options)
        async for event in stream:
            if isinstance(event, (AssistantMessageSnapshot, AssistantStreamEnd)):
                if event.message is not None:
                    last_message = event.message
            elif isinstance(event, AssistantAbortedEvent):
                # Build an aborted message
                msg = last_message
                if msg is not None:
                    msg.stop_reason = "aborted"
                    return msg
                # Fallback: create a minimal message
                return AssistantMessage(
                    content=[],
                    api=getattr(model, "api", ""),
                    provider=getattr(model, "provider", ""),
                    model=getattr(model, "model_id", ""),
                    stop_reason="aborted",
                    error_message=event.error,
                    timestamp=int(time.time() * 1000),
                )
            elif isinstance(event, AssistantErrorEvent):
                return AssistantMessage(
                    content=[],
                    api=getattr(model, "api", ""),
                    provider=getattr(model, "provider", ""),
                    model=getattr(model, "model_id", ""),
                    stop_reason="error",
                    error_message=event.error.error_message or "Stream error",
                    timestamp=int(time.time() * 1000),
                )
        if last_message is not None:
            return last_message
        return AssistantMessage(
            content=[],
            api=getattr(model, "api", ""),
            provider=getattr(model, "provider", ""),
            model=getattr(model, "model_id", ""),
            stop_reason="error",
            error_message="No response from stream",
            timestamp=int(time.time() * 1000),
        )
    return await complete_simple(model, context, options)


async def complete_summarization(
    model: Model,
    context: Context,
    options: SimpleStreamOptions,
    stream_fn: StreamFn | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> AssistantMessage:
    """Shared choke point for every compaction/branch-summary summarization call (corresponds to TS ``completeSummarization``).

    Wraps the single LLM call in retry_assistant_call so transient stream drops
    honor the configured retry policy instead of failing the whole compaction on
    the first attempt.
    """
    # Summaries are standalone requests, so isolate routing and avoid cache writes that cannot be reused.
    request_options = options.model_copy()
    request_options.cache_retention = "none"
    request_options.session_id = uuidv7()

    async def _produce() -> AssistantMessage:
        return await _complete_with_stream_fn(
            model, context, request_options, stream_fn
        )

    signal = getattr(options, "signal", None)
    return await retry_assistant_call(_produce, retry, signal, callbacks)


async def generate_summary(
    current_messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    signal: object = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    stream_fn: StreamFn | None = None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> str:
    """Generate a summary of the conversation using the LLM.

    If previousSummary is provided, uses the update prompt to merge.
    """
    result = await generate_summary_with_usage(
        current_messages,
        model,
        reserve_tokens,
        api_key=api_key,
        headers=headers,
        signal=signal,
        custom_instructions=custom_instructions,
        previous_summary=previous_summary,
        thinking_level=thinking_level,
        stream_fn=stream_fn,
        env=env,
        retry=retry,
        callbacks=callbacks,
    )
    return result.text


async def generate_summary_with_usage(
    current_messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    signal: object = None,
    custom_instructions: str | None = None,
    previous_summary: str | None = None,
    thinking_level: ThinkingLevel | None = None,
    stream_fn: StreamFn | None = None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> SummaryResult:
    """Generate or update a conversation summary and return its provider usage."""
    model_max_tokens = getattr(model, "max_tokens", 0) or 0
    max_tokens = min(
        int(0.8 * reserve_tokens),
        model_max_tokens if model_max_tokens > 0 else 2147483647,
    )

    # Use update prompt if we have a previous summary, otherwise initial prompt
    base_prompt = (
        UPDATE_SUMMARIZATION_PROMPT if previous_summary else SUMMARIZATION_PROMPT
    )
    if custom_instructions:
        base_prompt = f"{base_prompt}\n\nAdditional focus: {custom_instructions}"

    # Serialize conversation to text so model doesn't try to continue it
    # Convert to LLM messages first (handles custom types like bashExecution, custom, etc.)
    llm_messages = convert_to_llm(current_messages)
    conversation_text = serialize_conversation(llm_messages)

    # Build the prompt with conversation wrapped in tags
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n"
    if previous_summary:
        prompt_text += (
            f"<previous-summary>\n{previous_summary}\n</previous-summary>\n\n"
        )
    prompt_text += base_prompt

    summarization_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": int(time.time() * 1000),
        }
    ]

    # Build context
    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=cast("list[Any]", summarization_messages),
    )
    if (
        thinking_level
        and thinking_level != "off"
        and getattr(model, "reasoning", False)
    ):
        context.thinking_level = thinking_level

    completion_options = _create_summarization_options(
        model,
        max_tokens,
        headers=headers,
        thinking_level=thinking_level,
    )
    if api_key:
        object.__setattr__(completion_options, "api_key", api_key)
    if env:
        object.__setattr__(completion_options, "env", env)
    if signal is not None:
        object.__setattr__(completion_options, "signal", signal)

    response = await complete_summarization(
        model,
        context,
        completion_options,
        stream_fn=stream_fn,
        retry=retry,
        callbacks=callbacks,
    )

    if response.stop_reason == "error":
        raise RuntimeError(
            f"Summarization failed: {response.error_message or 'Unknown error'}"
        )

    text_content = content_text(response.content)  # type: ignore[arg-type]
    return SummaryResult(text=text_content, usage=response.usage)


# ============================================================================
# Compaction Preparation (for extensions)
# ============================================================================


class CompactionPreparation(BaseModel):
    """Pre-calculated preparation for compaction (corresponds to TS ``CompactionPreparation``)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    first_kept_entry_id: str
    """UUID of first entry to keep."""
    messages_to_summarize: list[AgentMessage]
    """Messages that will be summarized and discarded."""
    turn_prefix_messages: list[AgentMessage]
    """Messages that will be turned into turn prefix summary (if splitting)."""
    is_split_turn: bool
    """Whether this is a split turn (cut point in middle of turn)."""
    tokens_before: int
    previous_summary: str | None = None
    """Summary from previous compaction, for iterative update."""
    file_ops: FileOperations = FileOperations()
    """File operations extracted from messagesToSummarize."""
    settings: CompactionSettings = DEFAULT_COMPACTION_SETTINGS
    """Compaction settings from settings.jsonl."""


def prepare_compaction(
    path_entries: list[SessionEntry],
    settings: CompactionSettings,
) -> CompactionPreparation | None:
    """Prepare compaction data (corresponds to TS ``prepareCompaction``)."""
    if path_entries and isinstance(path_entries[-1], CompactionEntry):
        return None

    # Find previous compaction index
    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if isinstance(path_entries[i], CompactionEntry):
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    boundary_start = 0
    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        if isinstance(prev_compaction, CompactionEntry):
            previous_summary = prev_compaction.summary
            first_kept_entry_idx = -1
            for idx, entry in enumerate(path_entries):
                if entry.id == prev_compaction.first_kept_entry_id:
                    first_kept_entry_idx = idx
                    break
            boundary_start = (
                first_kept_entry_idx
                if first_kept_entry_idx >= 0
                else prev_compaction_index + 1
            )
    boundary_end = len(path_entries)

    # Estimate tokens before
    session_ctx = _build_session_context(path_entries)  # type: ignore[arg-type]
    tokens_before = estimate_context_tokens(session_ctx.messages).tokens

    # Find cut point
    cut_point = find_cut_point(
        path_entries, boundary_start, boundary_end, settings.keep_recent_tokens
    )

    # Get UUID of first kept entry
    first_kept_entry = path_entries[cut_point.first_kept_entry_index]
    if not first_kept_entry.id:
        return None  # Session needs migration
    first_kept_entry_id = first_kept_entry.id

    history_end = (
        cut_point.turn_start_index
        if cut_point.is_split_turn
        else cut_point.first_kept_entry_index
    )

    # Messages to summarize (will be discarded after summary)
    messages_to_summarize: list[AgentMessage] = []
    for i in range(boundary_start, history_end):
        msg = _get_message_from_entry_for_compaction(path_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    # Messages for turn prefix summary (if splitting a turn)
    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(path_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    if not messages_to_summarize and not turn_prefix_messages:
        return None

    # Extract file operations from messages and previous compaction
    file_ops = _extract_file_operations(
        messages_to_summarize, path_entries, prev_compaction_index
    )

    # Also extract file ops from turn prefix if splitting
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    return CompactionPreparation(
        first_kept_entry_id=first_kept_entry_id,
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )


# ============================================================================
# Main compaction function
# ============================================================================

TURN_PREFIX_SUMMARIZATION_PROMPT = """This is the PREFIX of a turn that was too large to keep. The SUFFIX (recent work) is retained.

Summarize the prefix to provide context for the retained suffix:

## Original Request
[What did the user ask for in this turn?]

## Early Progress
- [Key decisions and work done in the prefix]

## Context for Suffix
- [Information needed to understand the retained recent work]

Be concise. Focus on what's needed to understand the kept suffix."""


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    headers: dict[str, str] | None = None,
    thinking_level: ThinkingLevel | None = None,
    stream_fn: StreamFn | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> tuple[str, Usage]:
    """Generate a summary for a turn prefix (when splitting a turn) (corresponds to TS ``generateTurnPrefixSummary``)."""
    model_max_tokens = getattr(model, "max_tokens", 0) or 0
    max_tokens = min(
        int(0.5 * reserve_tokens),  # Smaller budget for turn prefix
        model_max_tokens if model_max_tokens > 0 else 2147483647,
    )
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"

    summarization_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": int(time.time() * 1000),
        }
    ]

    context = Context(
        system_prompt=SUMMARIZATION_SYSTEM_PROMPT,
        messages=cast("list[Any]", summarization_messages),
    )
    if (
        thinking_level
        and thinking_level != "off"
        and getattr(model, "reasoning", False)
    ):
        context.thinking_level = thinking_level

    completion_options = _create_summarization_options(
        model,
        max_tokens,
        headers=headers,
        thinking_level=thinking_level,
    )

    response = await complete_summarization(
        model,
        context,
        completion_options,
        stream_fn=stream_fn,
        retry=retry,
        callbacks=callbacks,
    )

    if response.stop_reason == "error":
        raise RuntimeError(
            f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}"
        )

    return content_text(response.content), response.usage or Usage()  # type: ignore[arg-type]


async def compact(
    preparation: CompactionPreparation,
    model: Model,
    *,
    api_key: str | None = None,
    headers: dict[str, str] | None = None,
    custom_instructions: str | None = None,
    signal: object = None,
    thinking_level: ThinkingLevel | None = None,
    stream_fn: StreamFn | None = None,
    env: dict[str, str] | None = None,
    retry: RetryPolicy | None = None,
    callbacks: RetryCallbacks | None = None,
) -> CompactionResult:
    """Generate summaries for compaction using prepared data (corresponds to TS ``compact``).

    Returns CompactionResult - SessionManager adds uuid/parentUuid when saving.
    """
    settings = preparation.settings

    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None

        if preparation.messages_to_summarize:
            history_result = await generate_summary_with_usage(
                preparation.messages_to_summarize,
                model,
                settings.reserve_tokens,
                api_key=api_key,
                headers=headers,
                signal=signal,
                custom_instructions=custom_instructions,
                previous_summary=preparation.previous_summary,
                thinking_level=thinking_level,
                stream_fn=stream_fn,
                env=env,
                retry=retry,
                callbacks=callbacks,
            )
            history_text = history_result.text
            history_usage = history_result.usage

        turn_prefix_result = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            model,
            settings.reserve_tokens,
            headers=headers,
            thinking_level=thinking_level,
            stream_fn=stream_fn,
            retry=retry,
            callbacks=callbacks,
        )
        turn_prefix_text, turn_prefix_usage = turn_prefix_result

        summary = f"{history_text}\n\n---\n\n**Turn Context (split turn):**\n\n{turn_prefix_text}"
        summary_usage = (
            _combine_usage(history_usage, turn_prefix_usage)
            if history_usage
            else turn_prefix_usage
        )
    else:
        result = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            model,
            settings.reserve_tokens,
            api_key=api_key,
            headers=headers,
            signal=signal,
            custom_instructions=custom_instructions,
            previous_summary=preparation.previous_summary,
            thinking_level=thinking_level,
            stream_fn=stream_fn,
            env=env,
            retry=retry,
            callbacks=callbacks,
        )
        summary = result.text
        summary_usage = result.usage or Usage()

    # Compute file lists and append to summary
    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    if not preparation.first_kept_entry_id:
        raise RuntimeError("First kept entry has no UUID - session may need migration")

    return CompactionResult(
        summary=summary,
        first_kept_entry_id=preparation.first_kept_entry_id,
        tokens_before=preparation.tokens_before,
        usage=summary_usage,
        details=CompactionDetails(read_files=read_files, modified_files=modified_files),
    )


__all__ = [
    "DEFAULT_COMPACTION_SETTINGS",
    "CompactionDetails",
    "CompactionPreparation",
    "CompactionResult",
    "CompactionSettings",
    "ContextUsageEstimate",
    "CutPointResult",
    "calculate_context_tokens",
    "compact",
    "complete_summarization",
    "estimate_context_tokens",
    "estimate_tokens",
    "find_cut_point",
    "find_turn_start_index",
    "generate_summary",
    "generate_summary_with_usage",
    "get_last_assistant_usage",
    "prepare_compaction",
    "should_compact",
]
