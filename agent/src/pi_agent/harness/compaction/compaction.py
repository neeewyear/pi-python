"""压缩主流程（对应 ``compaction.ts`` 的 compact / shouldCompact / prepareCompaction 等）。

包含：
- ``CompactionSettings`` / ``DEFAULT_COMPACTION_SETTINGS``
- ``should_compact`` / ``prepare_compaction`` / ``compact``
- ``CompactResult`` / ``CompactionDetails``
"""

from __future__ import annotations

from typing import Any

from pi_session.types import Entry, MessageEntry
from pydantic import BaseModel, ConfigDict

from ...result import Result, err, ok
from ...types import AgentMessage, Model, ThinkingLevel, Usage
from ..messages import create_compaction_summary_message
from ..types import CompactionError
from .compaction_summary import (
    TURN_PREFIX_SUMMARIZATION_PROMPT,
    generate_summary_with_usage,
)
from .compaction_token import (
    CompactionPreparation,
    estimate_context_tokens,
    find_cut_point,
)
from .utils import (
    FileOperations,
    compute_file_lists,
    create_file_ops,
    extract_file_ops_from_message,
    format_file_operations,
    serialize_conversation,
)

# ---------------------------------------------------------------------------
# 设置
# ---------------------------------------------------------------------------


class CompactionSettings(BaseModel):
    """压缩阈值设置（对应 TS ``CompactionSettings``）。"""

    enabled: bool = True
    reserve_tokens: int = 16384
    keep_recent_tokens: int = 20000


DEFAULT_COMPACTION_SETTINGS = CompactionSettings()


# ---------------------------------------------------------------------------
# CompactResult / CompactionDetails
# ---------------------------------------------------------------------------


class CompactionDetails(BaseModel):
    """压缩详情（对应 TS ``CompactionDetails``）。"""

    read_files: list[str] = []
    modified_files: list[str] = []


class CompactResult(BaseModel):
    """压缩结果（对应 TS ``CompactResult``）。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    summary: str
    tokens_before: int
    usage: Usage | None = None
    retained_tail: list[AgentMessage] = []
    details: CompactionDetails | None = None


# ---------------------------------------------------------------------------
# shouldCompact
# ---------------------------------------------------------------------------


def should_compact(
    context_tokens: int,
    context_window: int,
    settings: CompactionSettings,
) -> bool:
    """判断是否需要压缩（对应 TS ``shouldCompact``）。"""
    if not settings.enabled:
        return False
    return context_tokens > context_window - settings.reserve_tokens


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _extract_file_operations(
    messages: list[AgentMessage],
    entries: list[Entry],
    prev_compaction_index: int,
) -> FileOperations:
    """提取文件操作（对应 TS ``extractFileOperations``）。"""
    file_ops = create_file_ops()
    if prev_compaction_index >= 0:
        prev_compaction = entries[prev_compaction_index]
        if prev_compaction.type == "compaction" and prev_compaction.details is not None:
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
    for msg in messages:
        extract_file_ops_from_message(msg, file_ops)
    return file_ops


def _get_message_from_entry(entry: Entry) -> AgentMessage | None:
    """从条目获取消息（对应 TS ``getMessageFromEntry``）。"""
    if entry.type == "message":
        return entry.message
    if entry.type == "branch_summary":
        from ..messages import create_branch_summary_message

        return create_branch_summary_message(
            entry.summary, entry.from_id, entry.timestamp
        )
    if entry.type == "compaction":
        return create_compaction_summary_message(
            entry.summary, entry.tokens_before, entry.timestamp
        )
    return None


def _get_message_from_entry_for_compaction(entry: Entry) -> AgentMessage | None:
    """获取压缩用消息（跳过 compaction 条目，对应 TS ``getMessageFromEntryForCompaction``）。"""
    if entry.type == "compaction":
        return None
    return _get_message_from_entry(entry)


# ---------------------------------------------------------------------------
# prepareCompaction
# ---------------------------------------------------------------------------


def prepare_compaction(
    path_entries: list[Entry],
    settings: CompactionSettings,
) -> Result[CompactionPreparation | None, CompactionError]:
    """准备压缩数据（对应 TS ``prepareCompaction``）。"""
    if not path_entries or path_entries[-1].type == "compaction":
        return ok(None)

    # 查找上一个 compaction 条目
    prev_compaction_index = -1
    for i in range(len(path_entries) - 1, -1, -1):
        if path_entries[i].type == "compaction":
            prev_compaction_index = i
            break

    previous_summary: str | None = None
    compactable_entries = list(path_entries)

    if prev_compaction_index >= 0:
        prev_compaction = path_entries[prev_compaction_index]
        previous_summary = prev_compaction.summary  # type: ignore[union-attr]

        # 构造虚拟 retained 条目
        virtual_retained: list[Entry] = []
        for index, message in enumerate(prev_compaction.retained_tail):  # type: ignore[union-attr]
            parent_id = (
                prev_compaction.id
                if index == 0
                else f"{prev_compaction.id}:retained:{index - 1}"
            )
            virtual_retained.append(
                MessageEntry(
                    id=f"{prev_compaction.id}:retained:{index}",
                    parent_id=parent_id,
                    seq=prev_compaction.seq,
                    timestamp=message.timestamp,
                    message=message,
                )
            )
        compactable_entries = [
            *virtual_retained,
            *path_entries[prev_compaction_index + 1 :],
        ]

    boundary_end = len(compactable_entries)

    # 估算 token（延迟导入避免与 pi_session 初始化成环）
    from pi_session.context import build_session_context

    session_ctx = build_session_context(path_entries)
    tokens_before = estimate_context_tokens(session_ctx.messages).tokens

    # 查找切点
    cut_point = find_cut_point(
        compactable_entries, 0, boundary_end, settings.keep_recent_tokens
    )
    history_end = (
        cut_point.turn_start_index
        if cut_point.is_split_turn
        else cut_point.first_kept_entry_index
    )

    # 收集待摘要消息
    messages_to_summarize: list[AgentMessage] = []
    for i in range(history_end):
        msg = _get_message_from_entry_for_compaction(compactable_entries[i])
        if msg is not None:
            messages_to_summarize.append(msg)

    turn_prefix_messages: list[AgentMessage] = []
    if cut_point.is_split_turn:
        for i in range(cut_point.turn_start_index, cut_point.first_kept_entry_index):
            msg = _get_message_from_entry_for_compaction(compactable_entries[i])
            if msg is not None:
                turn_prefix_messages.append(msg)

    retained_tail: list[AgentMessage] = []
    for i in range(cut_point.first_kept_entry_index, boundary_end):
        msg = _get_message_from_entry_for_compaction(compactable_entries[i])
        if msg is not None:
            retained_tail.append(msg)

    file_ops = _extract_file_operations(
        messages_to_summarize, path_entries, prev_compaction_index
    )
    if cut_point.is_split_turn:
        for msg in turn_prefix_messages:
            extract_file_ops_from_message(msg, file_ops)

    preparation = CompactionPreparation(
        messages_to_summarize=messages_to_summarize,
        turn_prefix_messages=turn_prefix_messages,
        retained_tail=retained_tail,
        is_split_turn=cut_point.is_split_turn,
        tokens_before=tokens_before,
        previous_summary=previous_summary,
        file_ops=file_ops,
        settings=settings,
    )
    return ok(preparation)


# ---------------------------------------------------------------------------
# compact
# ---------------------------------------------------------------------------


def _combine_usage(first: Usage, second: Usage) -> Usage:
    """合并两个 Usage（对应 TS ``combineUsage``）。"""
    return Usage(
        input=first.input + second.input,
        output=first.output + second.output,
        cache_read=first.cache_read + second.cache_read,
        cache_write=first.cache_write + second.cache_write,
        total_tokens=first.total_tokens + second.total_tokens,
        cost=first.cost.model_copy(
            update={
                "input": first.cost.input + second.cost.input,
                "output": first.cost.output + second.cost.output,
                "cache_read": first.cost.cache_read + second.cost.cache_read,
                "cache_write": first.cost.cache_write + second.cost.cache_write,
                "total": first.cost.total + second.cost.total,
            }
        ),
    )


async def _generate_turn_prefix_summary(
    messages: list[AgentMessage],
    model: Model,
    reserve_tokens: int,
    *,
    thinking_level: ThinkingLevel | None = None,
) -> Result[tuple[str, Usage], CompactionError]:
    """生成回合前缀摘要（对应 TS ``generateTurnPrefixSummary``）。"""
    from ..messages import convert_to_llm

    max_tokens = min(int(0.5 * reserve_tokens), 4096)
    llm_messages = convert_to_llm(messages)
    conversation_text = serialize_conversation(llm_messages)
    prompt_text = f"<conversation>\n{conversation_text}\n</conversation>\n\n{TURN_PREFIX_SUMMARIZATION_PROMPT}"

    import time

    summarization_messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
            "timestamp": int(time.time() * 1000),
        }
    ]

    from .compaction_summary import (
        SUMMARIZATION_SYSTEM_PROMPT,
        complete_simple_with_retries,
    )

    try:
        response = await complete_simple_with_retries(
            model,
            summarization_messages,  # type: ignore[arg-type]
            SUMMARIZATION_SYSTEM_PROMPT,
            max_tokens,
        )
    except NotImplementedError:
        return err(CompactionError("pi-ai backend not available"))

    if response.stop_reason == "aborted":
        return err(
            CompactionError(
                response.error_message or "Turn prefix summarization aborted",
            )
        )
    if response.stop_reason == "error":
        return err(
            CompactionError(
                f"Turn prefix summarization failed: {response.error_message or 'Unknown error'}",
            )
        )

    def _get_text(content: list[Any]) -> str:
        parts: list[str] = []
        for block in content:
            if hasattr(block, "text"):
                parts.append(block.text)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts)

    return ok((_get_text(response.content), response.usage or Usage()))


async def compact(
    preparation: CompactionPreparation,
    model: Model,
    *,
    custom_instructions: str | None = None,
    thinking_level: ThinkingLevel | None = None,
) -> Result[CompactResult, CompactionError]:
    """执行压缩（对应 TS ``compact``）。"""
    settings: CompactionSettings = preparation.settings  # type: ignore[assignment]

    if preparation.is_split_turn and preparation.turn_prefix_messages:
        history_text = "No prior history."
        history_usage: Usage | None = None

        if preparation.messages_to_summarize:
            history_result = await generate_summary_with_usage(
                preparation.messages_to_summarize,
                model,
                settings.reserve_tokens,
                custom_instructions=custom_instructions,
                previous_summary=preparation.previous_summary,
                thinking_level=thinking_level,
            )
            if not history_result.is_ok():
                return err(history_result.error)
            history_text = history_result.value[0]
            history_usage = history_result.value[1]

        turn_prefix_result = await _generate_turn_prefix_summary(
            preparation.turn_prefix_messages,
            model,
            settings.reserve_tokens,
            thinking_level=thinking_level,
        )
        if not turn_prefix_result.is_ok():
            return err(turn_prefix_result.error)

        summary = (
            f"{history_text}\n\n---\n\n"
            f"**Turn Context (split turn):**\n\n{turn_prefix_result.value[0]}"
        )
        summary_usage = (
            _combine_usage(history_usage, turn_prefix_result.value[1])
            if history_usage
            else turn_prefix_result.value[1]
        )
    else:
        summary_result = await generate_summary_with_usage(
            preparation.messages_to_summarize,
            model,
            settings.reserve_tokens,
            custom_instructions=custom_instructions,
            previous_summary=preparation.previous_summary,
            thinking_level=thinking_level,
        )
        if not summary_result.is_ok():
            return err(summary_result.error)
        summary = summary_result.value[0]
        summary_usage = summary_result.value[1]

    read_files, modified_files = compute_file_lists(preparation.file_ops)
    summary += format_file_operations(read_files, modified_files)

    return ok(
        CompactResult(
            summary=summary,
            tokens_before=preparation.tokens_before,
            usage=summary_usage,
            retained_tail=preparation.retained_tail,
            details=CompactionDetails(
                read_files=read_files,
                modified_files=modified_files,
            ),
        )
    )
