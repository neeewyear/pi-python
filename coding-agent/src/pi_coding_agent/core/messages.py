"""自定义消息类型与 LLM 转换（对应 TS ``core/messages.ts``）。

从 ``pi_agent.types`` 和 ``pi_agent.harness.messages`` 再导出共享类型与函数。
"""

from __future__ import annotations

from pi_agent.harness.messages import (
    COMPACTION_SUMMARY_PREFIX,
    COMPACTION_SUMMARY_SUFFIX,
    BRANCH_SUMMARY_PREFIX,
    BRANCH_SUMMARY_SUFFIX,
    bash_execution_to_text,
    create_branch_summary_message,
    create_compaction_summary_message,
    create_custom_message,
    convert_to_llm,
)
from pi_agent.types import (
    BashExecutionMessage,
    BranchSummaryMessage,
    CompactionSummaryMessage,
    CustomMessage,
)

__all__ = [
    "BashExecutionMessage",
    "BranchSummaryMessage",
    "COMPACTION_SUMMARY_PREFIX",
    "COMPACTION_SUMMARY_SUFFIX",
    "BRANCH_SUMMARY_PREFIX",
    "BRANCH_SUMMARY_SUFFIX",
    "CompactionSummaryMessage",
    "CustomMessage",
    "bash_execution_to_text",
    "create_branch_summary_message",
    "create_compaction_summary_message",
    "create_custom_message",
    "convert_to_llm",
]