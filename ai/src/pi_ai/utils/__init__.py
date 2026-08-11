"""pi-ai utility modules (对应 ``pi/packages/ai/src/utils/``)。

包含 18 个工具模块：
- ``uuid``: UUIDv7 生成器
- ``event_stream``: 事件流（``EventStream`` / ``AssistantMessageEventStream``）
- ``retry``: 重试策略与退避（``retry_assistant_call`` / ``is_retryable_assistant_error``）
- ``overflow``: 上下文溢出检测（``is_context_overflow``）
- ``json_parse``: 流式 JSON 解析与修复
- ``validation``: 工具调用参数验证与类型转换
- ``abort``: 取消信号（``CancellationToken`` / ``combine_abort_signals``）
- ``diagnostics``: 诊断信息工具
- ``error_body``: Provider HTTP 错误标准化
- ``estimate``: Token 估算
- ``hash``: 快速确定性哈希
- ``headers``: HTTP 头处理
- ``text``: 文本提取
- ``typebox_helpers``: Pydantic 兼容的 schema 辅助
- ``sanitize_unicode``: Unicode 代理字符清理
- ``provider_env``: Provider 环境变量解析
- ``provider_retry``: Provider 级别重试
- ``node_http_proxy``: HTTP 代理解析
"""

from __future__ import annotations

from .abort import (
    CancellationToken,
    CombinedAbortSignal,
    combine_abort_signals,
    operation_signal,
    race_with_abort_signal,
)
from .diagnostics import (
    AssistantMessageDiagnostic,
    DiagnosticErrorInfo,
    append_assistant_message_diagnostic,
    create_assistant_message_diagnostic,
    extract_diagnostic_error,
    format_thrown_value,
)
from .error_body import (
    MAX_PROVIDER_ERROR_BODY_CHARS,
    NormalizedProviderError,
    format_provider_error,
    normalize_provider_error,
    safe_json_stringify,
    truncate_error_text,
)
from .estimate import (
    ContextUsageEstimate,
    calculate_context_tokens,
    estimate_context_tokens,
    estimate_message_tokens,
    estimate_text_and_image_content_tokens,
    estimate_text_tokens,
)
from .event_stream import (
    AssistantMessageEventStream,
    EventStream,
    create_assistant_message_event_stream,
)
from .hash import short_hash
from .headers import headers_to_record, provider_headers_to_record
from .json_parse import parse_json_with_repair, parse_streaming_json, repair_json
from .node_http_proxy import resolve_http_proxy_url_for_target
from .overflow import get_overflow_patterns, is_context_overflow, is_recoverable_length
from .provider_env import get_provider_env_value
from .provider_retry import (
    ProviderError,
    ProviderRetryOptions,
    retry_provider_request,
)
from .retry import (
    RetryCallbacks,
    RetryPolicy,
    RetrySleepAbortError,
    is_retryable_assistant_error,
    retry_assistant_call,
)
from .sanitize_unicode import sanitize_surrogates
from .text import content_text
from .typebox_helpers import string_enum
from .uuid import uuidv7
from .validation import validate_tool_arguments, validate_tool_call

__all__ = [
    "MAX_PROVIDER_ERROR_BODY_CHARS",
    "AssistantMessageDiagnostic",
    "AssistantMessageEventStream",
    "CancellationToken",
    "CombinedAbortSignal",
    "ContextUsageEstimate",
    "DiagnosticErrorInfo",
    "EventStream",
    "NormalizedProviderError",
    "ProviderError",
    "ProviderRetryOptions",
    "RetryCallbacks",
    "RetryPolicy",
    "RetrySleepAbortError",
    "append_assistant_message_diagnostic",
    "calculate_context_tokens",
    "combine_abort_signals",
    "content_text",
    "create_assistant_message_diagnostic",
    "create_assistant_message_event_stream",
    "estimate_context_tokens",
    "estimate_message_tokens",
    "estimate_text_and_image_content_tokens",
    "estimate_text_tokens",
    "extract_diagnostic_error",
    "format_provider_error",
    "format_thrown_value",
    "get_overflow_patterns",
    "get_provider_env_value",
    "headers_to_record",
    "is_context_overflow",
    "is_recoverable_length",
    "is_retryable_assistant_error",
    "normalize_provider_error",
    "operation_signal",
    "parse_json_with_repair",
    "parse_streaming_json",
    "provider_headers_to_record",
    "race_with_abort_signal",
    "repair_json",
    "resolve_http_proxy_url_for_target",
    "retry_assistant_call",
    "retry_provider_request",
    "safe_json_stringify",
    "sanitize_surrogates",
    "short_hash",
    "string_enum",
    "truncate_error_text",
    "uuidv7",
    "validate_tool_arguments",
    "validate_tool_call",
]
