"""诊断信息工具。

提供 ``format_thrown_value``、``extract_diagnostic_error``、
``create_assistant_message_diagnostic``、``append_assistant_message_diagnostic``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DiagnosticErrorInfo:
    """诊断错误信息。"""

    name: str | None = None
    message: str = ""
    stack: str | None = None
    code: str | int | None = None


@dataclass
class AssistantMessageDiagnostic:
    """assistant 消息诊断。"""

    type: str
    timestamp: float
    error: DiagnosticErrorInfo | None = None
    details: dict[str, object] | None = None


def format_thrown_value(value: object) -> str:
    """格式化抛出值。"""
    if isinstance(value, BaseException):
        return str(value) or type(value).__name__
    if isinstance(value, str):
        return value
    return str(value)


def extract_diagnostic_error(error: object) -> DiagnosticErrorInfo:
    """提取诊断错误信息。"""
    if not isinstance(error, BaseException):
        return DiagnosticErrorInfo(
            name="ThrownValue", message=format_thrown_value(error)
        )
    code = getattr(error, "code", None)
    return DiagnosticErrorInfo(
        name=type(error).__name__ if type(error).__name__ != "Exception" else None,
        message=str(error) or type(error).__name__,
        stack=getattr(error, "stack", None),
        code=code if isinstance(code, (str, int)) else None,
    )


def create_assistant_message_diagnostic(
    type_name: str,
    error: object,
    details: dict[str, object] | None = None,
) -> AssistantMessageDiagnostic:
    """创建 assistant 消息诊断。"""
    return AssistantMessageDiagnostic(
        type=type_name,
        timestamp=time.time(),
        error=extract_diagnostic_error(error),
        details=details,
    )


def append_assistant_message_diagnostic(
    message: dict[str, object],
    diagnostic: AssistantMessageDiagnostic,
) -> None:
    """追加诊断到消息。"""
    existing = message.get("diagnostics")
    if not isinstance(existing, list):
        existing = []
    existing.append(diagnostic)
    message["diagnostics"] = existing