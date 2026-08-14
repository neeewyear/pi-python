"""工具执行上下文。

为内置执行工具提供文件系统和 shell 能力。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ..types import ExecutionEnv


class ExecutionToolContext(BaseModel):
    """文件系统和 shell 上下文，供内置执行工具使用。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    env: Any  # ExecutionEnv — Protocol 不可直接用于 Pydantic 字段类型