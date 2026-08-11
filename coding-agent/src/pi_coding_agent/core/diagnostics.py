"""资源冲突与诊断类型定义（对应 TS ``core/diagnostics.ts``）。

提供 ``ResourceCollision`` 和 ``ResourceDiagnostic`` 两个 Pydantic 模型，
用于描述扩展/技能/提示词/主题等资源的注册冲突和诊断信息。
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ResourceCollision(BaseModel):
    """资源冲突描述。

    当多个来源注册了同名资源时，记录胜出者与落败者的路径和来源信息。
    """

    resource_type: Literal["extension", "skill", "prompt", "theme"]
    """资源类型。"""
    name: str
    """资源名称（技能名、命令名、提示词名、主题名等）。"""
    winner_path: str
    """胜出资源路径。"""
    loser_path: str
    """落败资源路径。"""
    winner_source: str | None = None
    """胜出资源来源（如 ``"npm:foo"``、``"git:..."``、``"local"``）。"""
    loser_source: str | None = None
    """落败资源来源。"""


class ResourceDiagnostic(BaseModel):
    """资源诊断信息。

    描述资源加载过程中产生的警告、错误或冲突。
    """

    type: Literal["warning", "error", "collision"]
    """诊断类型。"""
    message: str
    """诊断消息。"""
    path: str | None = None
    """关联文件路径。"""
    collision: ResourceCollision | None = None
    """冲突详情（仅 ``type == "collision"`` 时存在）。"""