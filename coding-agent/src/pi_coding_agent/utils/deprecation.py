"""弃用警告工具。

提供 ``deprecation_warning`` 函数，用于发出弃用警告。
"""

from __future__ import annotations

import warnings

_emitted_deprecation_warnings: set[str] = set()


def deprecation_warning(message: str) -> None:
    """发出弃用警告。

    每条消息只会在首次调用时发出警告，后续重复调用
    相同的消息将被忽略。

    Args:
        message: 弃用警告消息。
    """
    if message in _emitted_deprecation_warnings:
        return
    _emitted_deprecation_warnings.add(message)
    warnings.warn(message, DeprecationWarning, stacklevel=2)