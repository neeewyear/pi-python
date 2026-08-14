"""启动性能计时工具。

通过 ``PI_TIMING=1`` 环境变量启用。使用 ``time.time()`` 而非 ``Date.now()``，
所有输出走 ``print()`` 而非 ``console.error()``。
"""

from __future__ import annotations

import os
import time as _time

ENABLED = os.environ.get("PI_TIMING") == "1"


class _TimingNamespace:
    """单个命名空间的计时数据。"""

    timings: list[tuple[str, float]]
    last_time: float

    def __init__(self) -> None:
        self.timings = []
        self.last_time = _time.time()


_timing_namespaces: dict[str, _TimingNamespace] = {}


def reset_timings(namespace: str = "main") -> None:
    """重置指定命名空间的计时器。

    Args:
        namespace: 命名空间标签（默认 ``"main"``）。
    """
    if not ENABLED:
        return
    _timing_namespaces[namespace] = _TimingNamespace()


def time(label: str, namespace: str = "main") -> None:
    """记录一条计时数据。

    计算从上次记录（或 reset）到现在的毫秒数。

    Args:
        label: 计时点标签。
        namespace: 命名空间标签（默认 ``"main"``）。
    """
    if not ENABLED:
        return
    now = _time.time()

    if namespace not in _timing_namespaces:
        reset_timings(namespace)

    ns = _timing_namespaces[namespace]
    ns.timings.append((label, (now - ns.last_time) * 1000))
    ns.last_time = now


def _print_timing_group(title: str, timings: list[tuple[str, float]]) -> None:
    """打印单个命名空间的计时分组。"""
    printable = [(label, ms) for (label, ms) in timings if ms >= 0]
    if not printable:
        return
    print(f"\n--- {title} ---")
    for label, ms in printable:
        print(f"  {label}: {ms:.0f}ms")
    total = sum(ms for _, ms in printable)
    print(f"  TOTAL: {total:.0f}ms")
    print(f"{'-' * (len(title) + 8)}\n")


def print_timings() -> None:
    """打印所有命名空间的计时汇总。"""
    if not ENABLED:
        return
    for namespace, ns in _timing_namespaces.items():
        _print_timing_group(f"Startup Timings: {namespace}", ns.timings)
