"""快速确定性哈希（对应 ``utils/hash.ts``）。"""

from __future__ import annotations


def short_hash(text: str) -> str:
    """快速确定性哈希以缩短长字符串（对应 TS ``shortHash``）。

    使用双 32 位哈希（类似 TS 的 ``Math.imul`` 实现）。
    """
    h1 = 0xDEADBEEF
    h2 = 0x41C6CE57

    for ch in text:
        code = ord(ch)
        h1 = _imul(h1 ^ code, 2654435761)
        h2 = _imul(h2 ^ code, 1597334677)

    h1 = _imul(h1 ^ (h1 >> 16), 2246822507) ^ _imul(h2 ^ (h2 >> 13), 3266489909)
    h2 = _imul(h2 ^ (h2 >> 16), 2246822507) ^ _imul(h1 ^ (h1 >> 13), 3266489909)

    # Python 的 int 是无符号的，模拟 JS >>> 0
    return f"{(h2 & 0xFFFFFFFF):x}{(h1 & 0xFFFFFFFF):x}"


def _imul(a: int, b: int) -> int:
    """模拟 JS 的 ``Math.imul``（32 位整数乘法）。"""
    a = a & 0xFFFFFFFF
    b = b & 0xFFFFFFFF
    return (a * b) & 0xFFFFFFFF