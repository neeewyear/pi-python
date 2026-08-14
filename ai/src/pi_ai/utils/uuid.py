"""UUIDv7 生成器。"""

from __future__ import annotations

import os
import time

_last_timestamp = -1
_sequence = 0


def _fill_random_bytes(size: int) -> bytearray:
    """填充随机字节。"""
    return bytearray(os.urandom(size))


def uuidv7() -> str:
    """生成时间有序的 UUIDv7。

    返回格式：``xxxxxxxx-xxxx-7xxx-8xxx-xxxxxxxxxxxx``。
    """
    global _last_timestamp, _sequence

    random = _fill_random_bytes(16)
    timestamp = int(time.time() * 1000)

    if timestamp > _last_timestamp:
        _sequence = (
            random[6] * 0x1000000 + random[7] * 0x10000 + random[8] * 0x100 + random[9]
        )
        _last_timestamp = timestamp
    else:
        _sequence = (_sequence + 1) & 0xFFFFFFFF
        if _sequence == 0:
            _last_timestamp += 1

    bytes_arr = bytearray(16)
    bytes_arr[0] = (_last_timestamp >> 40) & 0xFF
    bytes_arr[1] = (_last_timestamp >> 32) & 0xFF
    bytes_arr[2] = (_last_timestamp >> 24) & 0xFF
    bytes_arr[3] = (_last_timestamp >> 16) & 0xFF
    bytes_arr[4] = (_last_timestamp >> 8) & 0xFF
    bytes_arr[5] = _last_timestamp & 0xFF
    bytes_arr[6] = 0x70 | ((_sequence >> 28) & 0x0F)
    bytes_arr[7] = (_sequence >> 20) & 0xFF
    bytes_arr[8] = 0x80 | ((_sequence >> 14) & 0x3F)
    bytes_arr[9] = (_sequence >> 6) & 0xFF
    bytes_arr[10] = ((_sequence & 0x3F) << 2) | (random[10] & 0x03)
    bytes_arr[11] = random[11]
    bytes_arr[12] = random[12]
    bytes_arr[13] = random[13]
    bytes_arr[14] = random[14]
    bytes_arr[15] = random[15]

    hex_str = "".join(f"{b:02x}" for b in bytes_arr)
    return f"{hex_str[0:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:32]}"
