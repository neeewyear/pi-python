"""终端颜色查询解析（对标 origin_pi ``terminal-colors.ts``）。

提供 OSC 11（背景色查询）与 DSR 996（配色方案查询）响应的解析函数，
供 ``TUI.query_terminal_background_color`` / ``query_terminal_color_scheme`` 使用。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

# 配色方案类型：终端报告深色或浅色
TerminalColorScheme = Literal["dark", "light"]


@dataclass(frozen=True)
class RgbColor:
    """RGB 颜色（各通道 0-255）。"""

    r: int
    g: int
    b: int


_OSC11_BG_RESPONSE_PATTERN = re.compile(
    r"^\x1b\]11;([^\x07\x1b]*)(?:\x07|\x1b\\)$", re.IGNORECASE
)
_COLOR_SCHEME_REPORT_PATTERN = re.compile(r"^(?:\x1b\[\?997;(1|2)n)+$")


def hex_to_rgb(hex_color: str) -> RgbColor:
    """将 6 位十六进制颜色（可带 # 前缀）解析为 RGB。"""
    normalized = hex_color.removeprefix("#")
    return RgbColor(
        r=int(normalized[0:2], 16),
        g=int(normalized[2:4], 16),
        b=int(normalized[4:6], 16),
    )


def parse_osc_hex_channel(channel: str) -> int | None:
    """解析 OSC 颜色通道（可为 2/4/8 位十六进制，归一化到 0-255）。"""
    if not re.fullmatch(r"[0-9a-f]+", channel, re.IGNORECASE):
        return None
    max_val = 16 ** len(channel) - 1
    if max_val <= 0:
        return None
    return int(round((int(channel, 16) / max_val) * 255))


def is_osc11_background_color_response(data: str) -> bool:
    """判断输入是否为 OSC 11 背景色响应。"""
    return _OSC11_BG_RESPONSE_PATTERN.match(data) is not None


def parse_osc11_background_color(data: str) -> RgbColor | None:
    """从 OSC 11 响应解析背景色。

    支持 ``#RRGGBB`` / ``#RRRRGGGGBBBB`` 十六进制与 ``rgb:r/g/b`` 两种格式。
    """
    match = _OSC11_BG_RESPONSE_PATTERN.match(data)
    if match is None:
        return None

    value = match.group(1).strip()
    if value.startswith("#"):
        hex_str = value[1:]
        if re.fullmatch(r"[0-9a-f]{6}", hex_str, re.IGNORECASE):
            return hex_to_rgb(value)
        if re.fullmatch(r"[0-9a-f]{12}", hex_str, re.IGNORECASE):
            r = parse_osc_hex_channel(hex_str[0:4])
            g = parse_osc_hex_channel(hex_str[4:8])
            b = parse_osc_hex_channel(hex_str[8:12])
            if r is not None and g is not None and b is not None:
                return RgbColor(r=r, g=g, b=b)
        return None

    rgb_value = re.sub(r"^rgba?:", "", value, flags=re.IGNORECASE)
    parts = rgb_value.split("/")
    if len(parts) != 3:
        return None
    r = parse_osc_hex_channel(parts[0])
    g = parse_osc_hex_channel(parts[1])
    b = parse_osc_hex_channel(parts[2])
    if r is not None and g is not None and b is not None:
        return RgbColor(r=r, g=g, b=b)
    return None


def parse_terminal_color_scheme_report(data: str) -> TerminalColorScheme | None:
    """解析 DSR 996 配色方案报告（``CSI ? 997 ; 1 n``=dark，``2 n``=light）。"""
    match = _COLOR_SCHEME_REPORT_PATTERN.match(data)
    if match is None:
        return None
    return "light" if match.group(1) == "2" else "dark"
