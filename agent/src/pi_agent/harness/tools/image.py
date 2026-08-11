"""图片 MIME 类型检测与 Base64 编码（对应 ``harness/tools/image.ts``）。

支持 JPEG / PNG / GIF / WebP / BMP 格式检测，以及自定义 Base64 编码。
"""

from __future__ import annotations

_PNG_SIGNATURE = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A])


def detect_supported_image_mime_type(buffer: bytes) -> str | None:
    """检测图片 MIME 类型（对应 TS ``detectSupportedImageMimeType``）。

    支持 JPEG / PNG / GIF / WebP / BMP。返回 ``None`` 表示非图片或
    不被支持的格式。
    """
    if _starts_with(buffer, bytes([0xFF, 0xD8, 0xFF])):
        return None if buffer[3] == 0xF7 else "image/jpeg"
    if _starts_with(buffer, _PNG_SIGNATURE):
        return "image/png" if _is_png(buffer) and not _is_animated_png(buffer) else None
    if _starts_with_ascii(buffer, 0, "GIF"):
        return "image/gif"
    if _starts_with_ascii(buffer, 0, "RIFF") and _starts_with_ascii(buffer, 8, "WEBP"):
        return "image/webp"
    if _starts_with_ascii(buffer, 0, "BM") and _is_bmp(buffer):
        return "image/bmp"
    return None


def encode_base64(data: bytes) -> str:
    """自定义 Base64 编码（对应 TS ``encodeBase64``）。"""
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    output: list[str] = []
    for i in range(0, len(data), 3):
        first = data[i]
        second = data[i + 1] if i + 1 < len(data) else None
        third = data[i + 2] if i + 2 < len(data) else None
        output.append(alphabet[first >> 2])
        output.append(alphabet[((first & 0x03) << 4) | ((second or 0) >> 4)])
        output.append("=" if second is None else alphabet[((second & 0x0F) << 2) | ((third or 0) >> 6)])
        output.append("=" if third is None else alphabet[third & 0x3F])
    return "".join(output)


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _is_png(buffer: bytes) -> bool:
    return (
        len(buffer) >= 16
        and _read_uint32_be(buffer, len(_PNG_SIGNATURE)) == 13
        and _starts_with_ascii(buffer, 12, "IHDR")
    )


def _is_animated_png(buffer: bytes) -> bool:
    offset = len(_PNG_SIGNATURE)
    while offset + 8 <= len(buffer):
        chunk_length = _read_uint32_be(buffer, offset)
        chunk_type_offset = offset + 4
        if _starts_with_ascii(buffer, chunk_type_offset, "acTL"):
            return True
        if _starts_with_ascii(buffer, chunk_type_offset, "IDAT"):
            return False
        next_offset = offset + 8 + chunk_length + 4
        if next_offset <= offset or next_offset > len(buffer):
            return False
        offset = next_offset
    return False


def _is_bmp(buffer: bytes) -> bool:
    if len(buffer) < 26:
        return False
    declared_file_size = _read_uint32_le(buffer, 2)
    pixel_data_offset = _read_uint32_le(buffer, 10)
    dib_header_size = _read_uint32_le(buffer, 14)
    if declared_file_size != 0 and declared_file_size < 26:
        return False
    if pixel_data_offset < 14 + dib_header_size:
        return False
    if declared_file_size != 0 and pixel_data_offset >= declared_file_size:
        return False

    if dib_header_size == 12:
        color_planes = _read_uint16_le(buffer, 22)
        bits_per_pixel = _read_uint16_le(buffer, 24)
    elif 40 <= dib_header_size <= 124:
        if len(buffer) < 30:
            return False
        color_planes = _read_uint16_le(buffer, 26)
        bits_per_pixel = _read_uint16_le(buffer, 28)
    else:
        return False
    return color_planes == 1 and bits_per_pixel in (1, 4, 8, 16, 24, 32)


def _read_uint16_le(buffer: bytes, offset: int) -> int:
    return buffer[offset] + (buffer[offset + 1] << 8)


def _read_uint32_be(buffer: bytes, offset: int) -> int:
    return (
        buffer[offset] * 0x1000000
        + (buffer[offset + 1] << 16)
        + (buffer[offset + 2] << 8)
        + buffer[offset + 3]
    )


def _read_uint32_le(buffer: bytes, offset: int) -> int:
    return (
        buffer[offset]
        + (buffer[offset + 1] << 8)
        + (buffer[offset + 2] << 16)
        + buffer[offset + 3] * 0x1000000
    )


def _starts_with(buffer: bytes, prefix: bytes) -> bool:
    if len(buffer) < len(prefix):
        return False
    return all(buffer[i] == prefix[i] for i in range(len(prefix)))


def _starts_with_ascii(buffer: bytes, offset: int, text: str) -> bool:
    if len(buffer) < offset + len(text):
        return False
    for i, ch in enumerate(text):
        if buffer[offset + i] != ord(ch):
            return False
    return True