"""Unicode 清理工具（对应 ``utils/sanitize-unicode.ts``）。"""

from __future__ import annotations

import re

# 未配对的代理字符：高代理（0xD800-0xDBFF）后无低代理（0xDC00-0xDFFF），
# 或低代理前无高代理
_UNPAIRED_SURROGATE_PATTERN = re.compile(
    r"[\uD800-\uDBFF](?![\uDC00-\uDFFF])|(?<![\uD800-\uDBFF])[\uDC00-\uDFFF]"
)


def sanitize_surrogates(text: str) -> str:
    """移除字符串中未配对的 Unicode 代理字符（对应 TS ``sanitizeSurrogates``）。

    有效的 emoji 和其他 BMP 之外的字符使用正确配对的代理，不会被影响。
    """
    return _UNPAIRED_SURROGATE_PATTERN.sub("", text)