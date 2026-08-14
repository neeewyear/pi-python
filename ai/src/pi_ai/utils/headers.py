"""HTTP 头处理工具。"""

from __future__ import annotations

from ..types import ProviderHeaders


def headers_to_record(headers: dict[str, str]) -> dict[str, str]:
    """将 ``Headers`` 对象转换为普通 dict。

    Python 侧直接使用 dict，此函数保持接口一致。
    """
    return dict(headers)


def provider_headers_to_record(
    headers: ProviderHeaders | None,
) -> dict[str, str] | None:
    """将 ``ProviderHeaders`` 转换为普通 dict，过滤掉 None 值。"""
    if not headers:
        return None
    result: dict[str, str] = {}
    for key, value in headers.items():
        if value is not None:
            result[key] = value
    return result if result else None