"""模型目录查询与过滤（对应 ``model-catalog.ts``）。

提供 ``flatten_model_catalog`` 工具函数，用于将分组的模型目录扁平化为
单层模型映射。
"""

from __future__ import annotations

from typing import Any


def flatten_model_catalog(
    provider: str,
    groups: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """将分组的模型目录扁平化为单层映射（对应 TS ``flattenModelCatalog``）。

    Args:
        provider: Provider ID（当前仅用于类型标注，Python 侧无编译时类型约束）。
        groups: 分组模型字典，每组为 ``{model_id: model_record}``。

    Returns:
        合并后的扁平字典 ``{model_id: model_record}``。
    """
    result: dict[str, Any] = {}
    for _group_name, group_models in groups.items():
        result.update(group_models)
    return result