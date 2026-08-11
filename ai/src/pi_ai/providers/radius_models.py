"""模型目录（对应 ``radius``）。

TODO: 在模型数据迁移后，从 JSON 数据文件加载实际模型数据。
注意：TS 侧不存在对应的 ``radius.models.ts``，此处仅为占位。
"""

from __future__ import annotations

from typing import Any

from ..model_catalog import flatten_model_catalog

# TODO: 从 JSON 数据文件加载后替换
# 当前使用空字典作为占位
_MODEL_DATA: dict[str, dict[str, Any]] = {}

RADIUS_MODELS: dict[str, Any] = flatten_model_catalog("radius", _MODEL_DATA)