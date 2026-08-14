"""Bedrock Provider 公共模块。"""

from __future__ import annotations

from .api.bedrock_converse_stream import stream, stream_simple

bedrock_provider_module = {
    "stream": stream,
    "stream_simple": stream_simple,
}