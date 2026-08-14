"""会话搜索。

提供全量扫描搜索实现：列出所有会话，逐条查找包含目标文本的条目。
"""

from __future__ import annotations

import orjson
from pydantic import BaseModel

from pi_agent.harness.types import FileError
from pi_agent.result import Result
from .types import (
    EntryQuery,
    SessionError,
    SessionErrorCode,
    SessionMetadata,
    SessionRepo,
)


class SessionSearchOptions(BaseModel):
    """搜索选项。"""

    text: str
    cwd: str | None = None


class SessionSearchHit(BaseModel):
    """搜索命中。"""

    metadata: SessionMetadata
    entry_id: str
    timestamp: str
    snippet: str | None = None
    score: float | None = None


class SessionSearch:
    """会话搜索接口。"""

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        """搜索命中（子类实现）。"""
        raise NotImplementedError


def get_file_system_result_or_throw(
    value: Result[object, FileError], message: str
) -> object:
    """把 ``FileError`` 结果转为 ``SessionError`` 抛出（not_found 保留语义）。"""
    if not value.is_ok():
        code: SessionErrorCode = (
            "not_found" if value.error.code == "not_found" else "storage"
        )
        raise SessionError(code, f"{message}: {value.error.message}", value.error)
    return value.value


class ScanningSessionSearch(SessionSearch):
    """全量扫描搜索实现。"""

    _source: SessionRepo

    def __init__(self, source: SessionRepo) -> None:
        self._source = source

    async def search(self, options: SessionSearchOptions) -> list[SessionSearchHit]:
        normalized = options.text.strip().lower()
        if not normalized:
            return []
        hits: list[SessionSearchHit] = []
        for metadata in await self._source.list():
            cwd = getattr(metadata, "cwd", None)
            if options.cwd is not None and cwd != options.cwd:
                continue
            session = await self._source.open(metadata)
            for entry in await session.find_entries(EntryQuery(order="oldestFirst")):
                payload = orjson.dumps(entry.model_dump(mode="json")).decode("utf-8")
                if normalized not in payload.lower():
                    continue
                hits.append(
                    SessionSearchHit(
                        metadata=metadata,
                        entry_id=entry.id,
                        timestamp=orjson.dumps(entry.timestamp).decode("utf-8"),
                        snippet=payload,
                    )
                )
        return hits


def create_scanning_session_search(source: SessionRepo) -> SessionSearch:
    """创建全量扫描搜索。"""
    return ScanningSessionSearch(source)
