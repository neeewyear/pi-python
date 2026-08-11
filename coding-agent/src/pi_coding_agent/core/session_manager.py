"""会话管理（对应 TS ``core/session-manager.ts`` 的简化版）。

提供会话条目类型定义（Pydantic 模型）和 ``SessionManager`` 类，
用于管理会话文件的 CRUD 操作。
"""

from __future__ import annotations

import datetime
import uuid
from pathlib import Path
from typing import Any, Generic, Literal, TypeAlias, TypeVar, cast

import orjson
from pi_agent.types import AgentMessage
from pi_ai.types import ImageContent, TextContent, Usage
from pydantic import BaseModel, Field

from ..config import get_sessions_dir

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

CURRENT_SESSION_VERSION = 3
"""当前会话版本。"""

T = TypeVar("T")
"""泛型类型变量，用于 CompactionEntry、BranchSummaryEntry 等。"""


# ---------------------------------------------------------------------------
# 会话条目类型
# ---------------------------------------------------------------------------


class SessionHeader(BaseModel):
    """会话头部（对应 TS ``SessionHeader``）。"""

    type: Literal["session"] = "session"
    version: int | None = None
    id: str
    timestamp: str
    cwd: str
    parent_session: str | None = None


class NewSessionOptions(BaseModel):
    """新建会话选项（对应 TS ``NewSessionOptions``）。"""

    id: str | None = None
    parent_session: str | None = None


class SessionEntryBase(BaseModel):
    """会话条目基类（对应 TS ``SessionEntryBase``）。"""

    type: str
    id: str
    parent_id: str | None = None
    timestamp: str


class SessionMessageEntry(SessionEntryBase):
    """消息条目（对应 TS ``SessionMessageEntry``）。"""

    type: Literal["message"] = "message"
    message: AgentMessage


class ThinkingLevelChangeEntry(SessionEntryBase):
    """思考级别变更条目（对应 TS ``ThinkingLevelChangeEntry``）。"""

    type: Literal["thinking_level_change"] = "thinking_level_change"
    thinking_level: str


class ModelChangeEntry(SessionEntryBase):
    """模型变更条目（对应 TS ``ModelChangeEntry``）。"""

    type: Literal["model_change"] = "model_change"
    provider: str
    model_id: str


class CompactionEntry(SessionEntryBase, Generic[T]):
    """上下文压缩条目（对应 TS ``CompactionEntry<T>``）。"""

    type: Literal["compaction"] = "compaction"
    summary: str
    first_kept_entry_id: str
    tokens_before: int
    details: T | None = None
    usage: Usage | None = None
    from_hook: bool | None = None


class BranchSummaryEntry(SessionEntryBase, Generic[T]):
    """分支摘要条目（对应 TS ``BranchSummaryEntry<T>``）。"""

    type: Literal["branch_summary"] = "branch_summary"
    from_id: str
    summary: str
    details: T | None = None
    usage: Usage | None = None
    from_hook: bool | None = None


class CustomEntry(SessionEntryBase, Generic[T]):
    """自定义条目（扩展存储，对应 TS ``CustomEntry<T>``）。

    不参与 LLM 上下文。
    """

    type: Literal["custom"] = "custom"
    custom_type: str
    data: T | None = None


class LabelEntry(SessionEntryBase):
    """标签条目（用户标记，对应 TS ``LabelEntry``）。"""

    type: Literal["label"] = "label"
    target_id: str
    label: str | None = None


class SessionInfoEntry(SessionEntryBase):
    """会话元信息条目（对应 TS ``SessionInfoEntry``）。"""

    type: Literal["session_info"] = "session_info"
    name: str | None = None


class CustomMessageEntry(SessionEntryBase, Generic[T]):
    """自定义消息条目（参与 LLM 上下文，对应 TS ``CustomMessageEntry<T>``）。"""

    type: Literal["custom_message"] = "custom_message"
    custom_type: str
    content: str | list[TextContent | ImageContent]
    details: T | None = None
    display: bool = True


# ---------------------------------------------------------------------------
# 会话条目联合
# ---------------------------------------------------------------------------

SessionEntry: TypeAlias = (
    SessionMessageEntry
    | ThinkingLevelChangeEntry
    | ModelChangeEntry
    | CompactionEntry[object]
    | BranchSummaryEntry[object]
    | CustomEntry[object]
    | CustomMessageEntry[object]
    | LabelEntry
    | SessionInfoEntry
)
"""会话条目联合类型（对应 TS ``SessionEntry``）。"""

FileEntry: TypeAlias = SessionHeader | SessionEntry
"""文件条目（包含头部，对应 TS ``FileEntry``）。"""


class SessionTreeNode(BaseModel):
    """会话树节点（对应 TS ``SessionTreeNode``）。"""

    entry: SessionEntry
    children: list[SessionTreeNode] = Field(default_factory=list)
    label: str | None = None
    label_timestamp: str | None = None


class SessionContext(BaseModel):
    """会话上下文（对应 TS ``SessionContext``）。"""

    messages: list[AgentMessage] = Field(default_factory=list)
    thinking_level: str = "off"
    model: dict[str, str] | None = None


class SessionInfo(BaseModel):
    """会话摘要信息（对应 TS ``SessionInfo``）。"""

    path: str
    id: str
    cwd: str = ""
    name: str | None = None
    parent_session_path: str | None = None
    created: str
    modified: str
    message_count: int = 0
    first_message: str = ""
    all_messages_text: str = ""


# ---------------------------------------------------------------------------
# SessionManager
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    """获取当前 UTC 时间的 ISO 格式字符串。"""
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class SessionManager:
    """会话管理器（对应 TS ``SessionManager`` 的简化版）。

    管理会话条目（JSONL 格式）的增删改查操作。
    """

    def __init__(
        self,
        cwd: str,
        session_dir: str | None = None,
        persist: bool = True,
    ) -> None:
        self._cwd = str(Path(cwd).resolve())
        self._session_dir = (
            str(Path(session_dir).resolve()) if session_dir else str(get_sessions_dir())
        )
        self._persist = persist
        self._session_id: str = ""
        self._session_file: Path | None = None
        self._entries: list[FileEntry] = []
        self._by_id: dict[str, SessionEntry] = {}
        self._leaf_id: str | None = None

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _generate_id(self) -> str:
        """生成短 ID（8 位 hex）。"""
        for _ in range(100):
            id_ = uuid.uuid4().hex[:8]
            if id_ not in self._by_id:
                return id_
        return uuid.uuid4().hex

    def _build_index(self) -> None:
        """重建条目索引。"""
        self._by_id.clear()
        self._leaf_id = None
        for entry in self._entries:
            if isinstance(entry, SessionHeader):
                continue
            self._by_id[entry.id] = entry
            self._leaf_id = entry.id

    def _read_file(self) -> list[FileEntry]:
        """读取会话文件。"""
        if not self._session_file or not self._session_file.exists():
            return []
        try:
            raw = self._session_file.read_bytes()
            lines = raw.decode("utf-8").strip().split("\n")
            entries: list[FileEntry] = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = orjson.loads(line)
                    if isinstance(data, dict):
                        entry_type = data.get("type", "")
                        if entry_type == "session":
                            entries.append(SessionHeader(**data))
                        else:
                            parsed = self._parse_entry(data)
                            if parsed is not None:
                                entries.append(parsed)
                except orjson.JSONDecodeError:
                    continue
            return entries
        except (FileNotFoundError, OSError):
            return []

    @staticmethod
    def _parse_entry(data: dict[str, object]) -> SessionEntry | None:
        """根据 type 字段解析条目。"""
        d = cast(dict[str, Any], data)
        entry_type = data.get("type")
        if entry_type == "message":
            return SessionMessageEntry(**d)
        elif entry_type == "thinking_level_change":
            return ThinkingLevelChangeEntry(**d)
        elif entry_type == "model_change":
            return ModelChangeEntry(**d)
        elif entry_type == "compaction":
            return CompactionEntry(**d)
        elif entry_type == "branch_summary":
            return BranchSummaryEntry(**d)
        elif entry_type == "custom":
            return CustomEntry(**d)
        elif entry_type == "custom_message":
            return CustomMessageEntry(**d)
        elif entry_type == "label":
            return LabelEntry(**d)
        elif entry_type == "session_info":
            return SessionInfoEntry(**d)
        return None

    def _append_entry(self, entry: SessionEntry) -> None:
        """追加条目。"""
        self._entries.append(entry)
        self._by_id[entry.id] = entry
        self._leaf_id = entry.id
        if self._persist and self._session_file:
            self._session_file.parent.mkdir(parents=True, exist_ok=True)
            line = orjson.dumps(entry.model_dump(mode="json")).decode("utf-8")
            with self._session_file.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def get_cwd(self) -> str:
        return self._cwd

    def get_session_dir(self) -> str:
        return self._session_dir

    def get_session_id(self) -> str:
        return self._session_id

    def get_session_file(self) -> str | None:
        return str(self._session_file) if self._session_file else None

    def get_leaf_id(self) -> str | None:
        return self._leaf_id

    def get_leaf_entry(self) -> SessionEntry | None:
        return self._by_id.get(self._leaf_id) if self._leaf_id else None

    def get_entry(self, id_: str) -> SessionEntry | None:
        return self._by_id.get(id_)

    def get_header(self) -> SessionHeader | None:
        for entry in self._entries:
            if isinstance(entry, SessionHeader):
                return entry
        return None

    def get_entries(self) -> list[SessionEntry]:
        """获取所有会话条目（排除头部）。"""
        return [e for e in self._entries if not isinstance(e, SessionHeader)]

    def get_branch(self, from_id: str | None = None) -> list[SessionEntry]:
        """获取从根到指定条目的路径。"""
        path: list[SessionEntry] = []
        start_id = from_id or self._leaf_id
        if start_id is None:
            return path
        current = self._by_id.get(start_id)
        while current:
            path.append(current)
            current = self._by_id.get(current.parent_id) if current.parent_id else None
        path.reverse()
        return path

    def new_session(self, options: NewSessionOptions | None = None) -> str | None:
        """创建新会话。"""
        self._session_id = options.id if options and options.id else self._generate_id()
        timestamp = _now_iso()
        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            id=self._session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            parent_session=options.parent_session if options else None,
        )
        self._entries = [header]
        self._by_id.clear()
        self._leaf_id = None

        if self._persist:
            file_timestamp = timestamp.replace(":", "-").replace(".", "-")
            self._session_file = (
                Path(self._session_dir) / f"{file_timestamp}_{self._session_id}.jsonl"
            )
            self._session_file.parent.mkdir(parents=True, exist_ok=True)
            self._session_file.write_bytes(
                orjson.dumps(header.model_dump(mode="json")) + b"\n"
            )

        return str(self._session_file) if self._session_file else None

    def append_message(self, message: AgentMessage) -> str:
        """追加消息条目。"""
        entry = SessionMessageEntry(
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            message=message,
        )
        self._append_entry(entry)
        return entry.id

    def append_thinking_level_change(self, thinking_level: str) -> str:
        """追加思考级别变更条目。"""
        entry = ThinkingLevelChangeEntry(
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            thinking_level=thinking_level,
        )
        self._append_entry(entry)
        return entry.id

    def append_model_change(self, provider: str, model_id: str) -> str:
        """追加模型变更条目。"""
        entry = ModelChangeEntry(
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            provider=provider,
            model_id=model_id,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_entry(self, custom_type: str, data: object = None) -> str:
        """追加自定义条目。"""
        entry = CustomEntry[object](
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            custom_type=custom_type,
            data=data,
        )
        self._append_entry(entry)
        return entry.id

    def branch(self, branch_from_id: str) -> None:
        """分支到指定条目。"""
        if branch_from_id not in self._by_id:
            raise ValueError(f"Entry {branch_from_id} not found")
        self._leaf_id = branch_from_id

    def reset_leaf(self) -> None:
        """重置叶子指针。"""
        self._leaf_id = None

    def get_session_name(self) -> str | None:
        """获取会话名称。"""
        for entry in reversed(self._entries):
            if isinstance(entry, SessionInfoEntry) and entry.name is not None:
                return entry.name
        return None

    def append_session_info(self, name: str) -> None:
        """追加会话信息条目。"""
        entry = SessionInfoEntry(
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            name=name,
        )
        self._append_entry(entry)

    def append_compaction(
        self,
        summary: str,
        first_kept_entry_id: str,
        tokens_before: int,
        details: object = None,
        from_extension: bool = False,
        usage: Usage | None = None,
    ) -> str:
        """追加压缩条目。"""
        entry = CompactionEntry[object](
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            summary=summary,
            first_kept_entry_id=first_kept_entry_id,
            tokens_before=tokens_before,
            details=details,
            usage=usage,
            from_hook=from_extension,
        )
        self._append_entry(entry)
        return entry.id

    def append_label_change(self, target_id: str, label: str | None = None) -> str:
        """追加标签条目。"""
        entry = LabelEntry(
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            target_id=target_id,
            label=label,
        )
        self._append_entry(entry)
        return entry.id

    def append_custom_message_entry(
        self,
        custom_type: str,
        content: str | list[TextContent | ImageContent],
        display: bool = True,
        details: object = None,
    ) -> str:
        """追加自定义消息条目。"""
        entry = CustomMessageEntry[object](
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            custom_type=custom_type,
            content=content,
            display=display,
            details=details,
        )
        self._append_entry(entry)
        return entry.id

    def build_session_context(self) -> SessionContext:
        """构建会话上下文。"""
        entries = self.get_branch()

        messages: list[AgentMessage] = []
        # In a simplified version, just return empty context
        return SessionContext(messages=messages)

    def branch_with_summary(
        self,
        new_leaf_id: str | None,
        summary: str,
        details: object = None,
        from_extension: bool = False,
        usage: Usage | None = None,
    ) -> str:
        """分支到指定条目并追加分支摘要。"""
        entry = BranchSummaryEntry[object](
            id=self._generate_id(),
            parent_id=self._leaf_id,
            timestamp=_now_iso(),
            from_id=self._leaf_id or "",
            summary=summary,
            details=details,
            usage=usage,
            from_hook=from_extension,
        )
        self._append_entry(entry)
        if new_leaf_id is not None:
            self.branch(new_leaf_id)
        return entry.id

    @classmethod
    def create(
        cls,
        cwd: str,
        session_dir: str | None = None,
        options: NewSessionOptions | None = None,
    ) -> SessionManager:
        """创建新会话。"""
        mgr = cls(cwd, session_dir)
        mgr.new_session(options)
        return mgr

    @classmethod
    def open(cls, path: str, cwd_override: str | None = None) -> SessionManager:
        """打开现有会话文件。"""
        file_path = Path(path).resolve()
        cwd = cwd_override or str(file_path.parent)
        mgr = cls.__new__(cls)
        mgr._cwd = cwd
        mgr._session_dir = str(file_path.parent)
        mgr._persist = True
        mgr._session_file = file_path
        mgr._entries = mgr._read_file()
        mgr._session_id = ""
        for entry in mgr._entries:
            if isinstance(entry, SessionHeader):
                mgr._session_id = entry.id
                break
        mgr._build_index()
        return mgr

    def is_persisted(self) -> bool:
        """返回会话是否持久化。"""
        return self._persist

    def create_branched_session(self, leaf_id: str) -> str | None:
        """创建分支会话，返回新会话文件路径。

        Args:
            leaf_id: 分支目标条目 ID。

        Returns:
            新会话文件路径，如果未持久化则返回 None。
        """
        path = self.get_branch(leaf_id)
        if not path:
            raise ValueError(f"Entry {leaf_id} not found")

        # 移除 LabelEntry（与 TS 实现一致）
        path_without_labels: list[SessionEntry] = []
        path_parent_id: str | None = None
        for entry in path:
            if isinstance(entry, LabelEntry):
                continue
            entry_copy = entry.model_copy(update={"parent_id": path_parent_id})
            path_without_labels.append(entry_copy)
            path_parent_id = entry.id

        new_session_id = uuid.uuid4().hex
        timestamp = _now_iso()
        file_timestamp = timestamp.replace(":", "-").replace(".", "-")
        new_session_file = (
            Path(self._session_dir) / f"{file_timestamp}_{new_session_id}.jsonl"
        )
        new_session_file.parent.mkdir(parents=True, exist_ok=True)

        previous_session_file = str(self._session_file) if self._session_file else None
        header = SessionHeader(
            version=CURRENT_SESSION_VERSION,
            id=new_session_id,
            timestamp=timestamp,
            cwd=self._cwd,
            parent_session=previous_session_file,
        )

        with new_session_file.open("wb") as f:
            f.write(orjson.dumps(header.model_dump(mode="json")) + b"\n")
            for entry in path_without_labels:
                f.write(orjson.dumps(entry.model_dump(mode="json")) + b"\n")

        return str(new_session_file)

    @classmethod
    def in_memory(cls, cwd: str = "") -> SessionManager:
        """创建纯内存会话（不持久化）。"""
        mgr = cls.__new__(cls)
        mgr._cwd = str(Path(cwd).resolve()) if cwd else str(Path.cwd())
        mgr._session_dir = ""
        mgr._persist = False
        mgr._session_file = None
        mgr._session_id = ""
        mgr._entries = []
        mgr._by_id = {}
        mgr._leaf_id = None
        mgr.new_session()
        return mgr


__all__ = [
    "CURRENT_SESSION_VERSION",
    "BranchSummaryEntry",
    "CompactionEntry",
    "CustomEntry",
    "CustomMessageEntry",
    "FileEntry",
    "LabelEntry",
    "ModelChangeEntry",
    "NewSessionOptions",
    "SessionContext",
    "SessionEntry",
    "SessionEntryBase",
    "SessionHeader",
    "SessionInfo",
    "SessionInfoEntry",
    "SessionManager",
    "SessionMessageEntry",
    "SessionTreeNode",
    "ThinkingLevelChangeEntry",
]
