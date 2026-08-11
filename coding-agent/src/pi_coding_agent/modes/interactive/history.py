"""命令历史记录（内存 + 文件持久化）。

支持：
- 最近执行的命令列表（内存）
- 文件持久化到 JSON 文件
- 上/下箭头导航
- 最大历史条数限制
"""

from __future__ import annotations

import json
from pathlib import Path


class CommandHistory:
    """命令历史记录管理器。"""

    def __init__(self, max_history: int = 500, file_path: Path | None = None) -> None:
        self._max_history = max_history
        self._file_path = file_path
        self._entries: list[str] = []
        self._index: int = -1  # 当前导航位置（-1 = 新输入）
        self._load()

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def add(self, text: str) -> None:
        """添加一条命令到历史记录。"""
        if not text.strip():
            return
        # 避免连续重复
        if self._entries and self._entries[-1] == text:
            return
        self._entries.append(text)
        # 超出上限时移除最旧的
        if len(self._entries) > self._max_history:
            self._entries.pop(0)
        self._reset_index()
        self._save()

    def navigate_up(self) -> str | None:
        """上箭头：返回更早的历史项。"""
        if not self._entries:
            return None
        if self._index == -1:
            self._index = len(self._entries) - 1
        elif self._index <= 0:
            self._index = 0
        else:
            self._index -= 1
        return self._entries[self._index]

    def navigate_down(self) -> str | None:
        """下箭头：返回更新的历史项，到底部返回 None。"""
        if not self._entries or self._index == -1:
            return None
        self._index += 1
        if self._index >= len(self._entries):
            self._reset_index()
            return None
        return self._entries[self._index]

    def all(self) -> list[str]:
        """获取所有历史条目。"""
        return list(self._entries)

    def clear(self) -> None:
        """清空历史记录。"""
        self._entries.clear()
        self._reset_index()
        self._save()

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _reset_index(self) -> None:
        """重置导航索引到新输入位置。"""
        self._index = -1

    def _load(self) -> None:
        """从文件加载历史记录。"""
        if self._file_path is None:
            return
        try:
            if self._file_path.exists():
                data = json.loads(self._file_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._entries = [str(e) for e in data][-self._max_history :]
        except (json.JSONDecodeError, OSError):
            self._entries = []

    def _save(self) -> None:
        """持久化历史记录到文件。"""
        if self._file_path is None:
            return
        try:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)
            self._file_path.write_text(
                json.dumps(self._entries, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError:
            pass  # 持久化失败不阻塞使用
