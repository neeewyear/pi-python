"""Tab 补全模块（命令补全 + 扩展命令补全）。

支持：
- TUI 内建命令补全（/help, /clear, /exit 等）
- 扩展命令补全（通过 ExtensionRunner 注册的命令）
- 部分输入匹配和循环补全
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .commands import BUILTIN_COMMANDS, get_builtin_command_names

if TYPE_CHECKING:
    from collections.abc import Sequence


# 补全候选类型
CompletionCandidate = str
"""补全候选项（完整的命令字符串，如 "/help"）。"""

GetExtensionCommandsFn = Callable[[], "Sequence[str]"]
"""获取扩展命令列表的回调函数。"""


class TabCompleter:
    """Tab 补全器。

    管理补全候选列表，支持：
    - 首次 Tab：显示第一个匹配的候选
    - 重复 Tab：循环显示下一个匹配的候选
    - 输入变化：重置候选列表，重新匹配
    """

    def __init__(
        self,
        get_extension_commands: GetExtensionCommandsFn | None = None,
    ) -> None:
        self._get_extension_commands = get_extension_commands
        # 当前匹配的候选列表
        self._candidates: list[str] = []
        # 当前候选索引（-1 表示未开始补全）
        self._index: int = -1
        # 触发补全时的输入文本，用于检测输入变化后重置
        self._trigger_input: str = ""

    # ------------------------------------------------------------------
    # 公共方法
    # ------------------------------------------------------------------

    def complete(
        self, text: str, cursor_position: int = -1
    ) -> tuple[str | None, int, int]:
        """执行一次 Tab 补全。

        Args:
            text: 当前输入框的全部文本。
            cursor_position: 光标位置（-1 表示末尾）。

        Returns:
            (completion, start, end) 三元组：
            - completion: 补全后的文本，或 None 表示没有匹配。
            - start: 补全替换的起始位置（光标前查找起始）。
            - end: 补全替换的结束位置（光标位置）。
        """
        # 确定要补全的单词范围
        if cursor_position < 0:
            cursor_position = len(text)

        # 从光标位置往前查找单词起始
        start = cursor_position
        while start > 0 and not text[start - 1].isspace():
            start -= 1
        prefix = text[start:cursor_position]

        # 只在以 / 开头时进行命令补全
        if not prefix.startswith("/"):
            return None, 0, 0

        # 如果输入变了，重置候选列表
        if text != self._trigger_input:
            self._candidates = self._find_candidates(prefix)
            self._index = -1
            self._trigger_input = text

        if not self._candidates:
            return None, start, cursor_position

        # 循环候选
        self._index += 1
        if self._index >= len(self._candidates):
            self._index = 0

        candidate = self._candidates[self._index]
        # 补全后的文本：前缀替换为完整候选，后面保持不变
        completed = text[:start] + candidate + text[cursor_position:]
        return completed, start, cursor_position

    def reset(self) -> None:
        """重置补全状态（输入变化或提交时调用）。"""
        self._candidates.clear()
        self._index = -1
        self._trigger_input = ""

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _find_candidates(self, prefix: str) -> list[str]:
        """查找匹配前缀的补全候选。"""
        candidates: list[str] = []

        # 1. TUI 内建命令
        for cmd_name in get_builtin_command_names():
            full_cmd = f"/{cmd_name}"
            if full_cmd.startswith(prefix):
                candidates.append(full_cmd)

        # 2. 扩展命令
        if self._get_extension_commands is not None:
            try:
                ext_commands = self._get_extension_commands()
                for cmd in ext_commands:
                    if cmd.startswith(prefix):
                        candidates.append(cmd)
            except Exception:
                pass

        # 去重并排序
        seen: set[str] = set()
        unique: list[str] = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        unique.sort()
        return unique