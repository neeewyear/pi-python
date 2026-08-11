"""Agent 内部消息队列（对应 ``agent.ts`` 的 ``PendingMessageQueue``）。

支持两种排空模式：
- ``"all"``：一次性返回全部消息并清空队列
- ``"one-at-a-time"``：每次只返回队首消息
"""

from __future__ import annotations

from .types import AgentMessage, QueueMode


class PendingMessageQueue:
    """内部消息队列（对应 TS ``PendingMessageQueue``）。

    用于 steering 和 follow-up 消息的排队与排空。
    """

    def __init__(self, mode: QueueMode = "one-at-a-time") -> None:
        self.mode: QueueMode = mode
        self._messages: list[AgentMessage] = []

    def enqueue(self, message: AgentMessage) -> None:
        """入队一条消息。"""
        self._messages.append(message)

    def has_items(self) -> bool:
        """是否有待处理的消息。"""
        return len(self._messages) > 0

    def drain(self) -> list[AgentMessage]:
        """排空队列。

        - ``"all"``：返回全部消息并清空队列
        - ``"one-at-a-time"``：返回队首消息（单条）
        """
        if self.mode == "all":
            drained = list(self._messages)
            self._messages.clear()
            return drained

        if not self._messages:
            return []
        first = self._messages[0]
        self._messages = self._messages[1:]
        return [first]

    def clear(self) -> None:
        """清空队列。"""
        self._messages.clear()
