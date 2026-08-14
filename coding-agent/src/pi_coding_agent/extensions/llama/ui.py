"""UI 组件。

简化占位实现，TUI 组件依赖 pi-tui 框架，Python 侧暂不实现完整 TUI。
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any, Literal

from pydantic import BaseModel

from .client import LlamaModelInfo, LlamaProgress

# ============================================================================
# Types
# ============================================================================

LlamaManagerAction = (
    dict[Literal["type"], Literal["model"]]
    | dict[Literal["type"], Literal["download"]]
    | dict[Literal["type"], Literal["close"]]
)
"""模型管理器操作。

用法：``{"type": "model", "model": ...}`` 或 ``{"type": "download"}`` 或 ``{"type": "close"}``
"""


class ProgressState(BaseModel):
    """进度状态。"""

    title: str
    model: str
    message: str
    ratio: float | None = None
    detail: str | None = None


# ============================================================================
# LlamaUi 接口
# ============================================================================

"""Llama UI 接口。

简化版，提供基础 UI 交互方法。
"""

class LlamaUi:
    """Llama UI 接口。  

    简化版，提供基础 UI 交互方法。
    """

    def __init__(
        self,
        notify: Callable[[str, str | None], None] | None = None,
        select: Callable[
            [str, list[str]], Awaitable[str | None]
        ]
        | None = None,
        confirm: Callable[[str, str], Awaitable[bool]] | None = None,
        show_status: Callable[[str, str], None] | None = None,
    ) -> None:
        self._notify = notify
        self._select = select
        self._confirm = confirm
        self._show_status = show_status

    async def show_models(
        self, server_url: str, models: list[LlamaModelInfo]
        ) -> Any:
        """显示模型列表。"""
        raise NotImplementedError("TUI components not implemented in Python")

    async def select(
        self, title: str, options: list[str]
        ) -> str | None:
        """显示选择器。"""
        if self._select:
            return await self._select(title, options)
        raise NotImplementedError("TUI components not implemented in Python")

    async def confirm(self, title: str, message: str) -> bool:
        """显示确认对话框。"""
        if self._confirm:
            return await self._confirm(title, message)
        raise NotImplementedError("TUI components not implemented in Python")

    async def connection_error(
        self, server_url: str, message: str
    ) -> Literal["retry", "close"]:
        """显示连接错误。"""
        choice = await self.select(
            f"llama.cpp unavailable\n{server_url}\n\n{message}",
            ["Retry", "Close"],
        )
        return "retry" if choice == "Retry" else "close"

    async def search_models(
        self,
        search: Callable[
            [str, asyncio.Event], Awaitable[list[Any]]
        ],
    ) -> str | None:
        """搜索模型。"""
        raise NotImplementedError("TUI components not implemented in Python")

    def show_status(self, title: str, message: str) -> None:
        """显示状态。"""
        if self._show_status:
            self._show_status(title, message)

    async def progress(self, state: ProgressState) -> None:
        """显示进度。"""
        # 简化版：只打印进度信息
        print(f"[{state.title}] {state.model}: {state.message}")

    def update_progress(self, state: ProgressState) -> None:
        """更新进度。"""
        # 简化版：只打印进度信息
        print(f"[{state.title}] {state.model}: {state.message} (ratio={state.ratio})")


# ============================================================================
# 辅助函数
# ============================================================================


def model_description(model: LlamaModelInfo) -> str:
    """获取模型描述文本。"""
    details: list[str] = []
    loaded = model.status.value in ("loaded", "sleeping")
    if loaded:
        details.append("loaded")
    elif model.status.value != "unloaded":
        details.append(model.status.value)
    if loaded:
        ctx = model.meta.n_ctx if model.meta else None
        if ctx is None and model.meta:
            ctx = model.meta.n_ctx_train
        if ctx:
            details.append(
                f"{ctx // 1000}k context" if ctx >= 1000 else f"{ctx} context"
            )
    return " · ".join(details)


async def show_llama_ui(
    ctx: Any, run: Callable[[LlamaUi], Awaitable[None]]
) -> None:
    """显示 Llama UI。

    简化版，只在非 TUI 模式下提供基础交互。
    """
    if ctx.mode != "tui":
        ctx.ui.notify("/llama is available in interactive mode", "warning")
        return
    ui = LlamaUi(
        notify=ctx.ui.notify,
        select=lambda title, options: ctx.ui.select(title, options, None),
        confirm=lambda title, message: ctx.ui.confirm(title, message, None),
        show_status=lambda title, message: ctx.ui.set_status(title, message),
    )
    await run(ui)


async def run_with_progress(
    ui: LlamaUi,
    options: dict[str, Any],
) -> dict[str, Any]:
    """运行带进度的操作。

    options 包含:
    - title: str
    - model: str
    - initial_message: str
    - cancel_title: str
    - cancel_message: str
    - run: Callable[[asyncio.Event, Callable[[LlamaProgress], None]], Awaitable[Any]]
    - cancel: Callable[[], Awaitable[None]]
    """
    controller = asyncio.Event()
    state = ProgressState(
        title=options["title"],
        model=options["model"],
        message=options["initial_message"],
    )

    async def update(progress: LlamaProgress) -> None:
        state.message = progress.message
        state.ratio = progress.ratio
        state.detail = progress.detail
        ui.update_progress(state)

    settled = asyncio.create_task(options["run"](controller, update))
    completed = False

    def _set_completed(task: asyncio.Task[Any]) -> None:
        nonlocal completed
        completed = True

    settled.add_done_callback(_set_completed)

    try:
        while not completed:
            done, _ = await asyncio.wait(
                [settled, asyncio.create_task(ui.progress(state))],
                return_when=asyncio.FIRST_COMPLETED,
            )
            if settled in done:
                break
            stop = await ui.confirm(
                options["cancel_title"], options["cancel_message"]
            )
            if not stop or completed:
                continue
            try:
                await options["cancel"]()
            finally:
                controller.set()
            await settled
            return {"cancelled": True}

        result = await settled
        if isinstance(result, BaseException):
            raise result
        return {"cancelled": False, "value": result}
    except Exception as e:
        raise e
    finally:
        controller.set()


__all__ = [
    "LlamaManagerAction",
    "LlamaUi",
    "ProgressState",
    "model_description",
    "run_with_progress",
    "show_llama_ui",
]