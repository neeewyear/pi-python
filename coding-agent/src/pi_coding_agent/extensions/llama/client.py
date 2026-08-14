"""Llama 客户端。

简化版实现，使用 httpx 替代 fetch。
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel

# ============================================================================
# Types
# ============================================================================

LlamaModelStatus = Literal["unloaded", "loading", "loaded", "downloading", "sleeping"]
"""Llama 模型状态。"""


class LlamaModelStatusInfo(BaseModel):
    """模型状态信息。"""

    value: LlamaModelStatus
    args: list[str] | None = None
    failed: bool | None = None
    exit_code: int | None = None
    progress: dict[str, dict[str, int]] | None = None


class LlamaModelMeta(BaseModel):
    """模型元数据。"""  

    n_ctx: int | None = None
    n_ctx_train: int | None = None
    size: int | None = None
    ftype: str | None = None


class LlamaModelArchitecture(BaseModel):
    """模型架构信息。"""

    input_modalities: list[str] | None = None
    output_modalities: list[str] | None = None


class LlamaModelInfo(BaseModel):
    """Llama 模型信息。"""

    id: str
    aliases: list[str] | None = None
    status: LlamaModelStatusInfo
    architecture: LlamaModelArchitecture | None = None
    source: str | None = None
    meta: LlamaModelMeta | None = None


class LlamaModelsResponse(BaseModel):
    """模型列表响应。"""

    data: list[LlamaModelInfo]
    object: str | None = None


class LlamaModelEvent(BaseModel):
    """模型事件。"""

    model: str
    event: str
    data: Any = None


class LlamaProgress(BaseModel):
    """加载/下载进度。"""

    message: str
    ratio: float | None = None
    detail: str | None = None


# ============================================================================
# Helper Functions
# ============================================================================

ModelStatusDict = dict[str, Any]


def _error_message(payload: object, fallback: str) -> str:
    """从错误响应中提取错误消息。"""
    if not isinstance(payload, dict):
        return fallback
    error = payload.get("error")
    if not isinstance(error, dict):
        return fallback
    message = error.get("message")
    return str(message) if isinstance(message, str) and message else fallback


def _is_model_info(value: object) -> bool:
    """检查值是否为有效的模型信息。"""
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("id"), str) and isinstance(
        value.get("status", {}).get("value"), str
    )


def _link_signal(
    source: asyncio.Event | None, target: asyncio.Event
) -> Callable[[], None]:
    """将源信号链接到目标事件。"""
    if source is None:
        return lambda: None

    def _on_abort() -> None:
        target.set()

    return _on_abort


def _parse_load_progress(data: object) -> LlamaProgress | None:
    """解析加载进度。"""
    if not isinstance(data, dict):
        return None
    progress = data.get("progress")
    if not isinstance(progress, dict):
        return None
    stage: str | None = None
    if isinstance(progress.get("current"), str):
        stage = progress["current"]
    elif isinstance(progress.get("stage"), str):
        stage = progress["stage"]
    stages: list[str] = []
    if isinstance(progress.get("stages"), list):
        stages = [s for s in progress["stages"] if isinstance(s, str)]
    stage_ratio: float | None = None
    if isinstance(progress.get("value"), (int, float)):
        stage_ratio = max(0.0, min(1.0, float(progress["value"])))
    ratio = stage_ratio
    if stage and stages:
        index = stages.index(stage) if stage in stages else -1
        if index >= 0:
            ratio = (index + (stage_ratio or 0)) / len(stages)
    return LlamaProgress(
        message=f"Loading {stage.replace('_', ' ')}" if stage else "Loading model",
        ratio=ratio,
    )


def _parse_download_progress(data: object) -> LlamaProgress | None:
    """解析下载进度。"""
    if not isinstance(data, dict):
        return None
    nested = data.get("progress")
    files: dict[str, object] = (
        nested if isinstance(nested, dict) else data  # type: ignore[assignment]
    )
    done = 0
    total = 0
    for value in files.values():
        if not isinstance(value, dict):
            continue
        entry_done = value.get("done")
        entry_total = value.get("total")
        if not isinstance(entry_done, (int, float)) or not isinstance(
            entry_total, (int, float)
        ):
            continue
        done += int(entry_done)
        total += int(entry_total)
    if total <= 0:
        return None
    return LlamaProgress(
        message="Downloading model",
        ratio=done / total,
        detail=f"{_format_bytes(done)} / {_format_bytes(total)}",
    )


def _format_bytes(bytes_: int) -> str:
    """格式化字节数。"""
    if bytes_ < 1024:
        return f"{bytes_} B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = bytes_ / 1024
    unit = units[0]
    for i in range(1, len(units)):
        if value < 1024:
            break
        value /= 1024
        unit = units[i]
    precision = 1 if value >= 10 else 2
    return f"{value:.{precision}f} {unit}"


def normalize_llama_server_url(value: str) -> str:
    """规范化 llama.cpp 服务器 URL。"""
    from urllib.parse import urlparse, urlunparse

    value = value.strip()
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Server URL must use http or https")
    path = parsed.path.rstrip("/")
    path = path.removesuffix("/v1")
    if not path:
        path = "/"
    clean = urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))
    return clean.rstrip("/")


def llama_inference_url(server_url: str) -> str:
    """获取推理 URL。"""
    return f"{normalize_llama_server_url(server_url)}/v1"


# ============================================================================
# LlamaClient
# ============================================================================


class LlamaClient:
    """Llama.cpp 路由器客户端。"""

    def __init__(self, server_url: str, api_key: str | None = None) -> None:
        self.server_url = normalize_llama_server_url(server_url)
        self.api_key = api_key

    async def _request(
        self,
        path: str,
        method: str = "GET",
        body: object | None = None,
        signal: asyncio.Event | None = None,
    ) -> Any:
        """发送 HTTP 请求。"""
        import httpx

        headers: dict[str, str] = {}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = 15.0
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            try:
                response = await client.request(
                    method,
                    f"{self.server_url}{path}",
                    headers=headers,
                    json=body if body is not None else None,
                )
                payload: Any = None
                try:
                    payload = response.json()
                except Exception:
                    payload = None
                if not response.is_success:
                    raise RuntimeError(
                        _error_message(
                            payload, f"llama.cpp returned HTTP {response.status_code}"
                        )
                    )
                return payload
            except httpx.TimeoutException:
                raise RuntimeError("llama.cpp request timed out") from None

    async def list(
        self, reload: bool = False, signal: asyncio.Event | None = None
    ) -> list[LlamaModelInfo]:  # type: ignore[valid-type]
        """获取模型列表。"""
        path = f"/models{'?reload=1' if reload else ''}"
        payload = await self._request(path, signal=signal)
        if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
            raise RuntimeError("llama.cpp returned an invalid model catalog")
        data = payload["data"]
        if not all(_is_model_info(item) for item in data):
            raise RuntimeError("Server is not running in llama.cpp router mode")
        return [LlamaModelInfo(**item) for item in data]

    async def load(self, model: str, signal: asyncio.Event | None = None) -> None:
        """加载模型。"""
        await self._request(
            "/models/load", method="POST", body={"model": model}, signal=signal
        )

    async def unload(self, model: str, signal: asyncio.Event | None = None) -> None:
        """卸载模型。"""
        await self._request(
            "/models/unload", method="POST", body={"model": model}, signal=signal
        )

    async def unload_and_wait(
        self, model: str, signal: asyncio.Event | None = None
    ) -> None:
        """卸载模型并等待完成。"""
        await self.unload(model, signal)
        while True:
            models = await self.list(signal=signal)
            entry = next((m for m in models if m.id == model), None)
            if not entry or entry.status.value == "unloaded":
                return
            await asyncio.sleep(0.1)

    async def download(self, model: str, signal: asyncio.Event | None = None) -> None:
        """下载模型。"""
        await self._request(
            "/models", method="POST", body={"model": model}, signal=signal
        )

    async def watch(
        self,
        on_event: Callable[[LlamaModelEvent], None],
        signal: asyncio.Event | None = None,
    ) -> None:
        """监听 SSE 事件。"""
        import httpx

        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        timeout = httpx.Timeout(None)
        async with (
            httpx.AsyncClient(timeout=timeout) as client,
            client.stream(
                "GET", f"{self.server_url}/models/sse", headers=headers
            ) as response,
        ):
            if not response.is_success:
                raise RuntimeError(
                    f"llama.cpp SSE returned HTTP {response.status_code}"
                )
            buffer = ""
            async for chunk in response.aiter_text():
                if signal and signal.is_set():
                    break
                buffer += chunk.replace("\r\n", "\n")
                while "\n\n" in buffer:
                    boundary = buffer.index("\n\n")
                    frame = buffer[:boundary]
                    buffer = buffer[boundary + 2 :]
                    data_lines = [
                        line[5:].strip()
                        for line in frame.split("\n")
                        if line.startswith("data:")
                    ]
                    data = "\n".join(data_lines)
                    if data:
                        try:
                            event_data = json.loads(data)
                            event = LlamaModelEvent(**event_data)
                            if event.model and event.event:
                                on_event(event)
                        except Exception:
                            pass

    async def load_and_wait(
        self,
        model: str,
        on_progress: Callable[[LlamaProgress], None],
        signal: asyncio.Event | None = None,
    ) -> LlamaModelInfo:
        """加载模型并等待完成。"""
        watcher = asyncio.Event()
        event_loaded = False
        event_error: str | None = None

        async def _watch() -> None:
            nonlocal event_loaded, event_error

            def _on_event(event: LlamaModelEvent) -> None:
                nonlocal event_loaded, event_error
                if event.model != model:
                    return
                if event.event not in ("model_status", "status_change"):
                    return
                data = event.data if isinstance(event.data, dict) else None
                if data:
                    status = data.get("status")
                    if status == "loaded":
                        event_loaded = True
                    if status == "unloaded":
                        event_error = "Model failed to load"
                progress = _parse_load_progress(event.data)
                if progress:
                    on_progress(progress)

            try:
                await self.watch(_on_event, watcher)
            except Exception:
                pass

        watch_task = asyncio.create_task(_watch())
        try:
            await self.load(model, signal)
            on_progress(LlamaProgress(message="Loading model"))
            while True:
                if signal and signal.is_set():
                    raise RuntimeError("Cancelled")
                models = await self.list(signal=signal)
                entry = next((m for m in models if m.id == model), None)
                if entry and entry.status.value == "loaded":
                    return entry
                if event_loaded and not entry:
                    return LlamaModelInfo(
                        id=model, status=LlamaModelStatusInfo(value="loaded")
                    )
                if (entry and entry.status.failed) or event_error:
                    raise RuntimeError(
                        event_error
                        or (
                            f"Model exited with code {entry.status.exit_code}"
                            if entry and entry.status.exit_code is not None
                            else "Model failed to load"
                        )
                    )
                await asyncio.sleep(0.25)
        finally:
            watcher.set()
            await watch_task

    async def download_and_wait(
        self,
        model: str,
        on_progress: Callable[[LlamaProgress], None],
        signal: asyncio.Event | None = None,
    ) -> list[LlamaModelInfo]:  # type: ignore[valid-type]
        """下载模型并等待完成。"""
        watcher = asyncio.Event()
        finished = False
        failure: str | None = None
        saw_downloading = False
        polls = 0

        async def _watch() -> None:
            nonlocal finished, failure, saw_downloading

            def _on_event(event: LlamaModelEvent) -> None:
                nonlocal finished, failure, saw_downloading
                if event.model != model:
                    return
                if event.event == "download_finished":
                    finished = True
                if event.event == "download_failed":
                    failure = _error_message(event.data, "Download failed")
                if event.event == "download_progress":
                    saw_downloading = True
                    progress = _parse_download_progress(event.data)
                    if progress:
                        on_progress(progress)

            try:
                await self.watch(_on_event, watcher)
            except Exception:
                pass

        watch_task = asyncio.create_task(_watch())
        try:
            await self.download(model, signal)
            on_progress(LlamaProgress(message="Downloading model"))
            while True:
                if signal and signal.is_set():
                    raise RuntimeError("Cancelled")
                if failure:
                    raise RuntimeError(failure)
                models = await self.list(signal=signal)
                polls += 1
                entry = next((m for m in models if m.id == model), None)
                if entry and entry.status.value == "downloading":
                    saw_downloading = True
                    progress = _parse_download_progress(entry.status.progress)
                    if progress:
                        on_progress(progress)
                elif finished or (entry and (saw_downloading or polls >= 2)):
                    return await self.list(reload=True, signal=signal)
                await asyncio.sleep(0.5)
        finally:
            watcher.set()
            await watch_task


__all__ = [
    "LlamaClient",
    "LlamaModelArchitecture",
    "LlamaModelEvent",
    "LlamaModelInfo",
    "LlamaModelMeta",
    "LlamaModelStatus",
    "LlamaModelStatusInfo",
    "LlamaModelsResponse",
    "LlamaProgress",
    "llama_inference_url",
    "normalize_llama_server_url",
]
