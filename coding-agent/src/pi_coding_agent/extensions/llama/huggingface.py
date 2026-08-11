"""HuggingFace 集成（对应 TS ``extensions/llama/huggingface.ts``）。

简化版实现，使用 httpx 替代 fetch。
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# ============================================================================
# Constants
# ============================================================================

DEFAULT_HUGGING_FACE_URL = "https://huggingface.co"
QUANTIZATION_PATTERN_PREFIX = (
    r"(?:^|[-_.])((?:UD-)?(?:IQ\d(?:_[A-Z0-9]+)+|Q\d(?:_[A-Z0-9]+)+|BF16|F16|F32|MXFP\d(?:_[A-Z0-9]+)*))$"
)
SHARD_SUFFIX = r"-\d{5}-of-\d{5}$"

# ============================================================================
# Types
# ============================================================================


class HuggingFaceModel(BaseModel):
    """HuggingFace 模型搜索结果（对应 TS ``HuggingFaceModel``）。"""

    id: str
    downloads: int


class HuggingFaceQuantization(BaseModel):
    """量化信息（对应 TS ``HuggingFaceQuantization``）。"""

    name: str
    size: int | None = None


class HuggingFaceModelDetails(BaseModel):
    """模型详情（对应 TS ``HuggingFaceModelDetails``）。"""

    id: str
    gated: bool | str  # false | "auto" | "manual"
    quantizations: list[HuggingFaceQuantization]


# ============================================================================
# Helper Functions
# ============================================================================

import re


def _payload_error(payload: object, fallback: str) -> str:
    """从错误响应中提取错误消息。"""
    if not isinstance(payload, dict):
        return fallback
    error = payload.get("error")
    return str(error) if isinstance(error, str) and error else fallback


def _parse_rate_limit_delay(value: str | None) -> int | None:
    """解析速率限制延迟。"""
    if value is None:
        return None
    match = re.search(r"(?:^|;)t=(\d+)", value)
    return int(match.group(1)) if match else None


async def _read_token(path: str) -> str | None:
    """从文件读取 token。"""
    try:
        import aiofiles

        async with aiofiles.open(path, mode="r") as f:
            token = (await f.read()).strip()
        return token or None
    except Exception:
        return None


async def find_hugging_face_token(
    env: dict[str, str] | None = None,
) -> str | None:
    """查找 HuggingFace token（对应 TS ``findHuggingFaceToken``）。"""
    if env is None:
        env = dict(os.environ)
    hf_token = env.get("HF_TOKEN", "").strip()
    if hf_token:
        return hf_token

    paths: list[str] = []
    hf_token_path = env.get("HF_TOKEN_PATH")
    if hf_token_path:
        paths.append(hf_token_path)

    hf_home = env.get("HF_HOME")
    if hf_home:
        paths.append(str(Path(hf_home) / "token"))

    xdg_cache = env.get("XDG_CACHE_HOME")
    if xdg_cache:
        paths.append(str(Path(xdg_cache) / "huggingface" / "token"))

    paths.append(str(Path.home() / ".cache" / "huggingface" / "token"))

    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        token = await _read_token(path)
        if token:
            return token
    return None


# ============================================================================
# HuggingFaceClient
# ============================================================================


class HuggingFaceClient:
    """HuggingFace API 客户端（对应 TS ``HuggingFaceClient``）。"""

    def __init__(
        self, token: str | None = None, base_url: str = DEFAULT_HUGGING_FACE_URL
    ) -> None:
        self.token = token
        self.base_url = base_url.rstrip("/")

    async def _request(
        self, path: str, signal: asyncio.Event | None = None
    ) -> Any:
        """发送 HTTP 请求。"""
        import httpx

        headers: dict[str, str] = {}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        timeout = 15.0
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
            try:
                response = await client.get(
                    f"{self.base_url}{path}", headers=headers
                )
                payload: Any = None
                try:
                    payload = response.json()
                except Exception:
                    payload = None
                if not response.is_success:
                    fallback = f"Hugging Face returned HTTP {response.status_code}"
                    if response.status_code == 429:
                        retry_after = response.headers.get("retry-after")
                        ratelimit = response.headers.get("ratelimit")
                        delay = (
                            int(retry_after)
                            if retry_after
                            else _parse_rate_limit_delay(ratelimit)
                        )
                        raise RuntimeError(
                            f"Hugging Face rate limit reached; retry in {delay}s"
                            if delay
                            else "Hugging Face rate limit reached"
                        )
                    raise RuntimeError(_payload_error(payload, fallback))
                return payload
            except httpx.TimeoutException:
                raise RuntimeError("Hugging Face request timed out") from None

    async def search(
        self, query: str, signal: asyncio.Event | None = None
    ) -> list[HuggingFaceModel]:
        """搜索模型（对应 TS ``search``）。"""
        from urllib.parse import urlencode

        params = urlencode(
            {
                "search": query,
                "filter": "gguf",
                "sort": "downloads",
                "direction": "-1",
                "limit": "20",
            }
        )
        payload = await self._request(f"/api/models?{params}", signal)
        if not isinstance(payload, list):
            raise RuntimeError("Hugging Face returned invalid search results")
        models: list[HuggingFaceModel] = []
        for value in payload:
            if not isinstance(value, dict):
                continue
            model_id = value.get("id")
            if not isinstance(model_id, str):
                continue
            downloads = value.get("downloads", 0)
            models.append(
                HuggingFaceModel(
                    id=model_id,
                    downloads=downloads if isinstance(downloads, (int, float)) else 0,
                )
            )
        return models

    async def details(
        self, id_: str, signal: asyncio.Event | None = None
    ) -> HuggingFaceModelDetails:
        """获取模型详情（对应 TS ``details``）。"""
        from urllib.parse import quote

        encoded_id = "/".join(quote(part) for part in id_.split("/"))
        payload = await self._request(f"/api/models/{encoded_id}?blobs=true", signal)
        if not isinstance(payload, dict):
            raise RuntimeError("Hugging Face returned invalid model details")
        model_id = payload.get("id", id_)
        gated = payload.get("gated", False)
        siblings = payload.get("siblings")

        # 解析量化信息
        sizes: dict[str, dict[str, int | bool]] = {}
        if isinstance(siblings, list):
            for sibling in siblings:
                if not isinstance(sibling, dict):
                    continue
                rfilename = sibling.get("rfilename")
                if not isinstance(rfilename, str) or not rfilename.lower().endswith(
                    ".gguf"
                ):
                    continue
                filename = rfilename.split("/")[-1]
                if not filename:
                    continue
                if filename.lower().startswith("mmproj"):
                    continue
                stem = filename[: -len(".gguf")]
                stem = re.sub(SHARD_SUFFIX, "", stem)
                match = re.search(QUANTIZATION_PATTERN_PREFIX, stem)
                if not match:
                    continue
                quantization = match.group(1).upper()
                current = sizes.get(quantization, {"total": 0, "complete": True})
                file_size = sibling.get("size")
                if isinstance(file_size, (int, float)):
                    current["total"] = int(current["total"]) + int(file_size)  # type: ignore[operator]
                else:
                    current["complete"] = False
                sizes[quantization] = current

        quantizations = [
            HuggingFaceQuantization(
                name=name,
                size=info["total"] if info["complete"] else None,  # type: ignore[arg-type]
            )
            for name, info in sorted(
                sizes.items(),
                key=lambda x: (
                    0 if x[0] == "Q4_K_M" else 1,
                    x[1]["total"] if isinstance(x[1]["total"], int) else 9223372036854775807,
                    x[0],
                ),
            )
        ]

        gated_value: bool | str = False
        if gated == "auto":
            gated_value = "auto"
        elif gated == "manual":
            gated_value = "manual"

        return HuggingFaceModelDetails(
            id=str(model_id) if isinstance(model_id, str) else id_,
            gated=gated_value,
            quantizations=quantizations,
        )


__all__ = [
    "DEFAULT_HUGGING_FACE_URL",
    "HuggingFaceClient",
    "HuggingFaceModel",
    "HuggingFaceModelDetails",
    "HuggingFaceQuantization",
    "find_hugging_face_token",
]