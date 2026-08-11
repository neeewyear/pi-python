"""DeepSeek API 的 StreamFn 实现（OpenAI 兼容接口）。

通过 HTTP SSE 直接调用 DeepSeek API，不依赖代理服务器。
所有请求参数统一通过 ``create_deepseek_stream_fn`` 工厂传入：:

    from pi_ai.deepseek_provider import DeepSeekModel, create_deepseek_stream_fn
    from pi_ai.stream_fn import set_default_stream_fn

    stream_fn = create_deepseek_stream_fn(
        model_id="deepseek-chat",
        api_key="sk-xxx",        # 省略时依次从环境变量 / ~/.zshrc 解析
        base_url="https://api.deepseek.com",
        timeout_ms=60_000,
    )
    set_default_stream_fn(stream_fn)
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from collections.abc import AsyncIterator
from pathlib import Path

import httpx

from .types import (
    AssistantAbortedEvent,
    AssistantErrorEvent,
    AssistantMessage,
    AssistantMessageEvent,
    AssistantMessageSnapshot,
    AssistantStreamEnd,
    AssistantTextDelta,
    AssistantToolCallEnd,
    AssistantToolCallStart,
    AssistantToolCallUpdate,
    ContentBlock,
    Context,
    ImageContent,
    Message,
    Model,
    SimpleStreamOptions,
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
)

# ---------------------------------------------------------------------------
# DeepSeekModel
# ---------------------------------------------------------------------------


class DeepSeekModel:
    """满足 ``Model`` Protocol 的 DeepSeek 模型句柄。"""

    api: str = "chat"
    provider: str = "deepseek"
    model_id: str

    def __init__(self, model_id: str = "deepseek-v4-flash") -> None:
        self.model_id = model_id


# ---------------------------------------------------------------------------
# API key 解析
# ---------------------------------------------------------------------------


def _resolve_api_key(api_key: str | None) -> str:
    """解析 API key：显式参数 > 环境变量 > ``~/.zshrc`` 兜底。"""
    if api_key:
        return api_key
    env_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if env_key:
        return env_key
    zshrc = Path.home() / ".zshrc"
    if zshrc.exists():
        try:
            match = re.search(
                r'export\s+DEEPSEEK_API_KEY\s*=\s*["\']?([^"\'\n]+)["\']?',
                zshrc.read_text(encoding="utf-8", errors="ignore"),
            )
            if match:
                return match.group(1).strip()
        except OSError:
            pass
    return ""


# ---------------------------------------------------------------------------
# 消息格式转换
# ---------------------------------------------------------------------------


def _convert_message(msg: Message) -> dict[str, object]:
    """将内部 Message 转换为 OpenAI chat 格式。"""
    if msg.role == "user":
        content: list[dict[str, object]] = []
        for block in msg.content:
            if isinstance(block, TextContent):
                content.append({"type": "text", "text": block.text})
            elif isinstance(block, ImageContent):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:{block.mime_type};base64,{block.data}"
                        },
                    }
                )
        return {
            "role": "user",
            "content": content
            if len(content) != 1 or content[0]["type"] != "text"
            else content[0]["text"],
        }

    elif msg.role == "assistant":
        content_texts: list[str] = []
        tool_calls: list[dict[str, object]] = []
        for cb in msg.content:
            if isinstance(cb, TextContent):
                content_texts.append(cb.text)
            elif isinstance(cb, ToolCallContent):
                tool_calls.append(
                    {
                        "id": cb.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": cb.name,
                            "arguments": json.dumps(cb.args, ensure_ascii=False),
                        },
                    }
                )
        result: dict[str, object] = {"role": "assistant"}
        if content_texts:
            result["content"] = (
                content_texts[0]
                if len(content_texts) == 1
                else "\n".join(content_texts)
            )
        else:
            result["content"] = None
        if tool_calls:
            result["tool_calls"] = tool_calls
        return result

    elif msg.role == "toolResult":
        content_text = ""
        for block in msg.content:
            if isinstance(block, TextContent):
                content_text += block.text
        return {
            "role": "tool",
            "tool_call_id": msg.tool_call_id,
            "content": content_text,
        }

    return {"role": "user", "content": ""}


def _convert_tools(context: Context) -> list[dict[str, object]]:
    """将 AgentTool 列表转换为 OpenAI function 格式。"""
    if not context.tools:
        return []
    result: list[dict[str, object]] = []
    for tool in context.tools:
        result.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
        )
    return result


# ---------------------------------------------------------------------------
# 流式响应解析
# ---------------------------------------------------------------------------


def _parse_tool_call_args(raw: str) -> dict[str, object] | None:
    """把累积的 arguments 原始字符串解析为 JSON 对象。

    流式场景下单个 delta 的 arguments 通常是残缺 JSON 片段，
    必须累积完整后再解析；解析失败返回 None（继续累积）。
    """
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


# ---------------------------------------------------------------------------
# create_deepseek_stream_fn
# ---------------------------------------------------------------------------


def create_deepseek_stream_fn(
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    model_id: str | None = None,
    timeout_ms: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> DeepSeekStreamFn:
    """创建 DeepSeek StreamFn 工厂函数（统一在此传入参数配置）。

    Args:
        api_key: DeepSeek API key。为 None 时依次从环境变量
            ``DEEPSEEK_API_KEY`` 和 ``~/.zshrc`` 解析。
        base_url: DeepSeek API 基础 URL。
        model_id: 请求使用的模型 ID；为 None 时回退到循环传入的
            ``model.model_id``（即 ``AgentLoopConfig.model``）。
        timeout_ms: 单次请求超时（毫秒），None 表示不设超时。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度。
        max_retries: 请求失败时的重试次数。

    Raises:
        RuntimeError: 无法解析出任何 API key 时。
    """
    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY 未设置。请传入 api_key，或设置环境变量 "
            "DEEPSEEK_API_KEY，或在 ~/.zshrc 中配置: export DEEPSEEK_API_KEY=xxx"
        )
    return DeepSeekStreamFn(
        api_key=resolved_key,
        base_url=base_url,
        model_id=model_id,
        timeout_ms=timeout_ms,
        max_tokens=max_tokens,
        temperature=temperature,
        max_retries=max_retries,
    )


class DeepSeekStreamFn:
    """DeepSeek API 的 StreamFn 实现。

    直接调用 DeepSeek 的 OpenAI 兼容 chat completions 端点。
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com",
        model_id: str | None = None,
        timeout_ms: int | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        max_retries: int | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._timeout_ms = timeout_ms
        self._max_tokens = max_tokens
        self._temperature = temperature
        self._max_retries = max_retries

    def __call__(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None = None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """执行流式 LLM 请求，返回事件流（直接返回异步生成器）。"""
        return self._stream(model, context, options)

    async def _stream(
        self,
        model: Model,
        context: Context,
        options: SimpleStreamOptions | None,
    ) -> AsyncIterator[AssistantMessageEvent]:
        """内部流式生成器。"""
        messages = [_convert_message(msg) for msg in context.messages]
        tools = _convert_tools(context)

        # 模型 ID：工厂配置优先，未配置时回退到循环传入的 model
        request_model_id = self._model_id or model.model_id

        body: dict[str, object] = {
            "model": request_model_id,
            "messages": messages,
            "stream": True,
        }

        if context.system_prompt:
            body["messages"] = [
                {"role": "system", "content": context.system_prompt}
            ] + list(messages)

        if tools:
            body["tools"] = tools

        # 请求参数优先级：逐调用 options > 工厂默认值
        max_tokens_val = self._max_tokens
        temperature_val = self._temperature
        if options is not None:
            options_max_tokens = getattr(options, "max_tokens", None)
            options_temperature = getattr(options, "temperature", None)
            if options_max_tokens is not None:
                max_tokens_val = options_max_tokens
            if options_temperature is not None:
                temperature_val = options_temperature
        if max_tokens_val is not None:
            body["max_tokens"] = max_tokens_val
        if temperature_val is not None:
            body["temperature"] = temperature_val

        # 超时：options > 工厂配置 > 不设超时
        timeout_ms = None
        if options is not None and options.timeout_ms is not None:
            timeout_ms = options.timeout_ms
        elif self._timeout_ms is not None:
            timeout_ms = self._timeout_ms
        timeout = httpx.Timeout(None if timeout_ms is None else timeout_ms / 1000.0)

        # 重试：仅在工厂配置了 max_retries 时启用
        transport: httpx.AsyncHTTPTransport | None = None
        if self._max_retries is not None:
            transport = httpx.AsyncHTTPTransport(retries=self._max_retries)

        headers: dict[str, str] = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        url = f"{self._base_url}/v1/chat/completions"

        # 累积状态
        content_blocks: list[ContentBlock] = []
        current_text_idx: int | None = None
        current_tool_call_idx: dict[int, int] = {}  # index -> content_blocks index
        tool_call_args_raw: dict[int, str] = {}  # index -> 累积的 arguments 原始字符串
        finish_reason: str = "stop"
        usage_data: Usage = Usage()

        # 注意：except 子句中不能直接 yield —— 含 yield 的 except 会触发
        # CPython 关闭 async generator 时的 "async generator ignored GeneratorExit"。
        # 改为先捕获到局部变量，try/except 结束后再发射。
        error_event: AssistantErrorEvent | None = None
        abort_event: AssistantAbortedEvent | None = None

        try:
            async with httpx.AsyncClient(
                timeout=timeout, transport=transport
            ) as client:
                async with client.stream(
                    "POST",
                    url,
                    headers=headers,
                    json=body,
                ) as response:
                    if response.status_code != 200:
                        error_text = ""
                        async for chunk in response.aiter_bytes():
                            error_text += chunk.decode("utf-8", errors="replace")
                        yield AssistantErrorEvent(
                            error=f"DeepSeek API error: {response.status_code} - {error_text[:500]}"
                        )
                        yield AssistantAbortedEvent(
                            error=f"HTTP {response.status_code}"
                        )
                        return

                    # 发送初始 message_snapshot
                    initial_msg = _build_snapshot(
                        model,
                        content_blocks,
                        "pending",
                        usage_data,
                        model_id=request_model_id,
                    )
                    yield AssistantMessageSnapshot(message=initial_msg)

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data = line[6:].strip()
                        if data == "[DONE]":
                            break

                        try:
                            chunk = json.loads(data)
                        except json.JSONDecodeError:
                            continue

                        choices = chunk.get("choices", [])
                        if not choices:
                            continue

                        choice = choices[0]
                        delta = choice.get("delta", {})
                        finish = choice.get("finish_reason") or ""

                        # 处理 usage
                        chunk_usage = chunk.get("usage")
                        if chunk_usage:
                            usage_data = Usage(
                                input=chunk_usage.get("prompt_tokens", 0),
                                output=chunk_usage.get("completion_tokens", 0),
                                cache_read=chunk_usage.get(
                                    "prompt_cache_hit_tokens", 0
                                ),
                                cache_write=chunk_usage.get(
                                    "prompt_cache_miss_tokens", 0
                                ),
                                total_tokens=chunk_usage.get("total_tokens", 0),
                            )

                        # 处理文本内容
                        text_content = delta.get("content", "")
                        if text_content:
                            if current_text_idx is None:
                                content_blocks.append(TextContent(text=""))
                                current_text_idx = len(content_blocks) - 1
                            existing = content_blocks[current_text_idx]
                            assert isinstance(existing, TextContent)
                            content_blocks[current_text_idx] = TextContent(
                                text=existing.text + str(text_content),
                            )
                            yield AssistantTextDelta(delta=str(text_content))

                        # 处理工具调用
                        tool_calls = delta.get("tool_calls", [])
                        for tc in tool_calls:
                            if isinstance(tc, dict):
                                tc_index = tc.get("index", 0)
                                tc_id = tc.get("id", "")
                                tc_function = tc.get("function", {})

                                if tc_index not in current_tool_call_idx:
                                    # 新的工具调用
                                    content_blocks.append(
                                        ToolCallContent(
                                            tool_call_id=tc_id or "",
                                            name=tc_function.get("name", "")
                                            if isinstance(tc_function, dict)
                                            else "",
                                            args={},
                                        )
                                    )
                                    current_tool_call_idx[tc_index] = (
                                        len(content_blocks) - 1
                                    )
                                    tool_call_args_raw[tc_index] = ""
                                    yield AssistantToolCallStart(
                                        tool_call_id=tc_id or "",
                                        name=tc_function.get("name", "")
                                        if isinstance(tc_function, dict)
                                        else "",
                                    )

                                block_idx = current_tool_call_idx[tc_index]
                                existing_block = content_blocks[block_idx]
                                if isinstance(existing_block, ToolCallContent):
                                    if tc_id:
                                        existing_block.tool_call_id = tc_id
                                    if isinstance(tc_function, dict):
                                        name = tc_function.get("name", "")
                                        if name:
                                            existing_block.name = name
                                        args_delta = tc_function.get("arguments", "")
                                        if args_delta:
                                            # 累积原始参数字符串，完整后再解析
                                            tool_call_args_raw[tc_index] += str(
                                                args_delta
                                            )
                                            parsed = _parse_tool_call_args(
                                                tool_call_args_raw[tc_index]
                                            )
                                            if parsed is not None:
                                                existing_block.args = parsed
                                    yield AssistantToolCallUpdate(
                                        tool_call_id=existing_block.tool_call_id,
                                        args=existing_block.args,
                                    )

                        if finish:
                            finish_reason = finish
                            # 工具调用结束
                            for tc_index in current_tool_call_idx.values():
                                block = content_blocks[tc_index]
                                if isinstance(block, ToolCallContent):
                                    yield AssistantToolCallEnd(
                                        tool_call_id=block.tool_call_id,
                                        content=[block],
                                    )

                    # 流结束
                    stop_reason = _map_finish_reason(finish_reason)
                    final_msg = _build_snapshot(
                        model,
                        content_blocks,
                        stop_reason,
                        usage_data,
                        model_id=request_model_id,
                    )
                    yield AssistantMessageSnapshot(message=final_msg)
                    yield AssistantStreamEnd()

        except asyncio.CancelledError:
            abort_event = AssistantAbortedEvent(error="Request aborted by user")
        except Exception as exc:
            error_event = AssistantErrorEvent(error=str(exc))
            abort_event = AssistantAbortedEvent(error=str(exc))

        if error_event is not None:
            yield error_event
        if abort_event is not None:
            yield abort_event


def _map_finish_reason(reason: str) -> StopReason:
    """映射 OpenAI finish_reason 到内部 StopReason。"""
    mapping: dict[str, StopReason] = {
        "stop": "stop",
        "length": "length",
        "tool_calls": "tool_use",
        "content_filter": "error",
        "function_call": "tool_use",
    }
    return mapping.get(reason, "stop")


def _build_snapshot(
    model: Model,
    content_blocks: list[ContentBlock],
    stop_reason: StopReason,
    usage: Usage | None,
    model_id: str | None = None,
) -> AssistantMessage:
    """构建 AssistantMessage 快照。"""
    return AssistantMessage(
        role="assistant",
        content=list(content_blocks),
        api=model.api,
        provider=model.provider,
        model=model_id or model.model_id,
        usage=usage,
        stop_reason=stop_reason,
        timestamp=int(time.time() * 1000),
    )


__all__ = [
    "DeepSeekModel",
    "DeepSeekStreamFn",
    "create_deepseek_stream_fn",
]