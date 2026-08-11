"""DeepSeek API 的 stream() 实现（OpenAI 兼容接口）。

通过 HTTP SSE 直接调用 DeepSeek API，不依赖代理服务器。
使用 ``stream()`` 函数（返回 ``AssistantMessageEventStream``）替代旧的
``DeepSeekStreamFn`` 类（返回 ``AsyncIterator[AssistantMessageEvent]``）。

用法::

    from pi_ai.api.deepseek_provider import stream

    event_stream = stream(
        model,
        context,
        api_key="sk-xxx",        # 省略时依次从环境变量 / ~/.zshrc 解析
        base_url="https://api.deepseek.com",
        timeout_ms=60_000,
    )
    async for event in event_stream:
        ...
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from pathlib import Path
from typing import Any, cast

import httpx

from ..types import (
    AssistantErrorEvent,
    AssistantMessage,
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
    StopReason,
    TextContent,
    ToolCallContent,
    Usage,
)
from ..utils.event_stream import AssistantMessageEventStream

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
                            "url": f"data:{block.mime_type};base64,{block.data}",
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
# 辅助函数
# ---------------------------------------------------------------------------


def format_deepseek_error(error: object) -> str:
    """格式化 DeepSeek API 错误。"""
    if isinstance(error, httpx.HTTPStatusError):
        return f"DeepSeek API error: {error.response.status_code} - {error.response.text[:500]}"
    if isinstance(error, httpx.RequestError):
        return f"DeepSeek request error: {error}"
    return str(error)


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


# ---------------------------------------------------------------------------
# stream() - 主入口
# ---------------------------------------------------------------------------


def stream(
    model: Any,
    context: Context,
    options: Any | None = None,
    *,
    api_key: str | None = None,
    base_url: str = "https://api.deepseek.com",
    model_id: str | None = None,
    timeout_ms: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    max_retries: int | None = None,
) -> AssistantMessageEventStream:
    """DeepSeek API 流式生成函数。

    直接调用 DeepSeek 的 OpenAI 兼容 chat completions 端点。

    Args:
        model: 模型句柄（至少需要 ``model_id`` / ``api`` / ``provider`` 属性）。
        context: 请求上下文，包含 messages、system_prompt、tools。
        options: 流式选项（可选），可提供 ``timeout_ms``、``max_tokens``、
            ``temperature`` 等覆盖工厂默认值。
        api_key: DeepSeek API key。为 None 时依次从环境变量
            ``DEEPSEEK_API_KEY`` 和 ``~/.zshrc`` 解析。
        base_url: DeepSeek API 基础 URL。
        model_id: 请求使用的模型 ID；为 None 时回退到 ``model.model_id``。
        timeout_ms: 单次请求超时（毫秒），None 表示不设超时。
        max_tokens: 最大生成 token 数。
        temperature: 采样温度。
        max_retries: 请求失败时的重试次数。

    Returns:
        AssistantMessageEventStream: 事件流，可通过 ``async for`` 消费。
    """
    event_stream = AssistantMessageEventStream()

    async def _run() -> None:
        # 构建初始 output
        output = AssistantMessage(
            role="assistant",
            content=[],
            api=getattr(model, "api", ""),
            provider=getattr(model, "provider", ""),
            model=getattr(model, "model_id", ""),
            usage=Usage(
                input=0,
                output=0,
                cache_read=0,
                cache_write=0,
                total_tokens=0,
            ),
            stop_reason="pending",
            timestamp=int(time.time() * 1000),
        )

        try:
            # 解析 API key
            resolved_key = _resolve_api_key(api_key)
            if not resolved_key:
                raise RuntimeError(
                    "DEEPSEEK_API_KEY 未设置。请传入 api_key，或设置环境变量 "
                    "DEEPSEEK_API_KEY，或在 ~/.zshrc 中配置: export DEEPSEEK_API_KEY=xxx"
                )

            # 构建请求
            messages = [_convert_message(msg) for msg in context.messages]
            tools = _convert_tools(context)

            request_model_id = model_id or getattr(model, "model_id", "deepseek-chat")

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

            # 请求参数：options > 函数参数 > 默认值
            max_tokens_val = max_tokens
            temperature_val = temperature
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

            # 超时：options > 函数参数 > 不设超时
            timeout_val = timeout_ms
            if options is not None and getattr(options, "timeout_ms", None) is not None:
                timeout_val = options.timeout_ms
            timeout = httpx.Timeout(
                None if timeout_val is None else timeout_val / 1000.0
            )

            # 重试
            retries = max_retries
            transport: httpx.AsyncHTTPTransport | None = None
            if retries is not None:
                transport = httpx.AsyncHTTPTransport(retries=retries)

            headers: dict[str, str] = {
                "Authorization": f"Bearer {resolved_key}",
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
            }

            url = f"{base_url.rstrip('/')}/v1/chat/completions"

            # 累积状态
            content_blocks: list[ContentBlock] = []
            current_text_idx: int | None = None
            current_tool_call_idx: dict[int, int] = {}  # index -> content_blocks index
            tool_call_args_raw: dict[
                int, str
            ] = {}  # index -> 累积的 arguments 原始字符串
            finish_reason: str = "stop"
            usage_data: Usage = Usage()

            # 发送初始 message_snapshot
            initial_msg = _build_snapshot(
                cast(Model, model),
                content_blocks,
                "pending",
                usage_data,
                model_id=request_model_id,
            )
            output.content = list(initial_msg.content)
            output.stop_reason = initial_msg.stop_reason
            output.usage = initial_msg.usage
            output.model = initial_msg.model
            event_stream.push(AssistantMessageSnapshot(message=output))

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
                        error_msg = f"DeepSeek API error: {response.status_code} - {error_text[:500]}"
                        output.stop_reason = "error"
                        output.error_message = error_msg
                        event_stream.push(
                            AssistantErrorEvent(reason="error", error=output)
                        )
                        event_stream.end()
                        return

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
                            event_stream.push(
                                AssistantTextDelta(delta=str(text_content))
                            )

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
                                    event_stream.push(
                                        AssistantToolCallStart(
                                            tool_call_id=tc_id or "",
                                            name=tc_function.get("name", "")
                                            if isinstance(tc_function, dict)
                                            else "",
                                        )
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
                                    event_stream.push(
                                        AssistantToolCallUpdate(
                                            tool_call_id=existing_block.tool_call_id,
                                            args=existing_block.args,
                                        )
                                    )

                        if finish:
                            finish_reason = finish
                            # 工具调用结束
                            for tc_index in current_tool_call_idx.values():
                                block = content_blocks[tc_index]
                                if isinstance(block, ToolCallContent):
                                    event_stream.push(
                                        AssistantToolCallEnd(
                                            tool_call_id=block.tool_call_id,
                                            content=[block],
                                        )
                                    )

                    # 流结束
                    stop_reason = _map_finish_reason(finish_reason)
                    final_msg = _build_snapshot(
                        cast(Model, model),
                        content_blocks,
                        stop_reason,
                        usage_data,
                        model_id=request_model_id,
                    )
                    output.content = list(final_msg.content)
                    output.stop_reason = final_msg.stop_reason
                    output.usage = final_msg.usage

                    event_stream.push(
                        AssistantStreamEnd(reason=stop_reason, message=output)
                    )
                    event_stream.end()

        except asyncio.CancelledError:
            output.stop_reason = "aborted"
            output.error_message = "Request aborted by user"
            event_stream.push(AssistantErrorEvent(reason="aborted", error=output))
            event_stream.end()
        except Exception as exc:
            output.stop_reason = "error"
            output.error_message = format_deepseek_error(exc)
            event_stream.push(AssistantErrorEvent(reason="error", error=output))
            event_stream.end()

    asyncio.ensure_future(_run())
    return event_stream
