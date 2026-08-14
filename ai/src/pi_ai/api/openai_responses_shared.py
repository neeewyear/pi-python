"""OpenAI Responses API 格式转换、工具转换和流式处理逻辑。

提供 ``convert_responses_messages``、``convert_responses_tools``、``process_responses_stream``
三大核心函数，供 ``openai-responses`` 等上游模块使用。
"""

from __future__ import annotations

from collections.abc import AsyncIterable, Callable
from typing import Any, Literal, cast

from pydantic import BaseModel

# =============================================================================
# 公开配置类型
# =============================================================================


class OpenAIResponsesStreamOptions(BaseModel):
    """OpenAI Responses 流式选项。"""

    service_tier: str | None = None
    """请求的服务层（如 "auto"、"default"）。"""
    grammar_tool_input_properties: dict[str, str] | None = None
    """语法工具名称到输入属性名的映射。"""
    resolve_service_tier: Callable[[str | None, str | None], str | None] | None = None
    """服务层解析回调。"""
    apply_service_tier_pricing: Callable[[Any, str | None], None] | None = None
    """服务层定价调整回调。"""


class ConvertResponsesMessagesOptions(BaseModel):
    """消息转换选项。"""

    include_system_prompt: bool = True
    """是否包含系统提示词。"""
    grammar_tool_input_properties: dict[str, str] | None = None
    """语法工具名称到输入属性名的映射。"""
    deferred_tools: dict[str, Any] | None = None
    """延迟加载的工具字典（名称 -> Tool）。"""
    tool_options: ConvertResponsesToolsOptions | None = None
    """工具转换选项。"""


class ConvertResponsesToolsOptions(BaseModel):
    """工具转换选项。"""    

    strict: bool | None = None
    """是否强制使用严格模式。"""
    supports_strict_mode: bool = True
    """是否支持 JSON Schema 严格模式。"""
    supports_openai_grammar_tools: bool = False
    """是否支持 OpenAI 语法工具。"""
    defer_loading: bool = False
    """是否延迟加载工具定义。"""


# =============================================================================
# 流式中间事件类型
# =============================================================================


class ResponsesTextDelta(BaseModel):
    """文本增量中间事件。"""

    type: Literal["text_delta"] = "text_delta"
    content_index: int
    delta: str


class ResponsesThinkingDelta(BaseModel):
    """思考增量中间事件。"""

    type: Literal["thinking_delta"] = "thinking_delta"
    content_index: int
    delta: str


class ResponsesTextStart(BaseModel):
    """文本开始中间事件。"""

    type: Literal["text_start"] = "text_start"
    content_index: int


class ResponsesTextEnd(BaseModel):
    """文本结束中间事件。"""

    type: Literal["text_end"] = "text_end"
    content_index: int
    content: str


class ResponsesThinkingStart(BaseModel):
    """思考开始中间事件。"""

    type: Literal["thinking_start"] = "thinking_start"
    content_index: int


class ResponsesThinkingEnd(BaseModel):
    """思考结束中间事件。"""

    type: Literal["thinking_end"] = "thinking_end"
    content_index: int
    content: str


class ResponsesToolCallStart(BaseModel):
    """工具调用开始中间事件。"""

    type: Literal["toolcall_start"] = "toolcall_start"
    content_index: int


class ResponsesToolCallDelta(BaseModel):
    """工具调用增量中间事件。"""

    type: Literal["toolcall_delta"] = "toolcall_delta"
    content_index: int
    delta: str


class ResponsesToolCallEnd(BaseModel):
    """工具调用结束中间事件。"""

    type: Literal["toolcall_end"] = "toolcall_end"
    content_index: int
    tool_call: Any  # StreamingToolCall


ResponsesStreamEvent = (
    ResponsesTextDelta
    | ResponsesThinkingDelta
    | ResponsesTextStart
    | ResponsesTextEnd
    | ResponsesThinkingStart
    | ResponsesThinkingEnd
    | ResponsesToolCallStart
    | ResponsesToolCallDelta
    | ResponsesToolCallEnd
)
"""流式中间事件联合类型。"""


# =============================================================================
# 内部类型
# =============================================================================

ToolResultOutputContent = list[dict[str, object]]
"""工具结果输出内容类型。"""


class StreamingToolCall(BaseModel):
    """流式工具调用。"""

    type: Literal["toolCall"] = "toolCall"
    tool_call_id: str
    name: str
    args: dict[str, object]
    partial_json: str | None = None
    custom_input: dict[str, Any] | None = (
        None  # {property: str, jsonBuffer: GrammarToolInputJsonBuffer}
    )


ResponsesOutputSlot = (
    dict[Literal["type", "thinking"], Any]
    | dict[Literal["type", "text"], Any]
    | dict[Literal["type", "toolCall"], Any]
)
"""响应输出槽位类型。"""


# =============================================================================
# 辅助函数
# =============================================================================


def encode_text_signature_v1(id: str, phase: str | None = None) -> str:
    """编码文本签名 V1。

    将 ID 和可选的阶段信息编码为 JSON 字符串，用于多轮对话中的消息 ID 追踪。
    """
    payload: dict[str, object] = {"v": 1, "id": id}
    if phase is not None:
        payload["phase"] = phase
    import json

    return json.dumps(payload, separators=(",", ":"))


def parse_text_signature(
    signature: str | None,
) -> dict[str, str] | None:
    """解析文本签名。

    尝试解析 JSON 格式的签名，如果失败则回退到直接使用签名文本作为 ID。
    """
    if not signature:
        return None
    if signature.startswith("{"):
        try:
            import json

            parsed = json.loads(signature)
            if (
                isinstance(parsed, dict)
                and parsed.get("v") == 1
                and isinstance(parsed.get("id"), str)
            ):
                result: dict[str, str] = {"id": parsed["id"]}
                phase = parsed.get("phase")
                if phase in ("commentary", "final_answer"):
                    result["phase"] = phase
                return result
        except (json.JSONDecodeError, ValueError):
            pass
    return {"id": signature}


def convert_tool_result_output(
    model: Any,
    content: list[Any],
) -> str | list[dict[str, object]]:
    """转换工具结果输出。

    将内部工具结果内容转换为 OpenAI Responses API 格式。
    支持图片输出，但只有模型支持图片输入时才包含。
    """
    text_parts: list[str] = []
    images: list[Any] = []
    for c in content:
        if c.type == "text":
            text_parts.append(c.text)
        elif c.type == "image":
            images.append(c)

    text_result = "\n".join(text_parts)
    has_text = len(text_result) > 0

    input_types = getattr(model, "input_types", None) or getattr(model, "input", [])
    supports_image = "image" in input_types

    if not images or not supports_image:
        if has_text:
            from ..utils.sanitize_unicode import sanitize_surrogates

            return sanitize_surrogates(text_result)
        if images:
            return "(see attached image)"
        return "(no tool output)"

    output: list[dict[str, object]] = []
    if has_text:
        from ..utils.sanitize_unicode import sanitize_surrogates

        output.append({"type": "input_text", "text": sanitize_surrogates(text_result)})
    for image in images:
        output.append(
            {
                "type": "input_image",
                "detail": "auto",
                "image_url": f"data:{image.mime_type};base64,{image.data}",
            }
        )
    return output


# =============================================================================
# 消息转换
# =============================================================================


def convert_responses_messages(
    model: Any,
    context: Any,
    allowed_tool_call_providers: set[str],
    options: ConvertResponsesMessagesOptions | None = None,
) -> list[dict[str, object]]:
    """将内部 Message 格式转换为 OpenAI Responses API 的 input 格式。

    处理：
    - system/developer 角色
    - user 消息（文本和图片）
    - assistant 消息（思考、文本、工具调用）
    - tool_result 消息（含延迟工具加载）
    - 跨 provider 的 tool call ID 归一化
    """
    from ..utils.hash import short_hash
    from ..utils.sanitize_unicode import sanitize_surrogates
    from .transform_messages import transform_messages

    if options is None:
        options = ConvertResponsesMessagesOptions()

    messages: list[dict[str, object]] = []
    loaded_tool_names: set[str] = set()

    # 归一化 ID 部分：只保留字母数字、_、-
    def normalize_id_part(part: str) -> str:
        sanitized = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in part)
        normalized = sanitized[:64] if len(sanitized) > 64 else sanitized
        return normalized.rstrip("_")

    # 构建跨 provider 的 item ID
    def build_foreign_responses_item_id(item_id: str) -> str:
        normalized = f"fc_{short_hash(item_id)}"
        return normalized[:64] if len(normalized) > 64 else normalized

    # tool call ID 归一化回调
    def normalize_tool_call_id(id: str, _target_model: Any, source: Any) -> str:
        if model.provider not in allowed_tool_call_providers:
            return normalize_id_part(id)
        if "|" not in id:
            return normalize_id_part(id)
        call_id, item_id = id.split("|", 1)
        normalized_call_id = normalize_id_part(call_id)
        is_foreign = source.provider != model.provider or source.api != model.api
        normalized_item_id = (
            build_foreign_responses_item_id(item_id)
            if is_foreign
            else normalize_id_part(item_id)
        )
        # OpenAI Responses API 要求 item_id 以 "fc_" 开头
        if not normalized_item_id.startswith("fc_"):
            normalized_item_id = normalize_id_part(f"fc_{normalized_item_id}")
        return f"{normalized_call_id}|{normalized_item_id}"

    # 转换消息
    transformed_messages = transform_messages(
        context.messages, model, normalize_tool_call_id
    )

    # 处理系统提示词
    if options.include_system_prompt and context.system_prompt:
        # 推理模型默认使用 developer 角色
        reasoning = getattr(model, "reasoning", False)
        compat = getattr(model, "compat", None) or {}
        supports_developer = (
            compat.get("supports_developer_role", True)
            if isinstance(compat, dict)
            else True
        )
        role = (
            "developer" if (reasoning and supports_developer is not False) else "system"
        )
        messages.append(
            {
                "role": role,
                "content": sanitize_surrogates(context.system_prompt),
            }
        )

    msg_index = 0
    for msg in transformed_messages:
        if msg.role == "user":
            # 用户消息
            content = msg.content
            if isinstance(content, str):
                messages.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": sanitize_surrogates(content)}
                        ],
                    }
                )
            else:
                converted_content: list[dict[str, object]] = []
                for item in content:
                    if item.type == "text":
                        converted_content.append(
                            {
                                "type": "input_text",
                                "text": sanitize_surrogates(item.text),
                            }
                        )
                    elif item.type == "image":
                        converted_content.append(
                            {
                                "type": "input_image",
                                "detail": "auto",
                                "image_url": f"data:{item.mime_type};base64,{item.data}",
                            }
                        )
                if not converted_content:
                    msg_index += 1
                    continue
                messages.append(
                    {
                        "role": "user",
                        "content": converted_content,
                    }
                )

        elif msg.role == "assistant":
            # assistant 消息
            output: list[dict[str, object]] = []
            assistant_msg = msg
            is_different_model = (
                assistant_msg.model != model.model_id
                and assistant_msg.provider == model.provider
                and assistant_msg.api == model.api
            )
            text_block_index = 0

            for block in msg.content:
                if getattr(block, "type", None) == "thinking":
                    # 思考块：如果有签名，将签名作为 reasoning item 传递给 OpenAI
                    from ..types import ThinkingBlock

                    thinking_block = cast(ThinkingBlock, block)
                    if thinking_block.signature:
                        import json

                        try:
                            reasoning_item = json.loads(thinking_block.signature)
                            output.append(reasoning_item)
                        except (json.JSONDecodeError, ValueError):
                            pass
                elif block.type == "text":
                    # 文本块
                    text_block = block
                    text_signature = getattr(
                        text_block, "text_signature", None
                    ) or getattr(text_block, "textSignature", None)
                    parsed_signature = parse_text_signature(text_signature)
                    fallback_id = (
                        f"msg_pi_{msg_index}"
                        if text_block_index == 0
                        else f"msg_pi_{msg_index}_{text_block_index}"
                    )
                    text_block_index += 1
                    msg_id = parsed_signature.get("id") if parsed_signature else None
                    if not msg_id:
                        msg_id = fallback_id
                    elif len(msg_id) > 64:
                        msg_id = f"msg_{short_hash(msg_id)}"
                    text_entry: dict[str, object] = {
                        "type": "message",
                        "role": "assistant",
                        "content": [
                            {
                                "type": "output_text",
                                "text": sanitize_surrogates(text_block.text),
                                "annotations": [],
                            }
                        ],
                        "status": "completed",
                        "id": msg_id,
                    }
                    if parsed_signature and "phase" in parsed_signature:
                        text_entry["phase"] = parsed_signature["phase"]
                    output.append(text_entry)
                elif block.type == "toolCall":
                    # 工具调用块
                    tool_call = block
                    call_id, item_id_raw = (
                        tool_call.tool_call_id.split("|", 1) + [None]
                    )[:2]
                    custom_input_property = (
                        options.grammar_tool_input_properties.get(tool_call.name)
                        if options.grammar_tool_input_properties
                        else None
                    )
                    item_id: str | None = item_id_raw

                    # 跨模型消息：省略 fc_xxx 的 item_id 以避免配对验证
                    # 自定义工具调用：非 fc_xxx 的 item_id 也省略
                    if (
                        is_different_model
                        and item_id is not None
                        and item_id.startswith("fc_")
                    ) or (
                        custom_input_property is None
                        and item_id is not None
                        and not item_id.startswith("fc_")
                    ):
                        item_id = None

                    if custom_input_property is not None:
                        from .constrained_sampling import get_grammar_tool_input

                        tool_entry: dict[str, object] = {
                            "type": "custom_tool_call",
                            "id": item_id,
                            "call_id": call_id,
                            "name": tool_call.name,
                            "input": sanitize_surrogates(
                                get_grammar_tool_input(
                                    tool_call.name,
                                    tool_call.args,
                                    custom_input_property,
                                )
                            ),
                        }
                        output.append(tool_entry)
                    else:
                        import json

                        output.append(
                            {
                                "type": "function_call",
                                "id": item_id,
                                "call_id": call_id,
                                "name": tool_call.name,
                                "arguments": json.dumps(
                                    tool_call.args, separators=(",", ":")
                                ),
                            }
                        )

            if not output:
                msg_index += 1
                continue
            messages.extend(output)

        elif msg.role == "toolResult":
            # 工具结果消息
            call_id = msg.tool_call_id.split("|", 1)[0]
            tool_result_output = convert_tool_result_output(model, msg.content)

            is_grammar = (
                options.grammar_tool_input_properties is not None
                and msg.tool_name in options.grammar_tool_input_properties
            )
            if is_grammar:
                messages.append(
                    {
                        "type": "custom_tool_call_output",
                        "call_id": call_id,
                        "output": tool_result_output,
                    }
                )
            else:
                messages.append(
                    {
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": tool_result_output,
                    }
                )

            # 处理延迟工具加载
            deferred_tools: list[Any] = []
            for name in msg.added_tool_names or []:
                tool = (
                    options.deferred_tools.get(name) if options.deferred_tools else None
                )
                if not tool or name in loaded_tool_names:
                    continue
                loaded_tool_names.add(name)
                deferred_tools.append(tool)

            if deferred_tools:
                tool_names = [t.name for t in deferred_tools]
                tool_call_key = f"{msg.tool_call_id}:{','.join(tool_names)}"
                search_call_id = f"pi_tool_load_{short_hash(tool_call_key)}"
                messages.append(
                    {
                        "type": "tool_search_call",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "arguments": {
                            "query": " ".join(tool_names),
                            "limit": len(tool_names),
                        },
                    }
                )
                tool_options = (
                    options.tool_options.model_copy(update={"defer_loading": True})
                    if options.tool_options
                    else ConvertResponsesToolsOptions(defer_loading=True)
                )
                messages.append(
                    {
                        "type": "tool_search_output",
                        "call_id": search_call_id,
                        "execution": "client",
                        "status": "completed",
                        "tools": convert_responses_tools(deferred_tools, tool_options),
                    }
                )

        msg_index += 1

    return messages


# =============================================================================
# 工具转换
# =============================================================================


def convert_responses_tools(
    tools: list[Any],
    options: ConvertResponsesToolsOptions | None = None,
) -> list[dict[str, object]]:
    """将内部 Tool 格式转换为 OpenAI 工具格式。

    支持：
    - 语法约束采样（grammar constrained sampling）
    - JSON Schema 严格模式
    - 延迟加载标记
    """
    from .constrained_sampling import (
        resolve_grammar_constrained_sampling,
        resolve_json_schema_strict_sampling,
    )

    if options is None:
        options = ConvertResponsesToolsOptions()

    default_strict = False if options.strict is None else options.strict
    supports_strict_mode = options.supports_strict_mode
    supports_openai_grammar_tools = options.supports_openai_grammar_tools

    result: list[dict[str, object]] = []
    for tool in tools:
        # 尝试语法约束采样
        grammar = resolve_grammar_constrained_sampling(
            tool, supports_openai_grammar_tools
        )
        if grammar is not None:
            tool_entry: dict[str, object] = {
                "type": "custom",
                "name": tool.name,
                "description": tool.description,
                "format": {
                    "type": "grammar",
                    "syntax": grammar.format,
                    "definition": grammar.definition,
                },
            }
            if options.defer_loading:
                tool_entry["defer_loading"] = True
            result.append(tool_entry)
            continue

        # 常规 function 工具
        constrained_strict = resolve_json_schema_strict_sampling(
            tool, supports_strict_mode
        )
        function_tool: dict[str, object] = {
            "type": "function",
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        }
        if options.defer_loading:
            function_tool["defer_loading"] = True
        if supports_strict_mode:
            function_tool["strict"] = (
                constrained_strict if constrained_strict is not None else default_strict
            )
        result.append(function_tool)

    return result


# =============================================================================
# 流式处理
# =============================================================================


def _get_custom_tool_call_input(block: StreamingToolCall) -> str:
    """获取自定义工具调用的输入字符串。"""
    if block.custom_input is None:
        return ""
    property_name = block.custom_input.get("property", "")
    if not property_name:
        return ""
    value = block.args.get(property_name)
    return str(value) if isinstance(value, str) else ""


def _append_custom_tool_call_input(
    block: StreamingToolCall,
    next_input: str,
    close: bool,
) -> str | None:
    """追加自定义工具调用输入 delta。"""
    from .constrained_sampling import append_grammar_tool_input_json_delta

    custom_input = block.custom_input
    if custom_input is None:
        return None
    property_name = custom_input.get("property", "")
    json_buffer = custom_input.get("jsonBuffer")
    if json_buffer is None:
        return None
    delta = append_grammar_tool_input_json_delta(
        json_buffer, property_name, next_input, close
    )
    block.args = {property_name: next_input}
    return delta


async def process_responses_stream(
    openai_stream: AsyncIterable[Any],
    output: Any,
    stream: Any,
    model: Any,
    options: OpenAIResponsesStreamOptions | None = None,
) -> None:
    """处理 OpenAI Responses 流式事件。

    将 OpenAI 的 ``ResponseStreamEvent`` 流转换为内部 ``ResponsesStreamEvent`` 流，
    同时构建最终的 ``AssistantMessage`` 对象。

    Args:
        openai_stream: OpenAI Responses 流式事件异步迭代器。
        output: 用于累积结果的 AssistantMessage 对象（会被原地修改）。
        stream: 用于推送中间事件的事件流。
        model: 模型对象。
        options: 流式处理选项。
    """
    from ..models import calculate_cost
    from ..utils.json_parse import parse_streaming_json

    if options is None:
        options = OpenAIResponsesStreamOptions()

    saw_terminal_response_event = False
    output_slots: dict[int, Any] = {}  # output_index -> slot
    reasoning_blocks_by_id: dict[str, Any] = {}  # item_id -> ThinkingBlock

    def apply_message_phase_stop_reason(item: Any) -> None:
        """根据消息的 phase 设置 stop_reason。"""
        if (
            hasattr(item, "type")
            and item.type == "message"
            and getattr(item, "phase", None) == "final_answer"
        ):
            output.stop_reason = "stop"

    def get_slot(
        output_index: int,
        slot_type: str,
    ) -> Any | None:
        """按类型获取指定索引的槽位。"""
        slot = output_slots.get(output_index)
        if slot is None:
            return None
        return slot if slot["type"] == slot_type else None

    def push_tool_call_delta(slot: Any, delta: str | None) -> None:
        """推送工具调用增量事件。"""
        if delta is None:
            return
        stream.push(
            ResponsesToolCallDelta(
                content_index=slot["content_index"],
                delta=delta,
            )
        )

    def create_slot(output_index: int, item: Any) -> Any | None:
        """根据 output_item 创建槽位。"""
        item_type = item.type if hasattr(item, "type") else item.get("type")

        if item_type == "reasoning":
            # 思考块
            block = type(
                "ThinkingBlock", (), {"type": "thinking", "text": "", "signature": None}
            )()
            output.thinking = (output.thinking or []) + [block]
            slot = {
                "type": "thinking",
                "block": block,
                "content_index": len(output.thinking) - 1,
            }
            output_slots[output_index] = slot
            stream.push(ResponsesThinkingStart(content_index=slot["content_index"]))
            return slot

        if item_type == "message":
            # 文本消息
            apply_message_phase_stop_reason(item)
            from ..types import TextContent

            block = TextContent(text="")
            output.content.append(block)
            slot = {
                "type": "text",
                "block": block,
                "content_index": len(output.content) - 1,
            }
            output_slots[output_index] = slot
            stream.push(ResponsesTextStart(content_index=slot["content_index"]))
            return slot

        if item_type == "function_call":
            # 函数调用
            call_id = (
                item.call_id if hasattr(item, "call_id") else item.get("call_id", "")
            )
            item_id = item.id if hasattr(item, "id") else item.get("id", "")
            item_name = item.name if hasattr(item, "name") else item.get("name", "")
            item_arguments = (
                item.arguments
                if hasattr(item, "arguments")
                else item.get("arguments", "")
            )

            block = StreamingToolCall(
                tool_call_id=f"{call_id}|{item_id}",
                name=item_name,
                args={},
                partial_json=item_arguments or "",
            )
            output.content.append(block)
            slot = {
                "type": "toolCall",
                "block": block,
                "content_index": len(output.content) - 1,
            }
            output_slots[output_index] = slot
            stream.push(ResponsesToolCallStart(content_index=slot["content_index"]))
            return slot

        if item_type == "custom_tool_call":
            # 自定义工具调用（语法约束采样）
            item_name = item.name if hasattr(item, "name") else item.get("name", "")
            call_id = (
                item.call_id if hasattr(item, "call_id") else item.get("call_id", "")
            )
            item_id = item.id if hasattr(item, "id") else item.get("id", "")
            item_input = item.input if hasattr(item, "input") else item.get("input", "")

            input_property = (
                options.grammar_tool_input_properties.get(item_name, "input")
                if options.grammar_tool_input_properties
                else "input"
            )
            inp = item_input or ""

            from .constrained_sampling import GrammarToolInputJsonBuffer

            block = StreamingToolCall(
                tool_call_id=f"{call_id}|{item_id}",
                name=item_name,
                args={input_property: inp},
                custom_input={
                    "property": input_property,
                    "jsonBuffer": GrammarToolInputJsonBuffer(),
                },
            )
            output.content.append(block)
            slot = {
                "type": "toolCall",
                "block": block,
                "content_index": len(output.content) - 1,
            }
            output_slots[output_index] = slot
            stream.push(ResponsesToolCallStart(content_index=slot["content_index"]))
            return slot

        return None

    def get_or_create_slot(output_index: int, item: Any) -> Any | None:
        """获取或创建槽位。"""
        return output_slots.get(output_index) or create_slot(output_index, item)

    def backfill_reasoning_signatures(response_output: list[Any]) -> None:
        """回填思考签名（Azure OpenAI 可能省略 reasoning.encrypted_content）。"""
        import json

        for item in response_output:
            item_type = item.type if hasattr(item, "type") else item.get("type")
            if item_type != "reasoning":
                continue
            encrypted = (
                item.encrypted_content
                if hasattr(item, "encrypted_content")
                else item.get("encrypted_content")
            )
            if not encrypted:
                continue
            item_id = item.id if hasattr(item, "id") else item.get("id", "")
            block = reasoning_blocks_by_id.get(item_id)
            if block is None:
                continue
            if getattr(block, "signature", None):
                try:
                    stored = json.loads(block.signature)
                    if stored.get("encrypted_content"):
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            block.signature = json.dumps(
                {
                    **(
                        json.loads(block.signature)
                        if getattr(block, "signature", None)
                        else {}
                    ),
                    "encrypted_content": encrypted,
                },
                separators=(",", ":"),
                )

    def finalize_response(response: Any) -> None:
        """终止响应处理。"""        
        nonlocal saw_terminal_response_event
        saw_terminal_response_event = True

        # 回填思考签名
        response_output = (
            response.output
            if hasattr(response, "output")
            else response.get("output", [])
        )
        backfill_reasoning_signatures(response_output)

        response_id = response.id if hasattr(response, "id") else response.get("id")
        if response_id:
            output.response_id = response_id

        response_usage = (
            response.usage if hasattr(response, "usage") else response.get("usage")
        )
        if response_usage:
            input_details = (
                getattr(response_usage, "input_tokens_details", None)
                or response_usage.get("input_tokens_details")
                if isinstance(response_usage, dict)
                else None
            )
            cached_tokens = 0
            cache_write_tokens = 0
            if input_details:
                cached_tokens = (
                    getattr(input_details, "cached_tokens", 0)
                    if hasattr(input_details, "cached_tokens")
                    else input_details.get("cached_tokens", 0)
                )
                cache_write_tokens = (
                    getattr(input_details, "cache_write_tokens", 0)
                    if hasattr(input_details, "cache_write_tokens")
                    else input_details.get("cache_write_tokens", 0)
                )

            input_tokens = (
                getattr(response_usage, "input_tokens", 0)
                if hasattr(response_usage, "input_tokens")
                else response_usage.get("input_tokens", 0)
            )
            output_tokens = (
                getattr(response_usage, "output_tokens", 0)
                if hasattr(response_usage, "output_tokens")
                else response_usage.get("output_tokens", 0)
            )
            total_tokens = (
                getattr(response_usage, "total_tokens", 0)
                if hasattr(response_usage, "total_tokens")
                else response_usage.get("total_tokens", 0)
            )

            output_tokens_details = (
                getattr(response_usage, "output_tokens_details", None)
                or response_usage.get("output_tokens_details")
                if isinstance(response_usage, dict)
                else None
            )
            reasoning_tokens = 0
            if output_tokens_details:
                reasoning_tokens = (
                    getattr(output_tokens_details, "reasoning_tokens", 0)
                    if hasattr(output_tokens_details, "reasoning_tokens")
                    else output_tokens_details.get("reasoning_tokens", 0)
                )

            from ..types import Cost, Usage

            output.usage = Usage(
                # OpenAI 在 input_tokens 中包含了缓存 token，需要减去
                input=max(0, input_tokens - cached_tokens - cache_write_tokens),
                output=output_tokens,
                cache_read=cached_tokens,
                cache_write=cache_write_tokens,
                total_tokens=total_tokens,
                cost=Cost(),
            )
            output.usage.reasoning = reasoning_tokens

        calculate_cost(model, output.usage)

        # 应用服务层定价
        if options.apply_service_tier_pricing:
            response_service_tier = (
                getattr(response, "service_tier", None)
                if hasattr(response, "service_tier")
                else response.get("service_tier")
            )
            service_tier = (
                options.resolve_service_tier(
                    response_service_tier, options.service_tier
                )
                if options.resolve_service_tier
                else (response_service_tier or options.service_tier)
            )
            options.apply_service_tier_pricing(output.usage, service_tier)

        # 映射 stop reason
        status = (
            response.status if hasattr(response, "status") else response.get("status")
        )
        incomplete_details = (
            response.incomplete_details
            if hasattr(response, "incomplete_details")
            else response.get("incomplete_details")
        )
        incomplete_reason = None
        if incomplete_details:
            incomplete_reason = (
                incomplete_details.reason
                if hasattr(incomplete_details, "reason")
                else (
                    incomplete_details.get("reason")
                    if isinstance(incomplete_details, dict)
                    else None
                )
            )
            if not isinstance(incomplete_reason, str):
                incomplete_reason = None

        if incomplete_reason:
            output.raw_stop_reason = f"{status}.{incomplete_reason}"
        else:
            output.raw_stop_reason = status

        mapped = _map_stop_reason(status, incomplete_reason)
        output.stop_reason = mapped["stop_reason"]
        if mapped.get("error_message"):
            output.error_message = mapped["error_message"]

        # 如果内容中有 toolCall 且 stop_reason 为 "stop"，修正为 "tool_use"
        if (
            any(b.type == "toolCall" for b in output.content)
            and output.stop_reason == "stop"
        ):
            output.stop_reason = "tool_use"

    # =============================================================
    # 主事件循环
    # =============================================================
    async for event in openai_stream:
        event_type = event.type if hasattr(event, "type") else event.get("type")

        if event_type == "response.created":
            response = (
                event.response
                if hasattr(event, "response")
                else event.get("response", {})
            )
            response_id = response.id if hasattr(response, "id") else response.get("id")
            if response_id:
                output.response_id = response_id

        elif event_type == "response.output_item.added":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            item = event.item if hasattr(event, "item") else event.get("item", {})
            create_slot(output_index, item)

        elif event_type == "response.reasoning_summary_text.delta":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot["block"].text += delta
            stream.push(
                ResponsesThinkingDelta(
                    content_index=slot["content_index"],
                    delta=delta,
                )
            )

        elif event_type == "response.reasoning_summary_part.done":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot["block"].text += "\n\n"
            stream.push(
                ResponsesThinkingDelta(
                    content_index=slot["content_index"],
                    delta="\n\n",
                )
            )

        elif event_type == "response.reasoning_text.delta":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            slot = get_slot(output_index, "thinking")
            if slot is None:
                continue
            slot["block"].text += delta
            stream.push(
                ResponsesThinkingDelta(
                    content_index=slot["content_index"],
                    delta=delta,
                )
            )

        elif (
            event_type == "response.output_text.delta"
            or event_type == "response.refusal.delta"
        ):
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            slot = get_slot(output_index, "text")
            if slot is None:
                continue
            slot["block"].text += delta
            stream.push(
                ResponsesTextDelta(
                    content_index=slot["content_index"],
                    delta=delta,
                )
            )

        elif event_type == "response.function_call_arguments.delta":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            slot = get_slot(output_index, "toolCall")
            if slot is None or slot["block"].partial_json is None:
                continue
            slot["block"].partial_json += delta
            slot["block"].args = parse_streaming_json(slot["block"].partial_json)
            push_tool_call_delta(slot, delta)

        elif event_type == "response.function_call_arguments.done":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            slot = get_slot(output_index, "toolCall")
            if slot is None or slot["block"].partial_json is None:
                continue
            previous_partial = slot["block"].partial_json
            event_arguments = (
                event.arguments
                if hasattr(event, "arguments")
                else event.get("arguments", "")
            )
            slot["block"].partial_json = event_arguments
            slot["block"].args = parse_streaming_json(slot["block"].partial_json)

            if event_arguments.startswith(previous_partial):
                delta = event_arguments[len(previous_partial) :]
                if delta:
                    push_tool_call_delta(slot, delta)

        elif event_type == "response.custom_tool_call_input.delta":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            delta = event.delta if hasattr(event, "delta") else event.get("delta", "")
            slot = get_slot(output_index, "toolCall")
            if slot is None or slot["block"].custom_input is None:
                continue
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(
                    slot["block"],
                    _get_custom_tool_call_input(slot["block"]) + delta,
                    False,
                ),
            )

        elif event_type == "response.custom_tool_call_input.done":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            slot = get_slot(output_index, "toolCall")
            if slot is None or slot["block"].custom_input is None:
                continue
            event_input = (
                event.input if hasattr(event, "input") else event.get("input", "")
            )
            push_tool_call_delta(
                slot,
                _append_custom_tool_call_input(slot["block"], event_input, True),
            )

        elif event_type == "response.output_item.done":
            output_index = (
                event.output_index
                if hasattr(event, "output_index")
                else event.get("output_index", 0)
            )
            item = event.item if hasattr(event, "item") else event.get("item", {})
            item_type = item.type if hasattr(item, "type") else item.get("type")

            apply_message_phase_stop_reason(item)
            slot = get_or_create_slot(output_index, item)

            if (
                item_type == "reasoning"
                and slot is not None
                and slot["type"] == "thinking"
            ):
                # 思考块完成
                import json

                item_summary = (
                    item.summary
                    if hasattr(item, "summary")
                    else item.get("summary", [])
                )
                item_content = (
                    item.content
                    if hasattr(item, "content")
                    else item.get("content", [])
                )
                summary_text = (
                    "\n\n".join(
                        s.text for s in (item_summary or []) if hasattr(s, "text")
                    )
                    if item_summary
                    else ""
                )
                content_text = (
                    "\n\n".join(
                        c.text for c in (item_content or []) if hasattr(c, "text")
                    )
                    if item_content
                    else ""
                )
                slot["block"].text = summary_text or content_text or slot["block"].text
                slot["block"].signature = json.dumps(
                    item, default=str, separators=(",", ":")
                )
                item_id = item.id if hasattr(item, "id") else item.get("id", "")
                reasoning_blocks_by_id[item_id] = slot["block"]
                stream.push(
                    ResponsesThinkingEnd(
                        content_index=slot["content_index"],
                        content=slot["block"].text,
                    )
                )
                output_slots.pop(output_index, None)

            elif item_type == "message" and slot is not None and slot["type"] == "text":
                # 文本消息完成
                item_content = (
                    item.content
                    if hasattr(item, "content")
                    else item.get("content", [])
                )
                text_parts: list[str] = []
                for c in item_content or []:
                    c_type = c.type if hasattr(c, "type") else c.get("type")
                    if c_type == "output_text":
                        text_parts.append(
                            c.text if hasattr(c, "text") else c.get("text", "")
                        )
                    elif c_type == "refusal":
                        text_parts.append(
                            c.refusal if hasattr(c, "refusal") else c.get("refusal", "")
                        )
                slot["block"].text = "".join(text_parts)
                item_id = item.id if hasattr(item, "id") else item.get("id", "")
                item_phase = item.phase if hasattr(item, "phase") else item.get("phase")
                slot["block"].text_signature = encode_text_signature_v1(
                    item_id, item_phase
                )
                stream.push(
                    ResponsesTextEnd(
                        content_index=slot["content_index"],
                        content=slot["block"].text,
                    )
                )
                output_slots.pop(output_index, None)

            elif (
                item_type == "function_call"
                and slot is not None
                and slot["type"] == "toolCall"
                and slot["block"].partial_json is not None
            ):
                # 函数调用完成
                item_arguments = (
                    item.arguments
                    if hasattr(item, "arguments")
                    else item.get("arguments", "")
                )
                slot["block"].args = parse_streaming_json(
                    item_arguments or slot["block"].partial_json or "{}"
                )
                # 清除临时缓冲区，回放时只使用解析后的参数
                slot["block"].partial_json = None
                stream.push(
                    ResponsesToolCallEnd(
                        content_index=slot["content_index"],
                        tool_call=slot["block"],
                    )
                )
                output_slots.pop(output_index, None)

            elif (
                item_type == "custom_tool_call"
                and slot is not None
                and slot["type"] == "toolCall"
                and slot["block"].custom_input is not None
            ):
                # 自定义工具调用完成
                item_input = (
                    item.input if hasattr(item, "input") else item.get("input", "")
                )
                push_tool_call_delta(
                    slot,
                    _append_custom_tool_call_input(
                        slot["block"],
                        item_input or _get_custom_tool_call_input(slot["block"]),
                        True,
                    ),
                )
                slot["block"].custom_input = None
                stream.push(
                    ResponsesToolCallEnd(
                        content_index=slot["content_index"],
                        tool_call=slot["block"],
                    )
                )
                output_slots.pop(output_index, None)

        elif event_type in ("response.completed", "response.incomplete"):
            event_response = (
                event.response
                if hasattr(event, "response")
                else event.get("response", {})
            )
            finalize_response(event_response)

        elif event_type == "error":
            code = (
                event.code if hasattr(event, "code") else event.get("code", "unknown")
            )
            message = (
                event.message
                if hasattr(event, "message")
                else event.get("message", "Unknown error")
            )
            raise RuntimeError(f"Error Code {code}: {message}")

        elif event_type == "response.failed":
            saw_terminal_response_event = True
            event_response = (
                event.response
                if hasattr(event, "response")
                else event.get("response", {})
            )
            status = (
                event_response.status
                if hasattr(event_response, "status")
                else event_response.get("status")
            )
            output.raw_stop_reason = status
            error = (
                event_response.error
                if hasattr(event_response, "error")
                else event_response.get("error")
            )
            details = (
                event_response.incomplete_details
                if hasattr(event_response, "incomplete_details")
                else event_response.get("incomplete_details")
            )
            if error:
                error_code = (
                    error.code
                    if hasattr(error, "code")
                    else error.get("code", "unknown")
                )
                error_message = (
                    error.message
                    if hasattr(error, "message")
                    else error.get("message", "no message")
                )
                msg = f"{error_code}: {error_message}"
            elif details:
                details_reason = (
                    details.reason
                    if hasattr(details, "reason")
                    else details.get("reason", "unknown")
                )
                msg = f"incomplete: {details_reason}"
            else:
                msg = "Unknown error (no error details in response)"
            raise RuntimeError(msg)

    if not saw_terminal_response_event:
        raise RuntimeError(
            "OpenAI Responses stream ended before a terminal response event"
        )


def _map_stop_reason(
    status: str | None,
    incomplete_reason: str | None = None,
) -> dict[str, Any]:
    """映射 OpenAI Responses 状态到内部 stop reason。"""

    if not status:
        return {"stop_reason": "stop"}

    if status == "completed":
        return {"stop_reason": "stop"}

    if status == "incomplete":
        if incomplete_reason == "max_output_tokens":
            return {"stop_reason": "length"}
        error_msg = (
            f"Response incomplete: {incomplete_reason}"
            if incomplete_reason
            else "Response incomplete without a provider reason"
        )
        return {"stop_reason": "error", "error_message": error_msg}

    if status in ("failed", "cancelled"):
        return {"stop_reason": "error"}

    # in_progress 和 queued 映射到 stop
    if status in ("in_progress", "queued"):
        return {"stop_reason": "stop"}

    # 穷举检查
    _exhaustive: str = status
    raise ValueError(f"Unhandled stop reason: {_exhaustive}")
