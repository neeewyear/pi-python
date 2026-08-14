"""API 格式化器包。

各 LLM API 的消息格式转换与流式传输实现。
"""

from .anthropic_messages import (
    AnthropicOptions,
)
from .anthropic_messages import (
    stream as anthropic_stream,
)
from .anthropic_messages import (
    stream_simple as anthropic_stream_simple,
)
from .azure_openai_responses import (
    AzureOpenAIResponsesOptions,
)
from .azure_openai_responses import (
    stream as azure_stream,
)
from .azure_openai_responses import (
    stream_simple as azure_stream_simple,
)
from .bedrock_converse_stream import (
    BedrockOptions,
)
from .bedrock_converse_stream import (
    stream as bedrock_stream,
)
from .bedrock_converse_stream import (
    stream_simple as bedrock_stream_simple,
)
from .constrained_sampling import create_grammar_tool_input_properties
from .github_copilot_headers import (
    build_copilot_dynamic_headers,
    has_copilot_vision_input,
)
from .google_generative_ai import GoogleOptions
from .google_generative_ai import stream as google_stream
from .google_generative_ai import stream_simple as google_stream_simple
from .google_vertex import GoogleVertexOptions
from .google_vertex import stream as vertex_stream
from .google_vertex import stream_simple as vertex_stream_simple
from .lazy import lazy_api, lazy_stream
from .openai_codex_responses import OpenAICodexResponsesOptions
from .openai_codex_responses import stream as codex_stream
from .openai_codex_responses import stream_simple as codex_stream_simple
from .openai_completions import OpenAICompletionsOptions
from .openai_completions import stream as completions_stream
from .openai_completions import stream_simple as completions_stream_simple
from .openai_prompt_cache import clamp_openai_prompt_cache_key
from .openai_responses import OpenAIResponsesOptions, stream, stream_simple
from .openai_responses_shared import (
    ConvertResponsesMessagesOptions,
    ConvertResponsesToolsOptions,
    OpenAIResponsesStreamOptions,
    convert_responses_messages,
    convert_responses_tools,
    process_responses_stream,
)
from .pi_messages import PiMessagesOptions
from .pi_messages import stream as pi_messages_stream
from .pi_messages import stream_simple as pi_messages_stream_simple
from .simple_options import build_base_options
from .transform_messages import transform_messages

__all__ = [
    "AnthropicOptions",
    "AzureOpenAIResponsesOptions",
    "BedrockOptions",
    "ConvertResponsesMessagesOptions",
    "ConvertResponsesToolsOptions",
    "GoogleOptions",
    "GoogleVertexOptions",
    "OpenAICodexResponsesOptions",
    "OpenAICompletionsOptions",
    "OpenAIResponsesOptions",
    "OpenAIResponsesStreamOptions",
    "PiMessagesOptions",
    "anthropic_stream",
    "anthropic_stream_simple",
    "azure_stream",
    "azure_stream_simple",
    "bedrock_stream",
    "bedrock_stream_simple",
    "build_base_options",
    "build_copilot_dynamic_headers",
    "clamp_openai_prompt_cache_key",
    "codex_stream",
    "codex_stream_simple",
    "completions_stream",
    "completions_stream_simple",
    "convert_responses_messages",
    "convert_responses_tools",
    "create_grammar_tool_input_properties",
    "google_stream",
    "google_stream_simple",
    "has_copilot_vision_input",
    "lazy_api",
    "lazy_stream",
    "pi_messages_stream",
    "pi_messages_stream_simple",
    "process_responses_stream",
    "stream",
    "stream_simple",
    "transform_messages",
    "vertex_stream",
    "vertex_stream_simple",
]
