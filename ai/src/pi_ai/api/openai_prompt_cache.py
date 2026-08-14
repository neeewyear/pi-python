"""OpenAI Prompt Cache Key 工具"""

from __future__ import annotations

OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH = 64


def clamp_openai_prompt_cache_key(key: str | None) -> str | None:
    """限制 OpenAI Prompt Cache Key 长度。"""
    if key is None:
        return None
    if len(key) <= OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH:
        return key
    return key[:OPENAI_PROMPT_CACHE_KEY_MAX_LENGTH]