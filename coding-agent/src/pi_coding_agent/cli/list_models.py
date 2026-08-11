"""模型列表（对应 TS ``cli/list-models.ts``）。

列出可用模型，支持可选的模糊搜索过滤。
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from pi_ai.auth.types import AuthOperationOptions
    from pi_ai.models import ModelRecord
    from pi_ai.types import Model

    from ..core.model_runtime import ModelRuntime


def _format_token_count(count: int) -> str:
    """格式化 token 数为人类可读形式（如 200000 -> "200K"）。

    Args:
        count: token 数量。

    Returns:
        格式化后的字符串。
    """
    if count >= 1_000_000:
        millions = count / 1_000_000
        return f"{millions!s}M" if millions % 1 == 0 else f"{millions:.1f}M"
    if count >= 1_000:
        thousands = count / 1_000
        return f"{thousands!s}K" if thousands % 1 == 0 else f"{thousands:.1f}K"
    return str(count)


def format_model_list(models: list[Model]) -> str:
    """将模型列表格式化为表格字符串。

    Args:
        models: 模型列表。

    Returns:
        格式化后的表格字符串。
    """
    from ..core.auth_guidance import format_no_models_available_message

    if not models:
        msg: str = format_no_models_available_message()
        return msg

    sorted_models = sorted(models, key=lambda m: (m.provider, m.model_id))

    rows: list[dict[str, str]] = []
    for m in sorted_models:
        rec = cast("ModelRecord", m)
        rows.append(
            {
                "provider": rec.provider,
                "model": rec.model_id,
                "context": _format_token_count(rec.context_window),
                "max_out": _format_token_count(rec.max_tokens),
                "thinking": "yes" if rec.reasoning else "no",
                "images": "yes" if "image" in rec.input_types else "no",
            }
        )

    headers = {
        "provider": "provider",
        "model": "model",
        "context": "context",
        "max_out": "max-out",
        "thinking": "thinking",
        "images": "images",
    }

    widths = {
        key: max(len(headers[key]), *(len(r[key]) for r in rows)) for key in headers
    }

    lines: list[str] = []
    lines.append("  ".join(headers[key].ljust(widths[key]) for key in headers))

    for row in rows:
        lines.append("  ".join(row[key].ljust(widths[key]) for key in headers))

    return "\n".join(lines)


async def list_available_models(
    model_runtime: ModelRuntime,
    search_pattern: str | None = None,
    signal: object = None,
) -> None:
    """列出可用模型，可选按搜索模式过滤。

    Args:
        model_runtime: 模型运行时。
        search_pattern: 可选的模糊搜索模式。
        signal: 中止信号。
    """
    load_error = model_runtime.get_error()
    if load_error:
        import warnings

        warnings.warn(f"Warning: errors loading models.json:\n{load_error}")

    opts: AuthOperationOptions = {}
    if signal is not None:
        opts["signal"] = signal

    models = list(await model_runtime.get_available(None, opts))

    if not models:
        from ..core.auth_guidance import format_no_models_available_message

        print(format_no_models_available_message())
        return

    # 应用模糊搜索过滤
    if search_pattern:
        pattern_lower = search_pattern.lower()
        filtered = [
            m for m in models if pattern_lower in f"{m.provider} {m.model_id}".lower()
        ]
    else:
        filtered = models

    if not filtered:
        print(f'No models matching "{search_pattern}"')
        return

    print(format_model_list(filtered))


__all__ = [
    "format_model_list",
    "list_available_models",
]
