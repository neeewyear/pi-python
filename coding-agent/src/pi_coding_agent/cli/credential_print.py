"""凭据打印。

提供 ``pi auth`` 子命令的解析、验证和凭据解析功能。
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from pi_ai.auth.types import AuthOperationOptions
    from pi_ai.types import Model

    from ..core.model_resolver import ResolveCliModelResult
    from ..core.model_runtime import ModelRuntime
    from .args import Args

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

CredentialPrintKind = Literal["api_key", "bearer_token"]
"""凭据打印类型。"""

DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS: int = 30 * 60_000
"""Bearer token 默认最小过期时间（30 分钟）。"""


class CredentialPrintCommand:
    """凭据打印命令。"""

    def __init__(
        self,
        kind: CredentialPrintKind,
        args: list[str],
        min_expiry_ms: int | None = None,
    ) -> None:
        self.kind = kind
        self.args = args
        self.min_expiry_ms = min_expiry_ms


class CredentialPrintError(Exception):
    """凭据打印错误。"""


# ---------------------------------------------------------------------------
# 帮助信息
# ---------------------------------------------------------------------------


def is_credential_print_help(args: list[str]) -> bool:
    """检查参数是否为 ``auth`` 帮助命令。

    Args:
        args: 命令行参数列表。

    Returns:
        是否为 auth 帮助命令。
    """
    return bool(
        args
        and args[0] == "auth"
        and (len(args) < 2 or args[1] in ("help", "--help", "-h"))
    )


def print_credential_print_help() -> None:
    """打印凭据打印帮助信息。"""
    print(
        """Usage:
  pi auth print-api-key --model <model> [--provider <provider>]
  pi auth print-bearer-token --model <model> [--provider <provider>] [--min-expiry <duration>]

Prints the configured credential alone on stdout. Provider inference uses configured credentials; specify --provider to select explicitly. Bearer tokens have a 30-minute minimum expiry by default. --min-expiry accepts ms, s, m, or h (for example, 30m)."""
    )


# ---------------------------------------------------------------------------
# 解析
# ---------------------------------------------------------------------------


def parse_credential_print_command(args: list[str]) -> CredentialPrintCommand | None:
    """解析 ``pi auth`` 命令。

    Args:
        args: 命令行参数列表（不含 ``auth``）。

    Returns:
        解析后的命令，如果不是 auth 命令则返回 None。

    Raises:
        CredentialPrintError: 命令格式错误。
    """
    if not args or args[0] != "auth":
        return None

    kind: CredentialPrintKind | None = None
    if args[1] == "print-api-key":
        kind = "api_key"
    elif args[1] == "print-bearer-token":
        kind = "bearer_token"

    if not kind:
        raise CredentialPrintError(
            f'Unknown auth command "{args[1] if len(args) > 1 else ""}". '
            'Use "pi auth print-api-key" or "pi auth print-bearer-token".'
        )

    command_args: list[str] = []
    min_expiry_ms: int | None = None
    index = 2
    while index < len(args):
        if args[index] != "--min-expiry":
            command_args.append(args[index])
            index += 1
            continue

        if kind != "bearer_token":
            raise CredentialPrintError(
                "--min-expiry is only supported by print-bearer-token"
            )

        index += 1
        if index >= len(args):
            raise CredentialPrintError(
                "--min-expiry must use a duration such as 30m or 1h"
            )

        value = args[index]
        match = re.match(r"^(\d+)(ms|s|m|h)$", value, re.IGNORECASE)
        if not match:
            raise CredentialPrintError(
                "--min-expiry must use a duration such as 30m or 1h"
            )

        amount = int(match.group(1))
        unit = match.group(2).lower()
        multiplier = {"ms": 1, "s": 1000, "m": 60_000, "h": 3_600_000}
        min_expiry_ms = amount * multiplier[unit]
        index += 1

    return (
        CredentialPrintCommand(kind=kind, args=command_args)
        if min_expiry_ms is None
        else CredentialPrintCommand(
            kind=kind, args=command_args, min_expiry_ms=min_expiry_ms
        )
    )


# ---------------------------------------------------------------------------
# 验证
# ---------------------------------------------------------------------------


def validate_credential_print_args(args: Args) -> None:
    """验证凭据打印参数。

    Args:
        args: 解析后的 CLI 参数。

    Raises:
        CredentialPrintError: 参数无效。
    """
    if not args.model or not args.model.strip():
        raise CredentialPrintError("Credential printing requires --model <model>")

    if args.api_key is not None:
        raise CredentialPrintError(
            "Credential printing reads configured credentials; --api-key is not supported"
        )

    if args.messages or args.file_args or args.unknown_flags:
        raise CredentialPrintError(
            "Credential printing only accepts --provider and --model"
        )


# ---------------------------------------------------------------------------
# 凭据解析
# ---------------------------------------------------------------------------


async def resolve_credential_for_print(
    args: Args,
    model_runtime: ModelRuntime,
    kind: CredentialPrintKind,
    min_expiry_ms: int | None = None,
    signal: object = None,
) -> str:
    """解析一个请求的凭据，用于特定 provider/model 对。

    此函数会调用 ``ModelRuntime.get_auth()``，它会刷新并持久化
    剩余时间不足五分钟的 OAuth 凭据。

    Args:
        args: 解析后的 CLI 参数。
        model_runtime: 模型运行时。
        kind: 凭据类型。
        min_expiry_ms: bearer token 最小过期时间（毫秒）。
        signal: 中止信号。

    Returns:
        凭据值（API key 或 bearer token）。

    Raises:
        CredentialPrintError: 解析失败。
    """
    from ..core.model_resolver import resolve_cli_model

    validate_credential_print_args(args)

    opts: AuthOperationOptions = {}
    if signal is not None:
        opts["signal"] = signal

    credentials = await model_runtime.list_credentials(opts)
    credential_types = {c["provider_id"]: c["type"] for c in credentials}

    models: list[Model] = []
    if args.provider:
        resolved: ResolveCliModelResult = resolve_cli_model(
            cli_provider=args.provider,
            cli_model=args.model,
            model_runtime=model_runtime,
        )
        if resolved.error or not resolved.model:
            raise CredentialPrintError(
                resolved.error or "Unable to resolve the requested provider/model"
            )
        models.append(resolved.model)
    else:
        for provider in model_runtime.get_providers():
            if provider.id not in credential_types:
                continue
            resolved = resolve_cli_model(
                cli_provider=provider.id,
                cli_model=args.model,
                model_runtime=model_runtime,
            )
            if resolved.model and not resolved.error:
                if resolved.warning and "Using custom model id" in resolved.warning:
                    continue
                models.append(resolved.model)

        if not models:
            raise CredentialPrintError(
                f'Model "{args.model}" not found. Use --list-models to see available models.'
            )

    collected: list[dict[str, str]] = []
    for model in models:
        cred_type = credential_types.get(model.provider)
        if kind == "api_key" and cred_type == "oauth":
            continue
        if kind == "bearer_token" and cred_type != "oauth":
            continue

        auth_overrides: dict[str, object] = {}
        if kind == "bearer_token":
            auth_overrides["min_oauth_validity_ms"] = (
                min_expiry_ms or DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS
            )
        if signal is not None:
            auth_overrides["signal"] = signal

        from ..core.model_runtime import ModelRuntimeAuthOverrides

        auth = await model_runtime.get_auth(
            model,
            ModelRuntimeAuthOverrides(**cast("dict[str, Any]", auth_overrides)),
        )

        if not auth:
            continue

        auth_data = auth.get("auth")
        if not auth_data:
            continue

        authorization = None
        headers = auth_data.get("headers")
        if headers:
            for _name, _value in headers.items():
                if _name.lower() == "authorization":
                    authorization = _value
                    break

        bearer_token: str | None = None
        if isinstance(authorization, str):
            match = re.match(r"^Bearer\s+(.+)$", authorization, re.IGNORECASE)
            if match:
                bearer_token = match.group(1)

        credential_value: str | None = None
        if kind == "bearer_token":
            credential_value = auth_data.get("api_key") or bearer_token
        else:
            credential_value = auth_data.get("api_key")

        if credential_value:
            collected.append({"provider_id": model.provider, "value": credential_value})

    if len(collected) == 1:
        return collected[0]["value"]

    if not collected:
        provider_id = models[0].provider if models else None
        cred_type = credential_types.get(provider_id) if provider_id else None
        if args.provider and kind == "api_key" and cred_type == "oauth":
            raise CredentialPrintError(
                f'Provider "{provider_id}" is configured with OAuth, not an API key'
            )
        if args.provider and kind == "bearer_token" and cred_type != "oauth":
            raise CredentialPrintError(
                f'Provider "{provider_id}" is not configured with an OAuth bearer token'
            )
        raise CredentialPrintError(
            f"No usable {'API key' if kind == 'api_key' else 'OAuth bearer token'} is configured"
        )

    provider_ids = [c["provider_id"] for c in collected]
    raise CredentialPrintError(
        f'Model "{args.model}" has multiple configured providers ({", ".join(provider_ids)}). Specify --provider.'
    )


__all__ = [
    "DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS",
    "CredentialPrintCommand",
    "CredentialPrintError",
    "CredentialPrintKind",
    "is_credential_print_help",
    "parse_credential_print_command",
    "print_credential_print_help",
    "resolve_credential_for_print",
    "validate_credential_print_args",
]
