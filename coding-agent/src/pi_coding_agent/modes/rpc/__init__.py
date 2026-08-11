"""RPC mode for headless operation over JSONL stdin/stdout."""

from __future__ import annotations

from .jsonl import JsonlReader, JsonlWriter, serialize_json_line
from .rpc_client import ModelInfo, RpcClient, RpcClientOptions, RpcEventListener
from .rpc_mode import run_rpc_mode
from .rpc_types import (
    RpcCommand,
    RpcCommandType,
    RpcError,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcSessionState,
    RpcSlashCommand,
)

__all__: list[str] = [
    "JsonlReader",
    "JsonlWriter",
    "ModelInfo",
    "RpcClient",
    "RpcClientOptions",
    "RpcCommand",
    "RpcCommandType",
    "RpcError",
    "RpcEventListener",
    "RpcExtensionUIRequest",
    "RpcExtensionUIResponse",
    "RpcResponse",
    "RpcSessionState",
    "RpcSlashCommand",
    "run_rpc_mode",
    "serialize_json_line",
]