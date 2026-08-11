"""Run modes for the coding agent."""

from __future__ import annotations

from .interactive import run_interactive_mode
from .json_event import JsonAgentSessionEvent, serialize_agent_session_event
from .print_mode import PrintModeOptions, run_print_mode
from .rpc.rpc_client import ModelInfo, RpcClient, RpcClientOptions, RpcEventListener
from .rpc.rpc_mode import run_rpc_mode
from .rpc.rpc_types import (
    RpcCommand,
    RpcExtensionUIRequest,
    RpcExtensionUIResponse,
    RpcResponse,
    RpcSessionState,
)

__all__: list[str] = [
    "JsonAgentSessionEvent",
    "ModelInfo",
    "PrintModeOptions",
    "RpcClient",
    "RpcClientOptions",
    "RpcCommand",
    "RpcEventListener",
    "RpcExtensionUIRequest",
    "RpcExtensionUIResponse",
    "RpcResponse",
    "RpcSessionState",
    "run_interactive_mode",
    "run_print_mode",
    "run_rpc_mode",
    "serialize_agent_session_event",
]