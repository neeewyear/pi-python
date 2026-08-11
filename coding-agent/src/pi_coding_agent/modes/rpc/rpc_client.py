"""RPC Client for programmatic access to the coding agent.

Spawns the agent in RPC mode and provides a typed API for all operations.
"""

from __future__ import annotations

import asyncio
import json
import os
import signal
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeAlias

from .jsonl import serialize_json_line
from .rpc_types import RpcError

# ============================================================================
# Types
# ============================================================================


@dataclass
class RpcClientOptions:
    """Options for the RPC client."""

    cli_path: str | None = None
    """Path to the CLI entry point (default: searches for main entry)."""
    cwd: str | None = None
    """Working directory for the agent."""
    env: dict[str, str] | None = None
    """Environment variables."""
    provider: str | None = None
    """Provider to use."""
    model: str | None = None
    """Model ID to use."""
    args: list[str] | None = None
    """Additional CLI arguments."""


@dataclass
class ModelInfo:
    """Model information."""

    provider: str = ""
    id: str = ""
    context_window: int = 0
    reasoning: bool = False


RpcEventListener: TypeAlias = Callable[..., Any]
"""Event listener type for agent session events."""


# ============================================================================
# RPC Client
# ============================================================================


class RpcClient:
    """RPC client for programmatic access to the coding agent."""

    def __init__(self, options: RpcClientOptions | None = None) -> None:
        self._options = options or RpcClientOptions()
        self._process: asyncio.subprocess.Process | None = None
        self._event_listeners: list[RpcEventListener] = []
        self._pending_requests: dict[str, dict[str, Any]] = {}
        self._request_id = 0
        self._stderr = ""
        self._exit_error: Exception | None = None
        self._reader_task: asyncio.Task[None] | None = None
        self._stdin: asyncio.StreamWriter | None = None

    async def start(self) -> None:
        """Start the RPC agent process."""
        if self._process is not None:
            raise RpcError("Client already started")

        self._exit_error = None

        cli_path = self._options.cli_path or "python"
        args = ["-m", "pi_coding_agent", "--mode", "rpc"]

        if self._options.provider:
            args.extend(["--provider", self._options.provider])
        if self._options.model:
            args.extend(["--model", self._options.model])
        if self._options.args:
            args.extend(self._options.args)

        env = os.environ.copy()
        if self._options.env:
            env.update(self._options.env)

        self._process = await asyncio.create_subprocess_exec(
            cli_path,
            *args,
            cwd=self._options.cwd,
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Collect stderr for debugging
        if self._process.stderr is not None:

            async def _collect_stderr() -> None:
                assert self._process is not None
                assert self._process.stderr is not None
                while True:
                    data = await self._process.stderr.read(4096)
                    if not data:
                        break
                    self._stderr += data.decode("utf-8", errors="replace")

            stderr_task = asyncio.create_task(_collect_stderr())

        # Set up JSONL reader for stdout
        if self._process.stdout is not None:
            self._reader_task = asyncio.create_task(self._read_stdout())

        # Wait a moment for process to initialize
        await asyncio.sleep(0.1)

        if self._process.returncode is not None:
            error = self._exit_error or RpcError(
                f"Agent process exited (code={self._process.returncode}). Stderr: {self._stderr}"
            )
            self._exit_error = error
            raise error

    async def stop(self) -> None:
        """Stop the RPC agent process."""
        if self._process is None:
            return

        if self._reader_task is not None:
            self._reader_task.cancel()
            self._reader_task = None

        self._process.send_signal(signal.SIGTERM)
        try:
            await asyncio.wait_for(self._process.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            self._process.send_signal(signal.SIGKILL)
            await self._process.wait()

        self._process = None
        self._pending_requests.clear()

    def on_event(self, listener: RpcEventListener) -> Callable[[], None]:
        """Subscribe to agent events.

        Returns a function to unsubscribe.
        """
        self._event_listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._event_listeners:
                self._event_listeners.remove(listener)

        return unsubscribe

    def get_stderr(self) -> str:
        """Get collected stderr output (useful for debugging)."""
        return self._stderr

    # =========================================================================
    # Command Methods
    # =========================================================================

    async def prompt(
        self, message: str, images: list[dict[str, Any]] | None = None
    ) -> None:
        """Send a prompt to the agent."""
        cmd: dict[str, Any] = {"type": "prompt", "message": message}
        if images:
            cmd["images"] = images
        await self._send(cmd)

    async def steer(
        self, message: str, images: list[dict[str, Any]] | None = None
    ) -> None:
        """Queue a steering message to interrupt the agent mid-run."""
        cmd: dict[str, Any] = {"type": "steer", "message": message}
        if images:
            cmd["images"] = images
        await self._send(cmd)

    async def follow_up(
        self, message: str, images: list[dict[str, Any]] | None = None
    ) -> None:
        """Queue a follow-up message to be processed after the agent finishes."""
        cmd: dict[str, Any] = {"type": "follow_up", "message": message}
        if images:
            cmd["images"] = images
        await self._send(cmd)

    async def abort(self) -> None:
        """Abort current operation."""
        await self._send({"type": "abort"})

    async def new_session(self, parent_session: str | None = None) -> dict[str, bool]:
        """Start a new session, optionally with parent tracking."""
        cmd: dict[str, Any] = {"type": "new_session"}
        if parent_session is not None:
            cmd["parentSession"] = parent_session
        response = await self._send(cmd)
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_state(self) -> dict[str, Any]:
        """Get current session state."""
        response = await self._send({"type": "get_state"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def set_model(self, provider: str, model_id: str) -> dict[str, Any]:
        """Set model by provider and ID."""
        response = await self._send(
            {"type": "set_model", "provider": provider, "modelId": model_id}
        )
        return self._get_data(response)  # type: ignore[no-any-return]

    async def cycle_model(self) -> dict[str, Any] | None:
        """Cycle to next model."""
        response = await self._send({"type": "cycle_model"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_available_models(self) -> list[dict[str, Any]]:
        """Get list of available models."""
        response = await self._send({"type": "get_available_models"})
        data = self._get_data(response)
        return data.get("models", [])  # type: ignore[no-any-return]

    async def set_thinking_level(self, level: str) -> None:
        """Set thinking level."""
        await self._send({"type": "set_thinking_level", "level": level})

    async def cycle_thinking_level(self) -> dict[str, Any] | None:
        """Cycle thinking level."""
        response = await self._send({"type": "cycle_thinking_level"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_available_thinking_levels(self) -> list[str]:
        """Get list of available thinking levels for the current model."""
        response = await self._send({"type": "get_available_thinking_levels"})
        data = self._get_data(response)
        return data.get("levels", [])  # type: ignore[no-any-return]

    async def set_steering_mode(self, mode: str) -> None:
        """Set steering mode."""
        await self._send({"type": "set_steering_mode", "mode": mode})

    async def set_follow_up_mode(self, mode: str) -> None:
        """Set follow-up mode."""
        await self._send({"type": "set_follow_up_mode", "mode": mode})

    async def compact(self, custom_instructions: str | None = None) -> dict[str, Any]:
        """Compact session context."""
        cmd: dict[str, Any] = {"type": "compact"}
        if custom_instructions is not None:
            cmd["customInstructions"] = custom_instructions
        response = await self._send(cmd)
        return self._get_data(response)  # type: ignore[no-any-return]

    async def set_auto_compaction(self, enabled: bool) -> None:
        """Set auto-compaction enabled/disabled."""
        await self._send({"type": "set_auto_compaction", "enabled": enabled})

    async def set_auto_retry(self, enabled: bool) -> None:
        """Set auto-retry enabled/disabled."""
        await self._send({"type": "set_auto_retry", "enabled": enabled})

    async def abort_retry(self) -> None:
        """Abort in-progress retry."""
        await self._send({"type": "abort_retry"})

    async def bash(self, command: str) -> dict[str, Any]:
        """Execute a bash command."""
        response = await self._send({"type": "bash", "command": command})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def abort_bash(self) -> None:
        """Abort running bash command."""
        await self._send({"type": "abort_bash"})

    async def get_session_stats(self) -> dict[str, Any]:
        """Get session statistics."""
        response = await self._send({"type": "get_session_stats"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def export_html(self, output_path: str | None = None) -> dict[str, Any]:
        """Export session to HTML."""
        cmd: dict[str, Any] = {"type": "export_html"}
        if output_path is not None:
            cmd["outputPath"] = output_path
        response = await self._send(cmd)
        return self._get_data(response)  # type: ignore[no-any-return]

    async def switch_session(self, session_path: str) -> dict[str, bool]:
        """Switch to a different session file."""
        response = await self._send(
            {"type": "switch_session", "sessionPath": session_path}
        )
        return self._get_data(response)  # type: ignore[no-any-return]

    async def fork(self, entry_id: str) -> dict[str, Any]:
        """Fork from a specific message."""
        response = await self._send({"type": "fork", "entryId": entry_id})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def clone(self) -> dict[str, bool]:
        """Clone the current active branch into a new session."""
        response = await self._send({"type": "clone"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_fork_messages(self) -> list[dict[str, str]]:
        """Get messages available for forking."""
        response = await self._send({"type": "get_fork_messages"})
        data = self._get_data(response)
        return data.get("messages", [])  # type: ignore[no-any-return]

    async def get_entries(self, since: str | None = None) -> dict[str, Any]:
        """Get session entries in append order."""
        cmd: dict[str, Any] = {"type": "get_entries"}
        if since is not None:
            cmd["since"] = since
        response = await self._send(cmd)
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_tree(self) -> dict[str, Any]:
        """Get the session entry tree."""
        response = await self._send({"type": "get_tree"})
        return self._get_data(response)  # type: ignore[no-any-return]

    async def get_last_assistant_text(self) -> str | None:
        """Get text of last assistant message."""
        response = await self._send({"type": "get_last_assistant_text"})
        data = self._get_data(response)
        return data.get("text")  # type: ignore[no-any-return]

    async def set_session_name(self, name: str) -> None:
        """Set the session display name."""
        await self._send({"type": "set_session_name", "name": name})

    async def get_messages(self) -> list[dict[str, Any]]:
        """Get all messages in the session."""
        response = await self._send({"type": "get_messages"})
        data = self._get_data(response)
        return data.get("messages", [])  # type: ignore[no-any-return]

    async def get_commands(self) -> list[dict[str, Any]]:
        """Get available commands (extension commands, prompt templates, skills)."""
        response = await self._send({"type": "get_commands"})
        data = self._get_data(response)
        return data.get("commands", [])  # type: ignore[no-any-return]

    # =========================================================================
    # Helpers
    # =========================================================================

    async def wait_for_idle(self, timeout: float = 60.0) -> None:
        """Wait for agent to become idle (no streaming).

        Resolves when ``agent_settled`` event is received.
        """
        event_received = asyncio.Event()

        def on_event(event: dict[str, Any]) -> None:
            if event.get("type") == "agent_settled":
                event_received.set()

        unsubscribe = self.on_event(on_event)
        try:
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            unsubscribe()
            raise RpcError(
                f"Timeout waiting for agent to become idle. Stderr: {self._stderr}"
            )
        finally:
            unsubscribe()

    async def collect_events(self, timeout: float = 60.0) -> list[dict[str, Any]]:
        """Collect events until agent becomes idle."""
        events: list[dict[str, Any]] = []
        event_received = asyncio.Event()

        def on_event(event: dict[str, Any]) -> None:
            events.append(event)
            if event.get("type") == "agent_settled":
                event_received.set()

        unsubscribe = self.on_event(on_event)
        try:
            await asyncio.wait_for(event_received.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            unsubscribe()
            raise RpcError(f"Timeout collecting events. Stderr: {self._stderr}")
        finally:
            unsubscribe()

        return events

    async def prompt_and_wait(
        self,
        message: str,
        images: list[dict[str, Any]] | None = None,
        timeout: float = 60.0,
    ) -> list[dict[str, Any]]:
        """Send prompt and wait for completion, returning all events."""
        events_task = asyncio.create_task(self.collect_events(timeout))
        await self.prompt(message, images)
        return await events_task

    # =========================================================================
    # Internal
    # =========================================================================

    async def _read_stdout(self) -> None:
        """Read JSONL lines from stdout and dispatch them."""
        assert self._process is not None
        assert self._process.stdout is not None
        buffer = ""
        while True:
            try:
                chunk = await self._process.stdout.read(4096)
            except Exception:
                break
            if not chunk:
                if buffer:
                    self._handle_line(buffer)
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while True:
                newline_index = buffer.find("\n")
                if newline_index == -1:
                    break
                line = buffer[:newline_index]
                buffer = buffer[newline_index + 1 :]
                self._handle_line(line)

    def _handle_line(self, line: str) -> None:
        """Handle a single JSONL line from stdout."""
        line = line.removesuffix("\r")
        if not line:
            return
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return

        # Check if it's a response to a pending request
        if isinstance(data, dict):
            if data.get("type") == "response" and "id" in data:
                req_id = data["id"]
                if req_id in self._pending_requests:
                    pending = self._pending_requests.pop(req_id)
                    resolver = pending.get("resolve")
                    if resolver is not None:
                        resolver(data)
                    return

            # Otherwise it's an event
            for listener in self._event_listeners:
                listener(data)

    def _create_process_exit_error(self, code: int | None) -> RpcError:
        return RpcError(f"Agent process exited (code={code}). Stderr: {self._stderr}")

    def _reject_pending_requests(self, error: Exception) -> None:
        for req_id in list(self._pending_requests.keys()):
            pending = self._pending_requests.pop(req_id)
            rejecter = pending.get("reject")
            if rejecter is not None:
                rejecter(error)

    async def _send(self, command: dict[str, Any]) -> dict[str, Any]:
        """Send a command and wait for the response."""
        process = self._process
        if process is None or process.stdin is None:
            raise RpcError("Client not started")
        if self._exit_error is not None:
            raise self._exit_error
        if process.returncode is not None:
            error = self._create_process_exit_error(process.returncode)
            self._exit_error = error
            raise error

        self._request_id += 1
        req_id = f"req_{self._request_id}"
        full_command = {**command, "id": req_id}

        # Create a Future for the response
        loop = asyncio.get_running_loop()
        future = loop.create_future()

        timeout_handle: asyncio.TimerHandle | None = None

        def on_timeout() -> None:
            self._pending_requests.pop(req_id, None)
            if not future.done():
                future.set_exception(
                    RpcError(
                        f"Timeout waiting for response to {command.get('type')}. Stderr: {self._stderr}"
                    )
                )

        def on_response(response: dict[str, Any]) -> None:
            if timeout_handle is not None:
                timeout_handle.cancel()
            if not future.done():
                future.set_result(response)

        def on_reject(error: Exception) -> None:
            if timeout_handle is not None:
                timeout_handle.cancel()
            if not future.done():
                future.set_exception(error)

        self._pending_requests[req_id] = {
            "resolve": on_response,
            "reject": on_reject,
        }

        timeout_handle = loop.call_later(30.0, on_timeout)

        try:
            data = serialize_json_line(full_command)
            process.stdin.write(data.encode("utf-8"))
            await process.stdin.drain()
        except Exception as e:
            self._pending_requests.pop(req_id, None)
            if timeout_handle is not None:
                timeout_handle.cancel()
            pending = self._pending_requests.get(req_id)
            if pending:
                pending.get("reject", lambda _: None)(e)
            raise

        return await future  # type: ignore[no-any-return]

    @staticmethod
    def _get_data(response: dict[str, Any]) -> Any:
        """Extract data from a successful response."""
        if not response.get("success"):
            error_msg = response.get("error", "Unknown error")
            raise RpcError(error_msg)
        return response.get("data")
