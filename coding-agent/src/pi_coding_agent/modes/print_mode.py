"""Print mode (single-shot): Send prompts, output result, exit.

Used for:
- ``pi -p "prompt"`` - text output
- ``pi --mode json "prompt"`` - JSON event stream
"""

from __future__ import annotations

import json
import signal
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from pi_coding_agent.core.agent_session import PromptOptions
from pi_coding_agent.core.output_guard import (
    flush_raw_stdout,
    write_raw_stdout,
)

if TYPE_CHECKING:
    from pi_agent.types import ImageContent

    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime


@dataclass
class PrintModeOptions:
    """Options for print mode."""

    mode: str = "text"
    """Output mode: "text" for final response only, "json" for all events."""
    messages: list[str] = field(default_factory=list)
    """Array of additional prompts to send after initial_message."""
    initial_message: str | None = None
    """First message to send (may contain @file content)."""
    initial_images: list[ImageContent] | None = None
    """Images to attach to the initial message."""


async def run_print_mode(
    runtime_host: AgentSessionRuntime, options: PrintModeOptions
) -> int:
    """Run in print (single-shot) mode.

    Sends prompts to the agent and outputs the result.
    """
    mode = options.mode
    messages = options.messages
    initial_message = options.initial_message
    initial_images = options.initial_images
    exit_code = 0
    session = runtime_host.session
    unsubscribe: Callable[[], None] | None = None
    disposed = False
    signal_cleanup_handlers: list[Callable[[], None]] = []

    async def dispose_runtime() -> None:
        nonlocal disposed
        if disposed:
            return
        disposed = True
        if unsubscribe is not None:
            unsubscribe()
        await runtime_host.dispose()

    def register_signal_handlers() -> None:
        signals = [signal.SIGTERM]
        if sys.platform != "win32":
            signals.append(signal.SIGHUP)

        for sig in signals:
            sig_value = sig

            def handler(signum: int, frame: object) -> None:
                import asyncio

                asyncio.ensure_future(dispose_runtime())

            signal.signal(sig_value, handler)  # type: ignore[arg-type]

    register_signal_handlers()

    async def rebind_session(new_session: AgentSession) -> None:
        nonlocal session
        session = new_session

    runtime_host.set_rebind_session(rebind_session)

    try:
        if mode == "json":
            header = session.session_manager.get_header()
            if header is not None:
                header_dict = (
                    header.model_dump()
                    if hasattr(header, "model_dump")
                    else vars(header)
                )
                write_raw_stdout(f"{json.dumps(header_dict)}\n")

        if initial_message is not None:
            prompt_opts = PromptOptions()
            if initial_images:
                prompt_opts.images = list(initial_images)
            await session.prompt(initial_message, prompt_opts)

        for message in messages:
            await session.prompt(message)

        if mode == "text":
            state = session.state
            if state is not None and state.messages:
                last_message = state.messages[-1]
                if last_message.role == "assistant":
                    assistant_msg = last_message  # type: ignore[assignment]
                    if assistant_msg.stop_reason in ("error", "aborted"):
                        error_msg = (
                            assistant_msg.error_message
                            or f"Request {assistant_msg.stop_reason}"
                        )
                        print(error_msg, file=sys.stderr)
                        exit_code = 1
                    else:
                        for content in assistant_msg.content:
                            if content.type == "text":
                                write_raw_stdout(f"{content.text}\n")

        return exit_code
    except Exception as error:
        print(str(error), file=sys.stderr)
        return 1
    finally:
        for cleanup in signal_cleanup_handlers:
            cleanup()
        await dispose_runtime()
        await flush_raw_stdout()
