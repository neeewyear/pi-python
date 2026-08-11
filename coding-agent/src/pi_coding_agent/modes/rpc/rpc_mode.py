"""RPC mode: Headless operation with JSON stdin/stdout protocol.

Used for embedding the agent in other applications.
Receives commands as JSON on stdin, outputs events and responses as JSON on stdout.

Protocol:
- Commands: JSON objects with ``type`` field, optional ``id`` for correlation
- Responses: JSON objects with ``type: "response"``, ``command``, ``success``, and optional ``data``/``error``
- Events: AgentSessionEvent objects streamed as they occur
- Extension UI: Extension UI requests are emitted, client responds with ``extension_ui_response``
"""

from __future__ import annotations

import asyncio
import json
import signal
import sys
import uuid
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from pi_coding_agent.core.agent_session import ExtensionBindings, PromptOptions
from pi_coding_agent.core.extensions.types import (
    TUI,
    Component,
    EditorFactory,
    ExtensionCommandContextActions,
    ExtensionUIContext,
    ExtensionUIDialogOptions,
    ExtensionWidgetOptions,
    KeybindingsManager,
    ReadonlyFooterDataProvider,
    Theme,
    WorkingIndicatorOptions,
)
from pi_coding_agent.core.output_guard import (
    flush_raw_stdout,
    take_over_stdout,
    wait_for_raw_stdout_backpressure,
    write_raw_stdout,
)

from .jsonl import serialize_json_line

if TYPE_CHECKING:
    from pi_coding_agent.core.agent_session import AgentSession
    from pi_coding_agent.core.agent_session_runtime import AgentSessionRuntime


async def run_rpc_mode(runtime_host: AgentSessionRuntime) -> None:
    """Run in RPC mode.

    Listens for JSON commands on stdin, outputs events and responses on stdout.
    """
    take_over_stdout()
    session = runtime_host.session
    unsubscribe: Callable[[], None] | None = None

    # pending extension UI requests waiting for response
    pending_extension_requests: dict[str, dict[str, Any]] = {}

    # Shutdown request flag
    shutdown_requested = False
    shutting_down = False

    def output(obj: object) -> None:
        write_raw_stdout(serialize_json_line(obj))

    def make_success(command: str, data: object | None = None) -> dict[str, Any]:
        result: dict[str, Any] = {
            "type": "response",
            "command": command,
            "success": True,
        }
        if data is not None:
            result["data"] = data
        return result

    def make_error(command: str, message: str) -> dict[str, Any]:
        return {
            "type": "response",
            "command": command,
            "success": False,
            "error": message,
        }

    # Helper for dialog methods with signal/timeout support
    def create_dialog_promise(
        opts: dict[str, Any] | None,
        default_value: Any,
        request: dict[str, Any],
        parse_response: Callable[[dict[str, Any]], Any],
    ) -> Awaitable[Any]:
        async def _dialog() -> Any:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            dialog_id = str(uuid.uuid4())
            timeout_handle: asyncio.TimerHandle | None = None

            def cleanup() -> None:
                if timeout_handle is not None:
                    timeout_handle.cancel()
                pending_extension_requests.pop(dialog_id, None)

            if opts is not None and opts.get("timeout") is not None:

                def on_timeout() -> None:
                    cleanup()
                    if not future.done():
                        future.set_result(default_value)

                timeout_handle = loop.call_later(opts["timeout"] / 1000.0, on_timeout)

            def on_response(response: dict[str, Any]) -> None:
                cleanup()
                if not future.done():
                    future.set_result(parse_response(response))

            pending_extension_requests[dialog_id] = {"resolve": on_response}
            output({"type": "extension_ui_request", "id": dialog_id, **request})

            return await future

        return _dialog()

    # Create extension UI context
    ui_context = ExtensionUIContext(
        select=cast(
            Callable[
                [str, list[str], ExtensionUIDialogOptions | None], Awaitable[str | None]
            ],
            lambda title, options_list, opts=None: create_dialog_promise(
                opts,
                None,
                {"method": "select", "title": title, "options": options_list},
                lambda r: None if r.get("cancelled") else r.get("value"),
            ),
        ),
        confirm=cast(
            Callable[[str, str, ExtensionUIDialogOptions | None], Awaitable[bool]],
            lambda title, message, opts=None: create_dialog_promise(
                opts,
                False,
                {"method": "confirm", "title": title, "message": message},
                lambda r: False if r.get("cancelled") else r.get("confirmed", False),
            ),
        ),
        input=cast(
            Callable[
                [str, str | None, ExtensionUIDialogOptions | None],
                Awaitable[str | None],
            ],
            lambda title, placeholder=None, opts=None: create_dialog_promise(
                opts,
                None,
                {"method": "input", "title": title, "placeholder": placeholder},
                lambda r: None if r.get("cancelled") else r.get("value"),
            ),
        ),
        notify=cast(
            Callable[[str, str | None], None],
            lambda message, notify_type=None: output(
                {
                    "type": "extension_ui_request",
                    "id": str(uuid.uuid4()),
                    "method": "notify",
                    "message": message,
                    "notifyType": notify_type,
                }
            ),
        ),
        on_terminal_input=lambda handler: lambda: None,
        set_status=lambda key, text: output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setStatus",
                "statusKey": key,
                "statusText": text,
            }
        ),
        set_working_message=cast(
            Callable[[str | None], None], lambda message=None: None
        ),
        set_working_visible=lambda visible: None,
        set_working_indicator=cast(
            Callable[[WorkingIndicatorOptions | None], None], lambda options=None: None
        ),
        set_hidden_thinking_label=cast(
            Callable[[str | None], None], lambda label=None: None
        ),
        set_widget=cast(
            Callable[[str, Any | None, ExtensionWidgetOptions | None], None],
            lambda key, content=None, options=None: (
                output(
                    {
                        "type": "extension_ui_request",
                        "id": str(uuid.uuid4()),
                        "method": "setWidget",
                        "widgetKey": key,
                        "widgetLines": content,
                        "widgetPlacement": (
                            options.get("placement") if options else None
                        ),
                    }
                )
                if content is None or isinstance(content, list)
                else None
            ),
        ),
        set_footer=cast(
            Callable[
                [Callable[[TUI, Theme, ReadonlyFooterDataProvider], Component] | None],
                None,
            ],
            lambda factory=None: None,
        ),
        set_header=cast(
            Callable[[Callable[[TUI, Theme], Component] | None], None],
            lambda factory=None: None,
        ),
        set_title=lambda title: output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "setTitle",
                "title": title,
            }
        ),
        custom=cast(
            Callable[
                [
                    Callable[
                        [TUI, Theme, KeybindingsManager, Callable[[Any], None]], Any
                    ],
                    dict[str, Any] | None,
                ],
                Awaitable[Any],
            ],
            lambda: None,
        ),
        paste_to_editor=lambda text: None,
        set_editor_text=lambda text: output(
            {
                "type": "extension_ui_request",
                "id": str(uuid.uuid4()),
                "method": "set_editor_text",
                "text": text,
            }
        ),
        get_editor_text=lambda: "",
        editor=cast(
            Callable[[str, str | None], Awaitable[str | None]],
            lambda title, prefill=None: create_dialog_promise(
                None,
                None,
                {"method": "editor", "title": title, "prefill": prefill},
                lambda r: None if r.get("cancelled") else r.get("value"),
            ),
        ),
        add_autocomplete_provider=lambda provider: None,
        set_editor_component=cast(
            Callable[[EditorFactory | None], None], lambda component=None: None
        ),
        get_editor_component=lambda: None,
        theme=None,
        get_all_themes=list,
        get_theme=lambda name: None,
        set_theme=lambda theme_name: {
            "success": False,
            "error": "Theme switching not supported in RPC mode",
        },
        get_tools_expanded=lambda: False,
        set_tools_expanded=lambda expanded: None,
    )

    async def _rebind_session(new_session: AgentSession | None = None) -> None:
        nonlocal session, unsubscribe
        session = runtime_host.session
        if unsubscribe is not None:
            unsubscribe()

        command_context_actions = ExtensionCommandContextActions(
            wait_for_idle=lambda: session.wait_for_idle(),
            new_session=lambda opts=None: runtime_host.new_session(opts or {}),
            fork=lambda entry_id, fork_options=None: _fork(entry_id, fork_options),
            navigate_tree=lambda target_id, options=None: _navigate_tree(
                target_id, options
            ),
            switch_session=lambda session_path, options=None: (
                runtime_host.switch_session(session_path, options)
            ),
            reload=lambda: session.reload(),
        )

        await session.bind_extensions(
            ExtensionBindings(
                ui_context=ui_context,
                mode="rpc",
                command_context_actions=command_context_actions,
                shutdown_handler=lambda: _set_shutdown_requested(),
                on_error=lambda err: output(
                    {
                        "type": "extension_error",
                        "extensionPath": err.extension_path,
                        "event": err.event,
                        "error": err.error,
                    }
                ),
            )
        )

        unsubscribe = session.subscribe(lambda event: _on_session_event(event, output))

    runtime_host.set_rebind_session(_rebind_session)

    async def _fork(
        entry_id: str, fork_options: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        fork_result = await runtime_host.fork(entry_id, fork_options)
        return {"cancelled": cast(bool, fork_result.get("cancelled", False))}

    async def _navigate_tree(
        target_id: str, options: dict[str, Any] | None = None
    ) -> dict[str, bool]:
        nav_result = await session.navigate_tree(
            target_id,
            {
                "summarize": options.get("summarize") if options else None,
                "customInstructions": options.get("customInstructions")
                if options
                else None,
                "replaceInstructions": options.get("replaceInstructions")
                if options
                else None,
                "label": options.get("label") if options else None,
            },
        )
        return {"cancelled": nav_result.get("cancelled", False)}

    def _set_shutdown_requested() -> None:
        nonlocal shutdown_requested
        shutdown_requested = True

    # Register signal handlers
    signal_cleanup_handlers: list[Callable[[], None]] = []

    def register_signal_handlers() -> None:
        signals = [signal.SIGTERM]
        if sys.platform != "win32":
            signals.append(signal.SIGHUP)

        for sig in signals:
            sig_value = sig

            def handler(signum: int, frame: object) -> None:
                asyncio.ensure_future(
                    _shutdown(129 if signum == signal.SIGHUP else 143)
                )

            signal.signal(sig_value, handler)
            signal_cleanup_handlers.append(lambda: None)

    await _rebind_session()
    register_signal_handlers()

    async def _shutdown(exit_code: int = 0) -> None:
        nonlocal shutting_down
        if shutting_down:
            sys.exit(exit_code)
        shutting_down = True
        for cleanup in signal_cleanup_handlers:
            cleanup()
        if unsubscribe is not None:
            unsubscribe()
        await runtime_host.dispose()
        sys.stdin.readline()  # drain stdin
        if exit_code != 143:  # not SIGTERM
            await flush_raw_stdout()
        sys.exit(exit_code)

    async def check_shutdown_requested() -> None:
        nonlocal shutdown_requested
        if shutdown_requested:
            await _shutdown()

    # Handle a single command
    async def handle_command(command: dict[str, Any]) -> dict[str, Any] | None:
        cmd_type = command.get("type", "")
        cmd_id = command.get("id")

        # Prompting
        if cmd_type == "prompt":
            preflight_succeeded = False

            async def _do_prompt() -> None:
                nonlocal preflight_succeeded
                try:
                    prompt_opts = PromptOptions(
                        images=command.get("images"),
                        streaming_behavior=command.get("streamingBehavior"),
                        source="rpc",
                        preflight_result=lambda succeeded: (
                            output(make_success("prompt")) if succeeded else None
                        ),
                    )
                    await session.prompt(command["message"], prompt_opts)
                except Exception as e:
                    if not preflight_succeeded:
                        output(make_error("prompt", str(e)))

            asyncio.ensure_future(_do_prompt())
            return None

        elif cmd_type == "steer":
            await session.steer(command["message"], command.get("images"))
            return make_success("steer")

        elif cmd_type == "follow_up":
            await session.follow_up(command["message"], command.get("images"))
            return make_success("follow_up")

        elif cmd_type == "abort":
            await session.abort()
            return make_success("abort")

        elif cmd_type == "new_session":
            options = (
                {"parent_session": command.get("parentSession")}
                if command.get("parentSession")
                else None
            )
            new_session_result = await runtime_host.new_session(options)
            if not new_session_result.get("cancelled"):
                await _rebind_session()
            return make_success("new_session", new_session_result)

        # State
        elif cmd_type == "get_state":
            state = {
                "model": session.model,
                "thinkingLevel": session.thinking_level,
                "isStreaming": session.is_streaming,
                "isCompacting": session.is_compacting,
                "steeringMode": session.steering_mode,
                "followUpMode": session.follow_up_mode,
                "sessionFile": session.session_file,
                "sessionId": session.session_id,
                "sessionName": session.session_name,
                "autoCompactionEnabled": session.auto_compaction_enabled,
                "messageCount": len(session.messages),
                "pendingMessageCount": session.pending_message_count,
            }
            return make_success("get_state", state)

        # Model
        elif cmd_type == "set_model":
            models = (
                session.model_runtime.get_available_snapshot()
                if hasattr(session, "model_runtime")
                else []
            )
            model = next(
                (
                    m
                    for m in models
                    if m.get("provider") == command.get("provider")
                    and m.get("id") == command.get("modelId")
                ),
                None,
            )
            if model is None:
                return make_error(
                    "set_model",
                    f"Model not found: {command.get('provider')}/{command.get('modelId')}",
                )
            await session.set_model(model)
            return make_success("set_model", model)

        elif cmd_type == "cycle_model":
            cycle_result = await session.cycle_model()
            return make_success("cycle_model", cycle_result)

        elif cmd_type == "get_available_models":
            models = (
                session.model_runtime.get_available_snapshot()
                if hasattr(session, "model_runtime")
                else []
            )
            return make_success("get_available_models", {"models": models})

        # Thinking
        elif cmd_type == "set_thinking_level":
            session.set_thinking_level(command.get("level", "off"))
            return make_success("set_thinking_level")

        elif cmd_type == "cycle_thinking_level":
            level = session.cycle_thinking_level()
            return make_success(
                "cycle_thinking_level", {"level": level} if level else None
            )

        elif cmd_type == "get_available_thinking_levels":
            levels = session.get_available_thinking_levels()
            return make_success("get_available_thinking_levels", {"levels": levels})

        # Queue modes
        elif cmd_type == "set_steering_mode":
            session.set_steering_mode(command.get("mode", "all"))
            return make_success("set_steering_mode")

        elif cmd_type == "set_follow_up_mode":
            session.set_follow_up_mode(command.get("mode", "all"))
            return make_success("set_follow_up_mode")

        # Compaction
        elif cmd_type == "compact":
            result = await session.compact(command.get("customInstructions"))
            return make_success("compact", result)

        elif cmd_type == "set_auto_compaction":
            session.set_auto_compaction_enabled(command.get("enabled", False))
            return make_success("set_auto_compaction")

        # Retry
        elif cmd_type == "set_auto_retry":
            session.set_auto_retry_enabled(command.get("enabled", False))
            return make_success("set_auto_retry")

        elif cmd_type == "abort_retry":
            session.abort_retry()
            return make_success("abort_retry")

        # Bash
        elif cmd_type == "bash":
            if session.extension_runner is not None:
                event_result = await session.extension_runner.emit_user_bash(
                    {
                        "type": "user_bash",
                        "command": command.get("command", ""),
                        "excludeFromContext": command.get("excludeFromContext", False),
                        "cwd": session.session_manager.get_cwd(),
                    }  # type: ignore[arg-type]
                )
                if (
                    event_result is not None
                    and hasattr(event_result, "result")
                    and event_result.result is not None
                ):
                    session.record_bash_result(
                        command.get("command", ""),
                        event_result.result,
                        {"excludeFromContext": command.get("excludeFromContext")},
                    )
                    return make_success("bash", event_result.result)

            bash_result = await session.execute_bash(
                command.get("command", ""),
                None,
                {"excludeFromContext": command.get("excludeFromContext", False)},
            )
            return make_success("bash", bash_result)

        elif cmd_type == "abort_bash":
            session.abort_bash()
            return make_success("abort_bash")

        # Session
        elif cmd_type == "get_session_stats":
            stats = session.get_session_stats()
            return make_success("get_session_stats", stats)

        elif cmd_type == "export_html":
            path = await session.export_to_html(command.get("outputPath"))
            return make_success("export_html", {"path": path})

        elif cmd_type == "switch_session":
            switch_result = await runtime_host.switch_session(
                command.get("sessionPath", "")
            )
            if not switch_result.get("cancelled"):
                await _rebind_session()
            return make_success("switch_session", switch_result)

        elif cmd_type == "fork":
            fork_result = await runtime_host.fork(command.get("entryId", ""))
            if not fork_result.get("cancelled"):
                await _rebind_session()
            return make_success(
                "fork",
                {
                    "text": fork_result.get("selectedText"),
                    "cancelled": fork_result.get("cancelled"),
                },
            )

        elif cmd_type == "clone":
            leaf_id = session.session_manager.get_leaf_id()
            if not leaf_id:
                return make_error(
                    "clone", "Cannot clone session: no current entry selected"
                )
            clone_result = await runtime_host.fork(leaf_id, {"position": "at"})
            if not clone_result.get("cancelled"):
                await _rebind_session()
            return make_success("clone", {"cancelled": clone_result.get("cancelled")})

        elif cmd_type == "get_fork_messages":
            messages = session.get_user_messages_for_forking()
            return make_success("get_fork_messages", {"messages": messages})

        elif cmd_type == "get_entries":
            session_manager = session.session_manager
            entries = session_manager.get_entries()
            since = command.get("since")
            if since is not None:
                since_index = next(
                    (
                        i
                        for i, e in enumerate(entries)
                        if getattr(e, "id", None) == since
                    ),
                    -1,
                )
                if since_index == -1:
                    return make_error("get_entries", f"Entry not found: {since}")
                entries = entries[since_index + 1 :]
            return make_success(
                "get_entries",
                {"entries": entries, "leafId": session_manager.get_leaf_id()},
            )

        elif cmd_type == "get_tree":
            session_manager = session.session_manager
            return make_success(
                "get_tree",
                {
                    "tree": session_manager.get_entries(),
                    "leafId": session_manager.get_leaf_id(),
                },
            )

        elif cmd_type == "get_last_assistant_text":
            text = session.get_last_assistant_text()
            return make_success("get_last_assistant_text", {"text": text})

        elif cmd_type == "set_session_name":
            name = command.get("name", "").strip()
            if not name:
                return make_error("set_session_name", "Session name cannot be empty")
            session.set_session_name(name)
            return make_success("set_session_name")

        # Messages
        elif cmd_type == "get_messages":
            return make_success("get_messages", {"messages": session.messages})

        # Commands
        elif cmd_type == "get_commands":
            commands: list[dict[str, Any]] = []

            if session.extension_runner is not None:
                for cmd in session.extension_runner.get_registered_commands():
                    commands.append(
                        {
                            "name": cmd.invocation_name,
                            "description": cmd.description,
                            "source": "extension",
                            "sourceInfo": cmd.source_info,
                        }
                    )

            for template in session.prompt_templates:
                commands.append(
                    {
                        "name": template.name,
                        "description": template.description,
                        "source": "prompt",
                        "sourceInfo": template.source_info,
                    }
                )

            if session.resource_loader is not None:
                for skill in session.resource_loader.get_skills().skills:
                    commands.append(
                        {
                            "name": f"skill:{skill.name}",
                            "description": skill.description,
                            "source": "skill",
                            "sourceInfo": skill.source_info,
                        }
                    )

            return make_success("get_commands", {"commands": commands})

        else:
            return make_error(cmd_type, f"Unknown command: {cmd_type}")

    # Handle input lines
    async def handle_input_line(line: str) -> None:
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as e:
            output(make_error("parse", f"Failed to parse command: {e}"))
            await wait_for_raw_stdout_backpressure()
            return

        if not isinstance(parsed, dict):
            output(make_error("parse", "Command must be a JSON object"))
            await wait_for_raw_stdout_backpressure()
            return

        # Handle extension UI responses
        if parsed.get("type") == "extension_ui_response":
            response_id = parsed.get("id", "")
            pending = pending_extension_requests.get(response_id)
            if pending is not None:
                pending_extension_requests.pop(response_id)
                resolver = pending.get("resolve")
                if resolver is not None:
                    resolver(parsed)
            return

        try:
            response = await handle_command(parsed)
            if response is not None:
                output(response)
                await wait_for_raw_stdout_backpressure()
            await check_shutdown_requested()
        except Exception as e:
            output(make_error(parsed.get("type", "unknown"), str(e)))
            await wait_for_raw_stdout_backpressure()

    # Read stdin line by line
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_running_loop().connect_read_pipe(lambda: protocol, sys.stdin)

    try:
        while True:
            line = await reader.readline()
            if not line:
                await _shutdown()
                break
            line_str = line.decode("utf-8", errors="replace").rstrip("\n").rstrip("\r")
            if line_str:
                await handle_input_line(line_str)
    except asyncio.CancelledError:
        pass


def _on_session_event(event: Any, output: Callable[[object], None]) -> None:
    """Handle session events by serializing to JSON and outputting."""
    from pi_coding_agent.modes.json_event import serialize_agent_session_event

    try:
        data = serialize_agent_session_event(event)
        output(data)
    except Exception:
        pass
