"""Edit tool - precise file editing with exact text replacement.

Uses aiofiles for file operations. Implements fuzzy matching and
unified patch generation.
"""

from __future__ import annotations

from pi_agent.types import (
    AgentTool,
    AgentToolResult,
    AgentToolUpdateCallback,
    CancellationToken,
    TextContent,
)
from pydantic import BaseModel, ConfigDict

from .edit_diff import (
    Edit,
    apply_edits_to_normalized_content,
    detect_line_ending,
    generate_diff_string,
    generate_unified_patch,
    normalize_to_lf,
    restore_line_endings,
    strip_bom,
)
from .file_mutation_queue import with_file_mutation_queue
from .path_utils import resolve_path
from .tool_definition_wrapper import ToolDefinition, wrap_tool_definition

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


class ReplaceEdit(BaseModel):
    """A single replacement edit."""

    old_text: str
    new_text: str


class EditToolInput(BaseModel):
    """Edit tool input parameters."""

    path: str
    edits: list[ReplaceEdit]


class EditToolDetails(BaseModel):
    """Edit tool output details.""" 

    diff: str
    patch: str
    first_changed_line: int | None = None


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class EditOperations:
    """Pluggable operations for the edit tool.

    Override these to delegate file editing to remote systems.
    """

    async def read_file(self, absolute_path: str) -> bytes:
        """Read file contents as bytes."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="rb") as f:
            return await f.read()

    async def write_file(self, absolute_path: str, content: str) -> None:
        """Write content to a file."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="w", encoding="utf-8") as f:
            await f.write(content)

    async def access(self, absolute_path: str) -> None:
        """Check if file is readable and writable (throw if not)."""
        import aiofiles

        async with aiofiles.open(absolute_path, mode="rb") as f:
            await f.read(1)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


class EditToolOptions(BaseModel):
    """Edit tool options."""    

    model_config = ConfigDict(arbitrary_types_allowed=True)

    operations: EditOperations | None = None


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _prepare_edit_arguments(input_data: dict[str, object]) -> dict[str, object]:
    """Prepare edit arguments, handling legacy format and JSON string edits.


    """
    args = dict(input_data)

    edits_raw = args.get("edits")
    if isinstance(edits_raw, str):
        import json

        try:
            parsed = json.loads(edits_raw)
            if isinstance(parsed, list):
                args["edits"] = parsed
        except (json.JSONDecodeError, ValueError):
            pass

    old_text = args.get("oldText")
    new_text = args.get("newText")
    if isinstance(old_text, str) and isinstance(new_text, str):
        existing_edits = args.get("edits")
        edits: list[dict[str, object]] = (
            list(existing_edits) if isinstance(existing_edits, list) else []
        )
        edits.append({"oldText": old_text, "newText": new_text})
        args.pop("oldText", None)
        args.pop("newText", None)
        args["edits"] = edits

    return args


def _validate_edit_input(input_data: dict[str, object]) -> tuple[str, list[Edit]]:
    """Validate and extract edit parameters.


    """
    edits_raw = input_data.get("edits")
    if not isinstance(edits_raw, list) or len(edits_raw) == 0:
        raise ValueError(
            "Edit tool input is invalid. edits must contain at least one replacement."
        )

    edits = [
        Edit(
            old_text=str(e.get("oldText", e.get("old_text", ""))),
            new_text=str(e.get("newText", e.get("new_text", ""))),
        )
        for e in edits_raw
        if isinstance(e, dict)
    ]
    return str(input_data["path"]), edits


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def create_edit_tool_definition(
    cwd: str,
    options: EditToolOptions | None = None,
) -> ToolDefinition[EditToolDetails | None]:
    """Create an edit tool definition.


    """
    ops = (
        options.operations
        if options is not None and options.operations is not None
        else EditOperations()
    )

    async def _execute(
        _tool_call_id: str,
        params: dict[str, object],
        signal: CancellationToken | None,
        _on_update: AgentToolUpdateCallback | None,
    ) -> AgentToolResult:
        args = _prepare_edit_arguments(params)
        path, edits = _validate_edit_input(args)

        absolute_path = resolve_path(path, cwd)

        async def _edit() -> AgentToolResult:
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            # Check if file exists
            try:
                await ops.access(absolute_path)
            except OSError as e:
                raise RuntimeError(f"Could not edit file: {path}. {e}") from e

            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            # Read the file
            buffer = await ops.read_file(absolute_path)
            raw_content = buffer.decode("utf-8")
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            # Strip BOM before matching
            stripped = strip_bom(raw_content)
            original_ending = detect_line_ending(stripped["text"])
            normalized_content = normalize_to_lf(stripped["text"])
            applied = apply_edits_to_normalized_content(normalized_content, edits, path)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            final_content = stripped["bom"] + restore_line_endings(
                applied.new_content, original_ending
            )
            await ops.write_file(absolute_path, final_content)
            if signal is not None and signal.aborted:
                raise RuntimeError("Operation aborted")

            diff_result = generate_diff_string(
                applied.base_content, applied.new_content
            )
            return AgentToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=f"Successfully replaced {len(edits)} block(s) in {path}.",
                    )
                ],
                details=EditToolDetails(
                    diff=str(diff_result["diff"]),
                    patch=generate_unified_patch(
                        path, applied.base_content, applied.new_content
                    ),
                    first_changed_line=(
                        int(diff_result["firstChangedLine"])  # type: ignore[call-overload]
                        if diff_result.get("firstChangedLine") is not None
                        else None
                    ),
                ),
            )

        return await with_file_mutation_queue(absolute_path, _edit)

    return ToolDefinition(
        name="edit",
        label="edit",
        description=(
            "Edit a single file using exact text replacement. Every "
            "edits[].oldText must match a unique, non-overlapping region of "
            "the original file. If two changes affect the same block or "
            "nearby lines, merge them into one edit instead of emitting "
            "overlapping edits."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file to edit (relative or absolute)",
                },
                "edits": {
                    "type": "array",
                    "description": "One or more targeted replacements. Each edit is matched against the original file.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "oldText": {
                                "type": "string",
                                "description": "Exact text to replace",
                            },
                            "newText": {
                                "type": "string",
                                "description": "Replacement text",
                            },
                        },
                        "required": ["oldText", "newText"],
                    },
                },
            },
            "required": ["path", "edits"],
        },
        execute=_execute,
        prepare_arguments=_prepare_edit_arguments,
    )


def create_edit_tool(
    cwd: str,
    options: EditToolOptions | None = None,
) -> AgentTool:
    """Create an edit tool (AgentTool).


    """
    return wrap_tool_definition(create_edit_tool_definition(cwd, options))
