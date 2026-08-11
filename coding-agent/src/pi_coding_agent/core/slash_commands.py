from __future__ import annotations

from typing import Optional, TYPE_CHECKING

from ..config import APP_NAME

if TYPE_CHECKING:
    from .source_info import SourceInfo

SlashCommandSource = str  # "extension" | "prompt" | "skill"


class SlashCommandInfo:
    def __init__(
        self,
        name: str,
        description: Optional[str] = None,
        source: Optional[SlashCommandSource] = None,
        source_info: Optional["SourceInfo"] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.source = source
        self.source_info = source_info


class BuiltinSlashCommand:
    def __init__(
        self,
        name: str,
        description: str,
        argument_hint: Optional[str] = None,
    ) -> None:
        self.name = name
        self.description = description
        self.argument_hint = argument_hint


BUILTIN_SLASH_COMMANDS: list[BuiltinSlashCommand] = [
    BuiltinSlashCommand(name="settings", description="Open settings menu"),
    BuiltinSlashCommand(name="model", description="Select model (opens selector UI)", argument_hint="<provider/model>"),
    BuiltinSlashCommand(name="scoped-models", description="Enable/disable models for Ctrl+P cycling"),
    BuiltinSlashCommand(name="export", description="Export session (HTML default, or specify path: .html/.jsonl)"),
    BuiltinSlashCommand(name="import", description="Import and resume a session from a JSONL file"),
    BuiltinSlashCommand(name="share", description="Share session as a secret GitHub gist"),
    BuiltinSlashCommand(name="copy", description="Copy last agent message to clipboard"),
    BuiltinSlashCommand(name="name", description="Set session display name"),
    BuiltinSlashCommand(name="session", description="Show session info and stats"),
    BuiltinSlashCommand(name="changelog", description="Show changelog entries"),
    BuiltinSlashCommand(name="hotkeys", description="Show all keyboard shortcuts"),
    BuiltinSlashCommand(name="fork", description="Create a new fork from a previous user message"),
    BuiltinSlashCommand(name="clone", description="Duplicate the current session at the current position"),
    BuiltinSlashCommand(name="tree", description="Navigate session tree (switch branches)"),
    BuiltinSlashCommand(name="trust", description="Save project trust decision for future sessions"),
    BuiltinSlashCommand(name="login", description="Configure provider authentication", argument_hint="<provider>"),
    BuiltinSlashCommand(name="logout", description="Remove provider authentication"),
    BuiltinSlashCommand(name="new", description="Start a new session"),
    BuiltinSlashCommand(name="compact", description="Manually compact the session context"),
    BuiltinSlashCommand(name="resume", description="Resume a different session"),
    BuiltinSlashCommand(
        name="reload",
        description="Reload keybindings, extensions, skills, prompts, themes, and context files",
    ),
    BuiltinSlashCommand(name="quit", description=f"Quit {APP_NAME}"),
]