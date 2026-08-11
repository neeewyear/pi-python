from __future__ import annotations

import json
import platform
from pathlib import Path

from ..config import get_agent_dir

KEYBINDINGS: dict[str, object] = {
    "app.interrupt": {"defaultKeys": "escape", "description": "Cancel or abort"},
    "app.clear": {"defaultKeys": "ctrl+c", "description": "Clear editor"},
    "app.exit": {"defaultKeys": "ctrl+d", "description": "Exit when editor is empty"},
    "app.suspend": {
        "defaultKeys": [] if platform.system() == "Windows" else "ctrl+z",
        "description": "Suspend to background",
    },
    "app.thinking.cycle": {
        "defaultKeys": "shift+tab",
        "description": "Cycle thinking level",
    },
    "app.model.cycleForward": {
        "defaultKeys": "ctrl+p",
        "description": "Cycle to next model",
    },
    "app.model.cycleBackward": {
        "defaultKeys": "shift+ctrl+p",
        "description": "Cycle to previous model",
    },
    "app.model.select": {"defaultKeys": "ctrl+l", "description": "Open model selector"},
    "app.tools.expand": {"defaultKeys": "ctrl+o", "description": "Toggle tool output"},
    "app.thinking.toggle": {
        "defaultKeys": "ctrl+t",
        "description": "Toggle thinking blocks",
    },
    "app.session.toggleNamedFilter": {
        "defaultKeys": "ctrl+n",
        "description": "Toggle named session filter",
    },
    "app.editor.external": {
        "defaultKeys": "ctrl+g",
        "description": "Open external editor",
    },
    "app.message.copy": {
        "defaultKeys": "ctrl+x",
        "description": "Copy message to clipboard",
    },
    "app.message.followUp": {
        "defaultKeys": "alt+enter",
        "description": "Queue follow-up message",
    },
    "app.message.dequeue": {
        "defaultKeys": "alt+up",
        "description": "Restore queued messages",
    },
    "app.clipboard.pasteImage": {
        "defaultKeys": "alt+v" if platform.system() == "Windows" else "ctrl+v",
        "description": "Paste image from clipboard (text fallback)",
    },
    "app.session.new": {"defaultKeys": [], "description": "Start a new session"},
    "app.session.tree": {"defaultKeys": [], "description": "Open session tree"},
    "app.session.fork": {"defaultKeys": [], "description": "Fork current session"},
    "app.session.resume": {"defaultKeys": [], "description": "Resume a session"},
    "app.tree.foldOrUp": {
        "defaultKeys": ["alt+left", "ctrl+left"]
        if platform.system() == "darwin"
        else ["ctrl+left", "alt+left"],
        "description": "Fold tree branch or move up",
    },
    "app.tree.unfoldOrDown": {
        "defaultKeys": ["alt+right", "ctrl+right"]
        if platform.system() == "darwin"
        else ["ctrl+right", "alt+right"],
        "description": "Unfold tree branch or move down",
    },
    "app.tree.editLabel": {"defaultKeys": "shift+l", "description": "Edit tree label"},
    "app.tree.toggleLabelTimestamp": {
        "defaultKeys": "shift+t",
        "description": "Toggle tree label timestamps",
    },
    "app.session.togglePath": {
        "defaultKeys": "ctrl+p",
        "description": "Toggle session path display",
    },
    "app.session.toggleSort": {
        "defaultKeys": "ctrl+s",
        "description": "Toggle session sort mode",
    },
    "app.session.rename": {"defaultKeys": "ctrl+r", "description": "Rename session"},
    "app.session.delete": {"defaultKeys": "ctrl+d", "description": "Delete session"},
    "app.session.deleteNoninvasive": {
        "defaultKeys": "ctrl+backspace",
        "description": "Delete session when query is empty",
    },
    "app.models.save": {"defaultKeys": "ctrl+s", "description": "Save model selection"},
    "app.models.enableAll": {
        "defaultKeys": "ctrl+a",
        "description": "Enable all models",
    },
    "app.models.clearAll": {"defaultKeys": "ctrl+x", "description": "Clear all models"},
    "app.models.toggleProvider": {
        "defaultKeys": "ctrl+p",
        "description": "Toggle all models for provider",
    },
    "app.models.reorderUp": {
        "defaultKeys": "alt+up",
        "description": "Move model up in order",
    },
    "app.models.reorderDown": {
        "defaultKeys": "alt+down",
        "description": "Move model down in order",
    },
    "app.tree.filter.default": {
        "defaultKeys": "ctrl+d",
        "description": "Tree filter: default view",
    },
    "app.tree.filter.noTools": {
        "defaultKeys": "ctrl+t",
        "description": "Tree filter: hide tool results",
    },
    "app.tree.filter.userOnly": {
        "defaultKeys": "ctrl+u",
        "description": "Tree filter: user messages only",
    },
    "app.tree.filter.labeledOnly": {
        "defaultKeys": "ctrl+l",
        "description": "Tree filter: labeled entries only",
    },
    "app.tree.filter.all": {
        "defaultKeys": "ctrl+a",
        "description": "Tree filter: show all entries",
    },
    "app.tree.filter.cycleForward": {
        "defaultKeys": "ctrl+o",
        "description": "Tree filter: cycle forward",
    },
    "app.tree.filter.cycleBackward": {
        "defaultKeys": "shift+ctrl+o",
        "description": "Tree filter: cycle backward",
    },
}

KEYBINDING_NAME_MIGRATIONS: dict[str, str] = {
    "cursorUp": "tui.editor.cursorUp",
    "cursorDown": "tui.editor.cursorDown",
    "cursorLeft": "tui.editor.cursorLeft",
    "cursorRight": "tui.editor.cursorRight",
    "cursorWordLeft": "tui.editor.cursorWordLeft",
    "cursorWordRight": "tui.editor.cursorWordRight",
    "cursorLineStart": "tui.editor.cursorLineStart",
    "cursorLineEnd": "tui.editor.cursorLineEnd",
    "jumpForward": "tui.editor.jumpForward",
    "jumpBackward": "tui.editor.jumpBackward",
    "pageUp": "tui.editor.pageUp",
    "pageDown": "tui.editor.pageDown",
    "deleteCharBackward": "tui.editor.deleteCharBackward",
    "deleteCharForward": "tui.editor.deleteCharForward",
    "deleteWordBackward": "tui.editor.deleteWordBackward",
    "deleteWordForward": "tui.editor.deleteWordForward",
    "deleteToLineStart": "tui.editor.deleteToLineStart",
    "deleteToLineEnd": "tui.editor.deleteToLineEnd",
    "yank": "tui.editor.yank",
    "yankPop": "tui.editor.yankPop",
    "undo": "tui.editor.undo",
    "newLine": "tui.input.newLine",
    "submit": "tui.input.submit",
    "tab": "tui.input.tab",
    "copy": "tui.input.copy",
    "selectUp": "tui.select.up",
    "selectDown": "tui.select.down",
    "selectPageUp": "tui.select.pageUp",
    "selectPageDown": "tui.select.pageDown",
    "selectConfirm": "tui.select.confirm",
    "selectCancel": "tui.select.cancel",
    "interrupt": "app.interrupt",
    "clear": "app.clear",
    "exit": "app.exit",
    "suspend": "app.suspend",
    "cycleThinkingLevel": "app.thinking.cycle",
    "cycleModelForward": "app.model.cycleForward",
    "cycleModelBackward": "app.model.cycleBackward",
    "selectModel": "app.model.select",
    "expandTools": "app.tools.expand",
    "toggleThinking": "app.thinking.toggle",
    "toggleSessionNamedFilter": "app.session.toggleNamedFilter",
    "externalEditor": "app.editor.external",
    "followUp": "app.message.followUp",
    "dequeue": "app.message.dequeue",
    "pasteImage": "app.clipboard.pasteImage",
    "newSession": "app.session.new",
    "tree": "app.session.tree",
    "fork": "app.session.fork",
    "resume": "app.session.resume",
    "treeFoldOrUp": "app.tree.foldOrUp",
    "treeUnfoldOrDown": "app.tree.unfoldOrDown",
    "treeEditLabel": "app.tree.editLabel",
    "treeToggleLabelTimestamp": "app.tree.toggleLabelTimestamp",
    "toggleSessionPath": "app.session.togglePath",
    "toggleSessionSort": "app.session.toggleSort",
    "renameSession": "app.session.rename",
    "deleteSession": "app.session.delete",
    "deleteSessionNoninvasive": "app.session.deleteNoninvasive",
}


def is_legacy_keybinding_name(key: str) -> bool:
    return key in KEYBINDING_NAME_MIGRATIONS


def to_keybindings_config(value: dict[str, object]) -> dict[str, object]:
    config: dict[str, object] = {}
    for key, binding in value.items():
        if (
            isinstance(binding, str)
            or isinstance(binding, list)
            and all(isinstance(e, str) for e in binding)
        ):
            config[key] = binding
    return config


def migrate_keybindings_config(
    raw_config: dict[str, object],
) -> tuple[dict[str, object], bool]:
    config: dict[str, object] = {}
    migrated = False

    for key, value in raw_config.items():
        next_key = KEYBINDING_NAME_MIGRATIONS.get(key, key)
        if next_key != key:
            migrated = True
        if key != next_key and next_key in raw_config:
            migrated = True
            continue
        config[next_key] = value

    return order_keybindings_config(config), migrated


def order_keybindings_config(config: dict[str, object]) -> dict[str, object]:
    ordered: dict[str, object] = {}
    for keybinding in KEYBINDINGS:
        if keybinding in config:
            ordered[keybinding] = config[keybinding]

    extras = sorted(k for k in config if k not in ordered)
    for key in extras:
        ordered[key] = config[key]

    return ordered


def load_raw_config(path: str) -> dict[str, object] | None:
    p = Path(path)
    if not p.exists():
        return None
    try:
        parsed = json.loads(p.read_text("utf-8"))
        if not isinstance(parsed, dict):
            return None
        return parsed
    except (json.JSONDecodeError, OSError):
        return None


class KeybindingsManager:
    def __init__(
        self,
        user_bindings: dict[str, object] | None = None,
        config_path: str | None = None,
    ) -> None:
        self._config_path = config_path
        self._user_bindings = user_bindings or {}
        self._all_bindings = dict(KEYBINDINGS)
        if user_bindings:
            self._all_bindings.update(user_bindings)

    @classmethod
    def create(cls, agent_dir: str | Path | None = None) -> KeybindingsManager:
        resolved_agent_dir = agent_dir or get_agent_dir()
        config_path = str(Path(resolved_agent_dir) / "keybindings.json")
        user_bindings = KeybindingsManager._load_from_file(config_path)
        return cls(user_bindings, config_path)

    def reload(self) -> None:
        if not self._config_path:
            return
        self._user_bindings = KeybindingsManager._load_from_file(self._config_path)
        self._all_bindings = dict(KEYBINDINGS)
        if self._user_bindings:
            self._all_bindings.update(self._user_bindings)

    def get_effective_config(self) -> dict[str, object]:
        return dict(self._all_bindings)

    @staticmethod
    def _load_from_file(path: str) -> dict[str, object]:
        raw_config = load_raw_config(path)
        if not raw_config:
            return {}
        migrated_config, _ = migrate_keybindings_config(raw_config)
        return to_keybindings_config(migrated_config)
