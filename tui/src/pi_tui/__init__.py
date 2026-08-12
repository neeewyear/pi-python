"""
pi_tui — Terminal User Interface library with differential rendering.

Python port of @mariozechner/pi-tui (TypeScript).
"""

from .alt_screen_search import AltScreenSearchComponent
from .autocomplete import (
    ApplyResult,
    AutocompleteItem,
    AutocompleteProvider,
    CombinedAutocompleteProvider,
    SlashCommand,
    SuggestionResult,
)
from .components import (
    AlignValue,
    Box,
    CancellableLoader,
    DefaultTextStyle,
    Editor,
    EditorOptions,
    EditorTheme,
    HStack,
    Image,
    ImageOptions,
    ImageTheme,
    Input,
    Loader,
    Markdown,
    MarkdownTheme,
    ScrollView,
    ScrollViewOptions,
    ScrollViewScrollToOptions,
    SelectItem,
    SelectList,
    SelectListTheme,
    SettingItem,
    SettingsList,
    SettingsListOptions,
    SettingsListTheme,
    Spacer,
    Stack,
    StackChild,
    StackEntry,
    StackEntryOptions,
    StackOptions,
    Text,
    TextChunk,
    TruncatedText,
    VStack,
    word_wrap_line,
)
from .editor_component import EditorComponent
from .fuzzy import fuzzy_filter, fuzzy_match
from .keybindings import (
    DEFAULT_EDITOR_KEYBINDINGS,
    EditorAction,
    EditorKeybindingsManager,
    get_editor_keybindings,
)
from .keys import (
    KEY,
    is_key_release,
    is_key_repeat,
    matches_key,
    parse_key,
    set_kitty_protocol_active,
)
from .kill_ring import KillRing
from .latex import render_latex
from .stdin_buffer import StdinBuffer
from .terminal import ProcessTerminal, Terminal
from .terminal_colors import (
    RgbColor,
    TerminalColorScheme,
    is_osc11_background_color_response,
    parse_osc11_background_color,
    parse_terminal_color_scheme_report,
)
from .terminal_image import (
    CellDimensions,
    ImageDimensions,
    ImageRenderOptions,
    TerminalCapabilities,
    get_capabilities,
    get_cell_dimensions,
    get_image_dimensions,
    image_fallback,
    is_image_line,
    render_image,
)
from .tui import CURSOR_MARKER, TUI
from .tui_alt_screen import TuiAltScreen, TuiAltScreenOptions
from .undo_stack import UndoStack
from .utils import (
    AnsiCodeTracker,
    apply_background_to_line,
    extract_ansi_code,
    extract_segments,
    is_punctuation_char,
    is_whitespace_char,
    slice_by_column,
    slice_with_width,
    truncate_to_width,
    visible_width,
    wrap_text_with_ansi,
)

__all__ = [
    # autocomplete
    "ApplyResult",
    "AutocompleteItem",
    "AutocompleteProvider",
    "CombinedAutocompleteProvider",
    "SlashCommand",
    "SuggestionResult",
    # components
    "AlignValue",
    "Box",
    "CancellableLoader",
    "DefaultTextStyle",
    "Editor",
    "EditorOptions",
    "EditorTheme",
    "HStack",
    "Image",
    "ImageOptions",
    "ImageTheme",
    "Input",
    "Loader",
    "Markdown",
    "MarkdownTheme",
    "ScrollView",
    "ScrollViewOptions",
    "ScrollViewScrollToOptions",
    "SelectItem",
    "SelectList",
    "SelectListTheme",
    "SettingItem",
    "SettingsList",
    "SettingsListOptions",
    "SettingsListTheme",
    "Spacer",
    "Stack",
    "StackChild",
    "StackEntry",
    "StackEntryOptions",
    "StackOptions",
    "Text",
    "TextChunk",
    "TruncatedText",
    "VStack",
    "word_wrap_line",
    # alt screen search
    "AltScreenSearchComponent",
    # editor component protocol
    "EditorComponent",
    # fuzzy
    "fuzzy_filter",
    "fuzzy_match",
    # keybindings
    "DEFAULT_EDITOR_KEYBINDINGS",
    "EditorAction",
    "EditorKeybindingsManager",
    "get_editor_keybindings",
    # keys
    "KEY",
    "is_key_release",
    "is_key_repeat",
    "matches_key",
    "parse_key",
    "set_kitty_protocol_active",
    # kill ring
    "KillRing",
    # latex
    "render_latex",
    # stdin buffer
    "StdinBuffer",
    # terminal
    "ProcessTerminal",
    "Terminal",
    # terminal colors
    "RgbColor",
    "TerminalColorScheme",
    "is_osc11_background_color_response",
    "parse_osc11_background_color",
    "parse_terminal_color_scheme_report",
    # terminal image
    "CellDimensions",
    "ImageDimensions",
    "ImageRenderOptions",
    "TerminalCapabilities",
    "get_capabilities",
    "get_cell_dimensions",
    "get_image_dimensions",
    "image_fallback",
    "is_image_line",
    "render_image",
    # tui
    "CURSOR_MARKER",
    "TUI",
    # tui alt screen
    "TuiAltScreen",
    "TuiAltScreenOptions",
    # undo stack
    "UndoStack",
    # utils
    "AnsiCodeTracker",
    "apply_background_to_line",
    "extract_ansi_code",
    "extract_segments",
    "is_punctuation_char",
    "is_whitespace_char",
    "slice_by_column",
    "slice_with_width",
    "truncate_to_width",
    "visible_width",
    "wrap_text_with_ansi",
]
