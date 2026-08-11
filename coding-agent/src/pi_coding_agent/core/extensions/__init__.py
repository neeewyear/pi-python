"""扩展系统（对应 TS ``core/extensions/index.ts``）。

重新导出所有扩展类型、加载器、运行器和包装器。
"""

from __future__ import annotations

from .loader import (
    clear_extension_cache,
    create_extension_runtime,
    discover_and_load_extensions,
    load_extension_from_factory,
    load_extensions,
    load_extensions_cached,
)
from .runner import (
    ExtensionRunner,
    emit_project_trust_event,
    emit_session_shutdown_event,
)
from .types import *
from .wrapper import (
    wrap_registered_tool,
    wrap_registered_tools,
)

__all__ = [
    # Loader
    "clear_extension_cache",
    "create_extension_runtime",
    "discover_and_load_extensions",
    "load_extension_from_factory",
    "load_extensions",
    "load_extensions_cached",
    # Runner
    "emit_project_trust_event",
    "emit_session_shutdown_event",
    "ExtensionRunner",
    # Wrapper
    "wrap_registered_tool",
    "wrap_registered_tools",
]
