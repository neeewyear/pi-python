"""CLI 扩展模块。

提供 CLI 交互辅助功能，包括：
- 参数解析（``args``）
- 配置选择器（``config_selector``）
- 凭据打印（``credential_print``）
- 文件处理（``file_processor``）
- 初始消息构建（``initial_message``）
- 模型列表（``list_models``）
- 项目信任（``project_trust``）
- 会话选择器（``session_picker``）
- 启动 UI（``startup_ui``）
"""

from __future__ import annotations

from . import args as args
from . import config_selector as config_selector
from . import credential_print as credential_print
from . import file_processor as file_processor
from . import initial_message as initial_message
from . import list_models as list_models
from . import project_trust as project_trust
from . import session_picker as session_picker
from . import startup_ui as startup_ui
from .args import (
    VALID_THINKING_LEVELS as VALID_THINKING_LEVELS,
)

# ---------------------------------------------------------------------------
# args
# ---------------------------------------------------------------------------
from .args import (
    Args as Args,
)
from .args import (
    Diagnostic as Diagnostic,
)
from .args import (
    Mode as Mode,
)
from .args import (
    is_valid_thinking_level as is_valid_thinking_level,
)
from .args import (
    parse_args as parse_args,
)
from .args import (
    print_help as print_help,
)

# ---------------------------------------------------------------------------
# config_selector
# ---------------------------------------------------------------------------
from .config_selector import show_config_selector as show_config_selector
from .credential_print import (
    DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS as DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS,
)

# ---------------------------------------------------------------------------
# credential_print
# ---------------------------------------------------------------------------
from .credential_print import (
    CredentialPrintCommand as CredentialPrintCommand,
)
from .credential_print import (
    CredentialPrintError as CredentialPrintError,
)
from .credential_print import (
    CredentialPrintKind as CredentialPrintKind,
)
from .credential_print import (
    is_credential_print_help as is_credential_print_help,
)
from .credential_print import (
    parse_credential_print_command as parse_credential_print_command,
)
from .credential_print import (
    print_credential_print_help as print_credential_print_help,
)
from .credential_print import (
    resolve_credential_for_print as resolve_credential_for_print,
)
from .credential_print import (
    validate_credential_print_args as validate_credential_print_args,
)

# ---------------------------------------------------------------------------
# file_processor
# ---------------------------------------------------------------------------
from .file_processor import (
    ProcessedFiles as ProcessedFiles,
)
from .file_processor import (
    ProcessFileOptions as ProcessFileOptions,
)
from .file_processor import (
    process_file_arguments as process_file_arguments,
)

# ---------------------------------------------------------------------------
# initial_message
# ---------------------------------------------------------------------------
from .initial_message import (
    InitialMessageInput as InitialMessageInput,
)
from .initial_message import (
    InitialMessageResult as InitialMessageResult,
)
from .initial_message import (
    build_initial_message as build_initial_message,
)

# ---------------------------------------------------------------------------
# list_models
# ---------------------------------------------------------------------------
from .list_models import (
    format_model_list as format_model_list,
)
from .list_models import (
    list_available_models as list_available_models,
)

# ---------------------------------------------------------------------------
# project_trust
# ---------------------------------------------------------------------------
from .project_trust import create_project_trust_context as create_project_trust_context

# ---------------------------------------------------------------------------
# session_picker
# ---------------------------------------------------------------------------
from .session_picker import pick_session as pick_session

# ---------------------------------------------------------------------------
# startup_ui
# ---------------------------------------------------------------------------
from .startup_ui import (
    StartupUIOptions as StartupUIOptions,
)
from .startup_ui import (
    should_run_first_time_setup as should_run_first_time_setup,
)
from .startup_ui import (
    show_first_time_setup as show_first_time_setup,
)
from .startup_ui import (
    show_startup_input as show_startup_input,
)
from .startup_ui import (
    show_startup_selector as show_startup_selector,
)
from .startup_ui import (
    show_startup_ui as show_startup_ui,
)

__all__ = [
    # 子模块
    "args",
    "config_selector",
    "credential_print",
    "file_processor",
    "initial_message",
    "list_models",
    "project_trust",
    "session_picker",
    "startup_ui",
    # args
    "Args",
    "Diagnostic",
    "Mode",
    "VALID_THINKING_LEVELS",
    "is_valid_thinking_level",
    "parse_args",
    "print_help",
    # config_selector
    "show_config_selector",
    # credential_print
    "CredentialPrintCommand",
    "CredentialPrintError",
    "CredentialPrintKind",
    "DEFAULT_BEARER_TOKEN_MIN_EXPIRY_MS",
    "is_credential_print_help",
    "parse_credential_print_command",
    "print_credential_print_help",
    "resolve_credential_for_print",
    "validate_credential_print_args",
    # file_processor
    "ProcessedFiles",
    "ProcessFileOptions",
    "process_file_arguments",
    # initial_message
    "InitialMessageInput",
    "InitialMessageResult",
    "build_initial_message",
    # list_models
    "format_model_list",
    "list_available_models",
    # project_trust
    "create_project_trust_context",
    # session_picker
    "pick_session",
    # startup_ui
    "StartupUIOptions",
    "show_first_time_setup",
    "show_startup_input",
    "show_startup_selector",
    "show_startup_ui",
    "should_run_first_time_setup",
]
