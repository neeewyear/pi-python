"""CLI 参数解析与帮助信息显示（对应 TS ``cli/args.ts``）。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

from pi_ai.types import ThinkingLevel

from ..config import APP_NAME, CONFIG_DIR_NAME, ENV_AGENT_DIR, ENV_SESSION_DIR
from ..core.extensions.types import ExtensionFlag
from ..core.settings_manager import UiMode

# ---------------------------------------------------------------------------
# 类型
# ---------------------------------------------------------------------------

Mode = Literal["text", "json", "rpc"]
"""输出模式。"""


class Diagnostic:
    """CLI 诊断信息（警告或错误）。"""

    def __init__(self, type: Literal["warning", "error"], message: str) -> None:
        self.type = type
        self.message = message


class Args:
    """解析后的 CLI 参数（对应 TS ``Args``）。"""

    def __init__(self) -> None:
        self.provider: str | None = None
        self.model: str | None = None
        self.api_key: str | None = None
        self.system_prompt: str | None = None
        self.append_system_prompt: list[str] | None = None
        self.thinking: ThinkingLevel | None = None
        self.continue_: bool | None = None
        self.resume: bool | None = None
        self.help: bool | None = None
        self.version: bool | None = None
        self.mode: Mode | None = None
        self.name: str | None = None
        self.no_session: bool | None = None
        self.session: str | None = None
        self.session_id: str | None = None
        self.fork: str | None = None
        self.session_dir: str | None = None
        self.models: list[str] | None = None
        self.tools: list[str] | None = None
        self.exclude_tools: list[str] | None = None
        self.no_tools: bool | None = None
        self.no_builtin_tools: bool | None = None
        self.extensions: list[str] | None = None
        self.no_extensions: bool | None = None
        self.print: bool | None = None
        self.export: str | None = None
        self.no_skills: bool | None = None
        self.skills: list[str] | None = None
        self.prompt_templates: list[str] | None = None
        self.no_prompt_templates: bool | None = None
        self.themes: list[str] | None = None
        self.no_themes: bool | None = None
        self.no_context_files: bool | None = None
        self.list_models: str | bool | None = None
        self.offline: bool | None = None
        self.ui_mode: UiMode | None = None
        self.verbose: bool | None = None
        self.project_trust_override: bool | None = None
        self.messages: list[str] = []
        self.file_args: list[str] = []
        self.unknown_flags: dict[str, bool | str] = {}
        self.diagnostics: list[Diagnostic] = []


# ---------------------------------------------------------------------------
# 有效思考级别
# ---------------------------------------------------------------------------

VALID_THINKING_LEVELS: tuple[ThinkingLevel, ...] = (
    "off",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)


def is_valid_thinking_level(level: str) -> bool:
    """检查字符串是否为有效的思考级别。

    Args:
        level: 待检查的字符串。

    Returns:
        是否为有效的 ThinkingLevel。
    """
    return level in VALID_THINKING_LEVELS


# ---------------------------------------------------------------------------
# 解析函数
# ---------------------------------------------------------------------------


def parse_args(args: list[str]) -> Args:
    """解析 CLI 参数字符串列表。

    Args:
        args: 命令行参数字符串列表（不含 ``sys.argv[0]``）。

    Returns:
        解析后的 ``Args`` 实例。
    """
    result = Args()

    i = 0
    while i < len(args):
        arg = args[i]

        if arg in ("--help", "-h"):
            result.help = True
        elif arg in ("--version", "-v"):
            result.version = True
        elif arg == "--mode" and i + 1 < len(args):
            i += 1
            mode = args[i]
            if mode in ("text", "json", "rpc"):
                result.mode = mode  # type: ignore[assignment]
        elif arg in ("--continue", "-c"):
            result.continue_ = True
        elif arg in ("--resume", "-r"):
            result.resume = True
        elif arg == "--provider" and i + 1 < len(args):
            i += 1
            result.provider = args[i]
        elif arg == "--model" and i + 1 < len(args):
            i += 1
            result.model = args[i]
        elif arg == "--api-key" and i + 1 < len(args):
            i += 1
            result.api_key = args[i]
        elif arg == "--system-prompt" and i + 1 < len(args):
            i += 1
            result.system_prompt = args[i]
        elif arg == "--append-system-prompt" and i + 1 < len(args):
            i += 1
            if result.append_system_prompt is None:
                result.append_system_prompt = []
            result.append_system_prompt.append(args[i])
        elif arg in ("--name", "-n"):
            if i + 1 < len(args):
                i += 1
                result.name = args[i]
            else:
                result.diagnostics.append(Diagnostic("error", "--name requires a value"))
        elif arg == "--no-session":
            result.no_session = True
        elif arg == "--session" and i + 1 < len(args):
            i += 1
            result.session = args[i]
        elif arg == "--session-id" and i + 1 < len(args):
            i += 1
            result.session_id = args[i]
        elif arg == "--fork" and i + 1 < len(args):
            i += 1
            result.fork = args[i]
        elif arg == "--session-dir" and i + 1 < len(args):
            i += 1
            result.session_dir = args[i]
        elif arg == "--models" and i + 1 < len(args):
            i += 1
            result.models = [s.strip() for s in args[i].split(",")]
        elif arg in ("--no-tools", "-nt"):
            result.no_tools = True
        elif arg in ("--no-builtin-tools", "-nbt"):
            result.no_builtin_tools = True
        elif arg in ("--tools", "-t") and i + 1 < len(args):
            i += 1
            result.tools = [s.strip() for s in args[i].split(",") if s.strip()]
        elif arg in ("--exclude-tools", "-xt") and i + 1 < len(args):
            i += 1
            result.exclude_tools = [s.strip() for s in args[i].split(",") if s.strip()]
        elif arg == "--thinking" and i + 1 < len(args):
            i += 1
            level = args[i]
            if is_valid_thinking_level(level):
                result.thinking = level  # type: ignore[assignment]
            else:
                result.diagnostics.append(
                    Diagnostic(
                        "warning",
                        f'Invalid thinking level "{level}". Valid values: {", ".join(VALID_THINKING_LEVELS)}',
                    )
                )
        elif arg in ("--print", "-p"):
            result.print = True
            next_idx = i + 1
            if next_idx < len(args):
                next_arg = args[next_idx]
                if not next_arg.startswith("@") and (not next_arg.startswith("-") or next_arg.startswith("---")):
                    result.messages.append(next_arg)
                    i += 1
        elif arg == "--export" and i + 1 < len(args):
            i += 1
            result.export = args[i]
        elif arg in ("--extension", "-e") and i + 1 < len(args):
            i += 1
            if result.extensions is None:
                result.extensions = []
            result.extensions.append(args[i])
        elif arg in ("--no-extensions", "-ne"):
            result.no_extensions = True
        elif arg == "--skill" and i + 1 < len(args):
            i += 1
            if result.skills is None:
                result.skills = []
            result.skills.append(args[i])
        elif arg == "--prompt-template" and i + 1 < len(args):
            i += 1
            if result.prompt_templates is None:
                result.prompt_templates = []
            result.prompt_templates.append(args[i])
        elif arg == "--theme" and i + 1 < len(args):
            i += 1
            if result.themes is None:
                result.themes = []
            result.themes.append(args[i])
        elif arg in ("--no-skills", "-ns"):
            result.no_skills = True
        elif arg in ("--no-prompt-templates", "-np"):
            result.no_prompt_templates = True
        elif arg == "--no-themes":
            result.no_themes = True
        elif arg in ("--no-context-files", "-nc"):
            result.no_context_files = True
        elif arg == "--list-models":
            if i + 1 < len(args) and not args[i + 1].startswith("-") and not args[i + 1].startswith("@"):
                i += 1
                result.list_models = args[i]
            else:
                result.list_models = True
        elif arg == "--ui-mode":
            next_idx = i + 1
            if next_idx < len(args):
                mode = args[next_idx]
                if mode in ("regular", "fullscreen"):
                    result.ui_mode = mode  # type: ignore[assignment]
                    i += 1
                elif mode.startswith("-"):
                    result.diagnostics.append(Diagnostic("error", "--ui-mode requires regular or fullscreen"))
                else:
                    i += 1
                    result.diagnostics.append(
                        Diagnostic("error", f'Invalid UI mode "{mode}". Valid values: regular, fullscreen')
                    )
            else:
                result.diagnostics.append(Diagnostic("error", "--ui-mode requires regular or fullscreen"))
        elif arg == "--verbose":
            result.verbose = True
        elif arg in ("--approve", "-a"):
            result.project_trust_override = True
        elif arg in ("--no-approve", "-na"):
            result.project_trust_override = False
        elif arg == "--offline":
            result.offline = True
        elif arg.startswith("@"):
            result.file_args.append(arg[1:])
        elif arg.startswith("--"):
            eq_idx = arg.find("=")
            if eq_idx != -1:
                result.unknown_flags[arg[2:eq_idx]] = arg[eq_idx + 1 :]
            else:
                flag_name = arg[2:]
                next_idx = i + 1
                if next_idx < len(args) and not args[next_idx].startswith("-") and not args[next_idx].startswith("@"):
                    result.unknown_flags[flag_name] = args[next_idx]
                    i += 1
                else:
                    result.unknown_flags[flag_name] = True
        elif arg.startswith("-") and not arg.startswith("--"):
            result.diagnostics.append(Diagnostic("error", f"Unknown option: {arg}"))
        elif not arg.startswith("-"):
            result.messages.append(arg)

        i += 1

    return result


# ---------------------------------------------------------------------------
# 帮助信息
# ---------------------------------------------------------------------------


def print_help(extension_flags: list[ExtensionFlag] | None = None) -> None:
    """打印帮助信息到标准输出。

    Args:
        extension_flags: 可选的扩展注册的标志列表。
    """
    extension_flags_text = ""
    if extension_flags:
        lines: list[str] = []
        lines.append("")
        lines.append("Extension CLI Flags:")
        for flag in extension_flags:
            value = " <value>" if flag.type == "string" else ""
            description = flag.description or f"Registered by {flag.extension_path}"
            lines.append(f"  --{flag.name}{value}" + " " * max(1, 30 - len(flag.name) - len(value)) + description)
        extension_flags_text = "\n".join(lines) + "\n"

    print(
        f"""{APP_NAME} - AI coding assistant with read, bash, edit, write tools

Usage:
  {APP_NAME} [options] [@files...] [messages...]

Commands:
  {APP_NAME} install <source> [-l]     Install extension source and add to settings
  {APP_NAME} remove <source> [-l]      Remove extension source from settings
  {APP_NAME} uninstall <source> [-l]   Alias for remove
  {APP_NAME} update [source|self|pi]   Update pi, extensions, or model catalogs
  {APP_NAME} list                      List installed extensions from settings
  {APP_NAME} config [-l]               Open TUI to enable/disable package resources (Tab switches scope)
  {APP_NAME} auth <command>            Print credentials for external clients
  {APP_NAME} <command> --help          Show help for install/remove/uninstall/update/list/config/auth

Options:
  --provider <name>              Provider name (default: google)
  --model <pattern>              Model pattern or ID (supports "provider/id" and optional ":<thinking>")
  --api-key <key>                API key (defaults to env vars)
  --system-prompt <text>         System prompt (default: coding assistant prompt)
  --append-system-prompt <text>  Append text or file contents to the system prompt (can be used multiple times)
  --mode <mode>                  Output mode: text (default), json, or rpc
  --print, -p                    Non-interactive mode: process prompt and exit
  --continue, -c                 Continue previous session
  --resume, -r                   Select a session to resume
  --session <path|id>            Use specific session file or partial UUID
  --session-id <id>              Use exact project session ID, creating it if missing
  --fork <path|id>               Fork specific session file or partial UUID into a new session
  --session-dir <dir>            Directory for session storage and lookup
  --no-session                   Don't save session (ephemeral)
  --name, -n <name>              Set session display name
  --models <patterns>            Comma-separated model patterns for Ctrl+P cycling
                                 Supports globs (anthropic/*, *sonnet*) and fuzzy matching
  --no-tools, -nt                Disable all tools by default (built-in and extension)
  --no-builtin-tools, -nbt       Disable built-in tools by default but keep extension/custom tools enabled
  --tools, -t <tools>            Comma-separated allowlist of tool names to enable
                                 Applies to built-in, extension, and custom tools
  --exclude-tools, -xt <tools>   Comma-separated denylist of tool names to disable
                                 Applies to built-in, extension, and custom tools
  --thinking <level>             Set thinking level: off, minimal, low, medium, high, xhigh, max
  --extension, -e <path>         Load an extension file (can be used multiple times)
  --no-extensions, -ne           Disable extension discovery (explicit -e paths still work)
  --skill <path>                 Load a skill file or directory (can be used multiple times)
  --no-skills, -ns               Disable skills discovery and loading
  --prompt-template <path>       Load a prompt template file or directory (can be used multiple times)
  --no-prompt-templates, -np     Disable prompt template discovery and loading
  --theme <path>                 Load a theme file or directory (can be used multiple times)
  --no-themes                    Disable theme discovery and loading
  --no-context-files, -nc        Disable AGENTS.md and CLAUDE.md discovery and loading
  --export <file>                Export session file to HTML and exit
  --list-models [search]         List available models (with optional fuzzy search)
  --verbose                      Force verbose startup (overrides quietStartup setting)
  --ui-mode <mode>               UI mode: regular (default) or fullscreen
  --approve, -a                  Trust project-local files for this run
  --no-approve, -na              Ignore project-local files for this run
  --offline                      Disable startup network operations (same as PI_OFFLINE=1)
  --help, -h                     Show this help
  --version, -v                  Show version number

Extensions can register additional flags (e.g., --plan from plan-mode extension).{extension_flags_text}

Examples:
  # Print a provider API key for an external client
  {APP_NAME} auth print-api-key --provider openai --model gpt-5.5

  # Print an OAuth bearer token for an external client (refreshes if expired)
  {APP_NAME} auth print-bearer-token --provider openai-codex --model gpt-5.5

  # Interactive mode
  {APP_NAME}

  # Interactive mode with initial prompt
  {APP_NAME} "List all .ts files in src/"

  # Include files in initial message
  {APP_NAME} @prompt.md @image.png "What color is the sky?"

  # Non-interactive mode (process and exit)
  {APP_NAME} -p "List all .ts files in src/"

  # Multiple messages (interactive)
  {APP_NAME} "Read package.json" "What dependencies do we have?"

  # Continue previous session
  {APP_NAME} --continue "What did we discuss?"

  # Start a named session
  {APP_NAME} --name "Refactor auth module"

  # Use different model
  {APP_NAME} --provider openai --model gpt-4o-mini "Help me refactor this code"

  # Use model with provider prefix (no --provider needed)
  {APP_NAME} --model openai/gpt-4o "Help me refactor this code"

  # Use model with thinking level shorthand
  {APP_NAME} --model sonnet:high "Solve this complex problem"

  # Limit model cycling to specific models
  {APP_NAME} --models claude-sonnet,claude-haiku,gpt-4o

  # Limit to a specific provider with glob pattern
  {APP_NAME} --models "github-copilot/*"

  # Cycle models with fixed thinking levels
  {APP_NAME} --models sonnet:high,haiku:low

  # Start with a specific thinking level
  {APP_NAME} --thinking high "Solve this complex problem"

  # Read-only mode (no file modifications possible)
  {APP_NAME} --tools read,grep,find,ls -p "Review the code in src/"

  # Disable one tool while keeping the rest available
  {APP_NAME} --exclude-tools ask_question

  # Export a session file to HTML
  {APP_NAME} --export ~/{CONFIG_DIR_NAME}/agent/sessions/--path--/session.jsonl
  {APP_NAME} --export session.jsonl output.html

Environment Variables:
  ANTHROPIC_AUTH_TOKEN             - Anthropic bearer auth token
  ANTHROPIC_API_KEY                - Anthropic Claude API key
  ANTHROPIC_OAUTH_TOKEN            - Anthropic OAuth token (alternative to API key)
  ANT_LING_API_KEY                 - Ant Ling API key
  OPENAI_API_KEY                   - OpenAI GPT API key
  AZURE_OPENAI_API_KEY             - Azure OpenAI API key
  AZURE_OPENAI_BASE_URL            - Azure OpenAI/Cognitive Services base URL (e.g. https://{{resource}}.openai.azure.com)
  AZURE_OPENAI_RESOURCE_NAME       - Azure OpenAI resource name (alternative to base URL)
  AZURE_OPENAI_API_VERSION         - Azure OpenAI API version (default: v1)
  AZURE_OPENAI_DEPLOYMENT_NAME_MAP - Azure OpenAI model=deployment map (comma-separated)
  DEEPSEEK_API_KEY                 - DeepSeek API key
  NVIDIA_API_KEY                   - NVIDIA NIM API key
  GEMINI_API_KEY                   - Google Gemini API key
  GROQ_API_KEY                     - Groq API key
  CEREBRAS_API_KEY                 - Cerebras API key
  XAI_API_KEY                      - xAI Grok API key
  FIREWORKS_API_KEY                - Fireworks API key
  TOGETHER_API_KEY                 - Together AI API key
  BASETEN_API_KEY                  - Baseten API key
  OPENROUTER_API_KEY               - OpenRouter API key
  AI_GATEWAY_API_KEY               - Vercel AI Gateway API key
  ZAI_API_KEY                      - ZAI Coding Plan API key (Global)
  ZAI_CODING_CN_API_KEY            - ZAI Coding Plan API key (China)
  MISTRAL_API_KEY                  - Mistral API key
  MINIMAX_API_KEY                  - MiniMax API key
  MOONSHOT_API_KEY                 - Moonshot AI API key
  OPENCODE_API_KEY                 - OpenCode Zen/OpenCode Go API key
  KIMI_API_KEY                     - Kimi For Coding API key
  CLOUDFLARE_API_KEY               - Cloudflare API token (Workers AI and AI Gateway)
  CLOUDFLARE_ACCOUNT_ID            - Cloudflare account id (required for both)
  CLOUDFLARE_GATEWAY_ID            - Cloudflare AI Gateway slug (required for AI Gateway)
  QWEN_TOKEN_PLAN_API_KEY          - Qwen Token Plan API key (international region)
  QWEN_TOKEN_PLAN_CN_API_KEY       - Qwen Token Plan API key (China region)
  XIAOMI_API_KEY                   - Xiaomi MiMo API key (api.xiaomimimo.com billing)
  XIAOMI_TOKEN_PLAN_CN_API_KEY     - Xiaomi MiMo Token Plan API key (China region)
  XIAOMI_TOKEN_PLAN_AMS_API_KEY    - Xiaomi MiMo Token Plan API key (Amsterdam region)
  XIAOMI_TOKEN_PLAN_SGP_API_KEY    - Xiaomi MiMo Token Plan API key (Singapore region)
  AWS_PROFILE                      - AWS profile for Amazon Bedrock
  AWS_ACCESS_KEY_ID                - AWS access key for Amazon Bedrock
  AWS_SECRET_ACCESS_KEY            - AWS secret key for Amazon Bedrock
  AWS_BEARER_TOKEN_BEDROCK         - Bedrock API key (bearer token)
  AWS_REGION                       - AWS region for Amazon Bedrock (e.g., us-east-1)
  {ENV_AGENT_DIR:<32} - Config directory (default: ~/{CONFIG_DIR_NAME}/agent)
  {ENV_SESSION_DIR:<32} - Session storage directory (overridden by --session-dir)
  PI_PACKAGE_DIR                   - Override package directory (for Nix/Guix store paths)
  PI_OFFLINE                       - Disable startup network operations when set to 1/true/yes
  PI_TELEMETRY                     - Override install telemetry when set to 1/true/yes or 0/false/no
  PI_SHARE_VIEWER_URL              - Base URL for /share command (default: https://pi.dev/session/)

Built-in Tool Names:
  read   - Read file contents
  bash   - Execute bash commands
  edit   - Edit files with find/replace
  write  - Write files (creates/overwrites)
  grep   - Search file contents (read-only, off by default)
  find   - Find files by glob pattern (read-only, off by default)
  ls     - List directory contents (read-only, off by default)
"""
    )