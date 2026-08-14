"""CLI 辅助工具。

提供 OAuth 登录的命令行入口。
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, cast

from .providers.registry import builtin_providers

AUTH_FILE = "auth.json"


def _load_auth() -> dict[str, Any]:
    """加载本地认证文件。"""
    path = Path(AUTH_FILE)
    if not path.exists():
        return {}
    try:
        return cast("dict[str, Any]", json.loads(path.read_text("utf-8")))
    except Exception:
        return {}


def _save_auth(auth: dict[str, Any]) -> None:
    """保存认证到本地文件。"""
    Path(AUTH_FILE).write_text(
        json.dumps(auth, indent=2, ensure_ascii=False),
        "utf-8",
    )


async def _login(provider_id: str) -> None:
    """执行 OAuth 登录。"""
    providers = [
        p
        for p in builtin_providers()
        if hasattr(p, "auth") and getattr(p.auth, "oauth", None)
    ]
    provider = next((p for p in providers if p.id == provider_id), None)
    if not provider:
        raise ValueError(f"Unknown provider: {provider_id}")

    credential = await provider.auth.oauth.login(
        interaction={
            "signal": ...,
            "notify": lambda event: _on_notify(event),
            "prompt": lambda prompt: _on_prompt(prompt),
        }
    )
    auth = _load_auth()
    auth[provider_id] = credential
    _save_auth(auth)
    print(f"\nCredentials saved to {AUTH_FILE}")


def _on_notify(event: Any) -> None:
    """处理 OAuth 通知事件。"""
    event_type = getattr(event, "type", None) or event.get("type", "")
    if event_type == "auth_url":
        print(f"\nOpen this URL in your browser:\n{event.get('url', '')}")
        instructions = event.get("instructions", "")
        if instructions:
            print(instructions)
    elif event_type == "device_code":
        print(f"\nOpen this URL in your browser:\n{event.get('verificationUri', '')}")
        print(f"Enter code: {event.get('userCode', '')}")
    elif event_type in ("info", "progress"):
        print(event.get("message", ""))


async def _on_prompt(prompt: Any) -> str:
    """处理 OAuth 提示。"""
    prompt_type = getattr(prompt, "type", None) or prompt.get("type", "")
    message = getattr(prompt, "message", None) or prompt.get("message", "")
    if prompt_type == "select":
        options = getattr(prompt, "options", None) or prompt.get("options", [])
        print(f"\n{message}")
        for i, opt in enumerate(options):
            label = getattr(opt, "label", None) or opt.get("label", "")
            print(f"  {i + 1}. {label}")
        choice = int(input(f"Enter number (1-{len(options)}): ")) - 1
        selected = options[choice]
        return cast(str, getattr(selected, "id", None) or selected.get("id", ""))
    placeholder = getattr(prompt, "placeholder", None) or prompt.get("placeholder", "")
    prompt_text = f"{message}"
    if placeholder:
        prompt_text += f" ({placeholder})"
    prompt_text += ": "
    return input(prompt_text)


async def _main_async() -> None:
    """异步主入口。"""
    args = sys.argv[1:]
    command = args[0] if args else "help"

    providers = [
        p
        for p in builtin_providers()
        if hasattr(p, "auth") and getattr(p.auth, "oauth", None)
    ]

    if command in ("help", "--help", "-h"):
        provider_list = "\n".join(f"  {p.id:<20} {p.name}" for p in providers)
        print(
            f"Usage: python -m pi_ai.cli <command> [provider]\n\n"
            f"Commands:\n"
            f"  login [provider]  Login to an OAuth provider\n"
            f"  list              List available providers\n\n"
            f"Providers:\n{provider_list}"
        )
        return

    if command == "list":
        for p in providers:
            print(f"{p.id:<20} {p.name}")
        return

    if command == "login":
        provider_id = args[1] if len(args) > 1 else None
        if not provider_id:
            for i, p in enumerate(providers):
                print(f"  {i + 1}. {p.name}")
            choice = int(input(f"Enter number (1-{len(providers)}): ")) - 1
            provider_id = providers[choice].id if 0 <= choice < len(providers) else None
        if not provider_id or not any(p.id == provider_id for p in providers):
            raise ValueError(f"Unknown provider: {provider_id or ''}")
        await _login(provider_id)
        return

    raise ValueError(f"Unknown command: {command}")


def main() -> None:
    """CLI 主入口。"""
    try:
        asyncio.run(_main_async())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
