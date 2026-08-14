"""PKCE OAuth 流程。"""

from __future__ import annotations

import base64
import hashlib
import secrets


async def generate_pkce() -> dict[str, str]:
    """生成 PKCE code verifier 和 challenge。"""
    verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).rstrip(b"=").decode("ascii")
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return {"verifier": verifier, "challenge": challenge}