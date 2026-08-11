"""Amazon Bedrock Provider（对应 ``amazon-bedrock.ts``）。"""

from __future__ import annotations

from typing import Any, cast

from ..api.bedrock_converse_stream_lazy import bedrock_converse_stream_api
from ..auth.types import ApiKeyAuth, ApiKeyCredential, AuthResult
from ..models import CreateProviderOptions, create_provider
from .amazon_bedrock_models import AMAZON_BEDROCK_MODELS


class _AmazonBedrockAuth:
    """Amazon Bedrock 认证实现。

    Bedrock 接受 bearer token 或 AWS SDK 默认凭证链。
    login 可以存储 token/profile 选择；resolve 也会检测环境中的 AWS 凭证。
    """

    name = "AWS credentials or bearer token"

    async def login(self, interaction: Any) -> ApiKeyCredential:
        if interaction.signal is not None:
            interaction.signal.throw_if_cancelled()
        method = await interaction.prompt(
            {
                "type": "select",
                "message": "Select Amazon Bedrock authentication method:",
                "options": [
                    {"id": "bearer-token", "label": "Bearer token"},
                    {"id": "aws-profile", "label": "AWS profile"},
                    {
                        "id": "credential-chain",
                        "label": "Existing AWS credential chain",
                    },
                ],
            }
        )
        if interaction.signal is not None:
            interaction.signal.throw_if_cancelled()
        if method == "bearer-token":
            return {
                "type": "api_key",
                "key": await interaction.prompt(
                    {
                        "type": "secret",
                        "message": "Enter Amazon Bedrock bearer token",
                        "signal": None,
                        "placeholder": None,
                        "options": [],
                    }
                ),
            }
        interaction.notify(
            {
                "type": "info",
                "message": "Amazon Bedrock supports AWS profiles, IAM credentials, and role-based credentials.",
                "links": [
                    {
                        "label": "AWS credential provider chain",
                        "url": "https://docs.aws.amazon.com/sdkref/latest/guide/standardized-credentials.html",
                    },
                ],
            }
        )
        if method == "aws-profile":
            return {
                "type": "api_key",
                "env": {
                    "AWS_PROFILE": await interaction.prompt(
                        {
                            "type": "text",
                            "message": "Enter AWS profile name",
                            "signal": None,
                            "placeholder": None,
                            "options": [],
                        }
                    ),
                },
            }
        if method != "credential-chain":
            raise RuntimeError(f"Unknown Amazon Bedrock auth method: {method}")
        await interaction.prompt(
            {
                "type": "text",
                "message": "Configure AWS credentials, then press Enter to continue",
                "signal": None,
                "placeholder": None,
                "options": [],
            }
        )
        return {"type": "api_key"}

    async def check(self, input: dict[str, Any]) -> Any:
        return None

    async def resolve(self, input: dict[str, Any]) -> AuthResult | None:
        ctx: Any = input.get("ctx")
        credential: Any = input.get("credential")
        signal: Any = input.get("signal")

        async def _env(name: str) -> str | None:
            if signal is not None:
                signal.throw_if_cancelled()
            value: str | None = cast("str | None", await ctx.env(name))
            if signal is not None:
                signal.throw_if_cancelled()
            return value

        if credential is not None and credential.get("key"):
            return {
                "auth": {"api_key": credential["key"]},
                "env": credential.get("env"),
                "source": "stored credential",
            }
        if await _env("AWS_BEARER_TOKEN_BEDROCK"):
            return {"auth": {}, "source": "AWS_BEARER_TOKEN_BEDROCK"}
        aws_profile = (
            credential.get("env", {}).get("AWS_PROFILE") if credential else None
        ) or (await _env("AWS_PROFILE"))
        if aws_profile:
            return {
                "auth": {},
                "env": credential.get("env") if credential else None,
                "source": "stored credential" if credential else "AWS_PROFILE",
            }
        if await _env("AWS_ACCESS_KEY_ID") and await _env("AWS_SECRET_ACCESS_KEY"):
            return {"auth": {}, "source": "AWS access keys"}
        if await _env("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"):
            return {"auth": {}, "source": "ECS task role"}
        if await _env("AWS_CONTAINER_CREDENTIALS_FULL_URI"):
            return {"auth": {}, "source": "ECS task role"}
        if await _env("AWS_WEB_IDENTITY_TOKEN_FILE"):
            return {"auth": {}, "source": "web identity token"}
        return None


def _amazon_bedrock_auth() -> ApiKeyAuth:
    """创建 Amazon Bedrock 认证实现。"""
    return _AmazonBedrockAuth()


def amazon_bedrock_provider() -> Any:
    """创建 Amazon Bedrock Provider。"""
    return create_provider(
        CreateProviderOptions(
            id="amazon-bedrock",
            name="Amazon Bedrock",
            models=list(AMAZON_BEDROCK_MODELS.values()),
            api=bedrock_converse_stream_api(),
        )
    )
