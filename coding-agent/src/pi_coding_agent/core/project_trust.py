from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..config import CONFIG_DIR_NAME

if TYPE_CHECKING:
    from .extensions.types import LoadExtensionsResult, ProjectTrustContext
    from .settings_manager import DefaultProjectTrust
    from .trust_manager import (
        ProjectTrustStore,
        get_project_trust_options,
        has_trust_requiring_project_resources,
    )


class ResolveProjectTrustedOptions:
    def __init__(
        self,
        cwd: str,
        trust_store: ProjectTrustStore,
        trust_override: bool | None = None,
        default_project_trust: DefaultProjectTrust | None = None,
        extensions_result: LoadExtensionsResult | None = None,
        project_trust_context: ProjectTrustContext | None = None,
        on_extension_error: Callable[..., Any] | None = None,
    ) -> None:
        self.cwd = cwd
        self.trust_store = trust_store
        self.trust_override = trust_override
        self.default_project_trust = default_project_trust
        self.extensions_result = extensions_result
        self.project_trust_context = project_trust_context
        self.on_extension_error = on_extension_error


def format_project_trust_prompt(cwd: str) -> str:
    return (
        f"Trust project folder?\n{cwd}\n\n"
        f"This allows pi to load {CONFIG_DIR_NAME} settings and resources, "
        f"install missing project packages, and execute project extensions."
    )


async def resolve_project_trusted(options: ResolveProjectTrustedOptions) -> bool:
    if options.trust_override is not None:
        return options.trust_override

    if not has_trust_requiring_project_resources(options.cwd):
        return True

    if options.extensions_result and options.project_trust_context is not None:
        from .extensions.runner import emit_project_trust_event
        from .extensions.types import ProjectTrustEvent

        event_output = await emit_project_trust_event(
            options.extensions_result,
            ProjectTrustEvent(type="project_trust", cwd=options.cwd),
            options.project_trust_context,
        )
        result = event_output.get("result")
        errors = event_output.get("errors", [])
        for error in errors:
            if options.on_extension_error:
                options.on_extension_error(
                    f'Extension "{error["extension_path"]}" project_trust error: {error["error"]}'
                )
        if result and isinstance(result, dict):
            trusted = result.get("trusted") == "yes"
            if result.get("remember") is True:
                options.trust_store.set(options.cwd, trusted)
            return trusted

    decision = options.trust_store.get(options.cwd)
    if decision is not None:
        return decision

    default_trust = options.default_project_trust or "ask"
    if default_trust == "always":
        return True
    elif default_trust == "never":
        return False

    if not options.project_trust_context or not options.project_trust_context.has_ui:
        return False

    options_list = get_project_trust_options(options.cwd, include_session_only=True)
    labels = [o.label for o in options_list]
    ui = options.project_trust_context.ui
    if ui is None:
        return False
    selected_label = await ui.select(
        format_project_trust_prompt(options.cwd),
        labels,
    )  # type: ignore[call-arg]
    selected = next((o for o in options_list if o.label == selected_label), None)
    if selected is not None:
        if selected.updates:
            options.trust_store.set_many(selected.updates)
        return selected.trusted
    return False
