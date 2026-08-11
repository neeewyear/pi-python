from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .package_manager import PathMetadata

SourceScope = str  # "user" | "project" | "temporary"
SourceOrigin = str  # "package" | "top-level"


class SourceInfo:
    def __init__(
        self,
        path: str,
        source: str,
        scope: SourceScope = "temporary",
        origin: SourceOrigin = "top-level",
        base_dir: str | None = None,
    ) -> None:
        self.path = path
        self.source = source
        self.scope = scope
        self.origin = origin
        self.base_dir = base_dir


def create_source_info(path: str, metadata: PathMetadata) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=metadata.source,
        scope=metadata.scope,
        origin=metadata.origin,
        base_dir=metadata.base_dir,
    )


def create_synthetic_source_info(
    path: str,
    options: dict[str, Any],
) -> SourceInfo:
    return SourceInfo(
        path=path,
        source=options.get("source", "temporary"),
        scope=options.get("scope", "temporary"),
        origin=options.get("origin", "top-level"),
        base_dir=options.get("base_dir"),
    )
