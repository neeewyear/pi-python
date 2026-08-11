from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class PiManifest:
    def __init__(
        self,
        extensions: Optional[list[str]] = None,
        skills: Optional[list[str]] = None,
        prompts: Optional[list[str]] = None,
        themes: Optional[list[str]] = None,
    ) -> None:
        self.extensions = extensions
        self.skills = skills
        self.prompts = prompts
        self.themes = themes


RESOURCE_FIELDS = ["extensions", "skills", "prompts", "themes"]


def _is_object(value: object) -> bool:
    return isinstance(value, dict)


def read_pi_manifest(package_json_path: str) -> Optional[PiManifest]:
    try:
        pkg = json.loads(Path(package_json_path).read_text("utf-8"))
        if not _is_object(pkg) or not _is_object(pkg.get("pi")):
            return None

        pi_data = pkg["pi"]
        manifest = PiManifest()
        for field in RESOURCE_FIELDS:
            entries = pi_data.get(field)
            if isinstance(entries, list) and all(isinstance(e, str) for e in entries):
                setattr(manifest, field, entries)
        return manifest
    except (json.JSONDecodeError, OSError):
        return None