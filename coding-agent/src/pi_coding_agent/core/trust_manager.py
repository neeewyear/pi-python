"""项目信任管理。

提供 ``ProjectTrustStore`` 类以及辅助函数，用于管理项目信任决策。
"""

from __future__ import annotations

from pathlib import Path
from typing import TypeAlias

import orjson

from ..config import CONFIG_DIR_NAME, get_agent_dir

# ---------------------------------------------------------------------------
# 类型别名
# ---------------------------------------------------------------------------

ProjectTrustDecision: TypeAlias = bool | None
"""项目信任决策（``True``=信任, ``False``=不信任, ``None``=未决策）。"""


# ---------------------------------------------------------------------------
# 数据模型
# ---------------------------------------------------------------------------


class ProjectTrustStoreEntry:
    """项目信任存储条目。"""

    def __init__(self, path: str, decision: bool) -> None:
        self.path = path
        self.decision = decision


class ProjectTrustUpdate:
    """项目信任更新。"""

    def __init__(self, path: str, decision: ProjectTrustDecision) -> None:
        self.path = path
        self.decision = decision


class ProjectTrustOption:
    """项目信任选项。"""

    def __init__(
        self,
        label: str,
        trusted: bool,
        updates: list[ProjectTrustUpdate],
        saved_path: str | None = None,
    ) -> None:
        self.label = label
        self.trusted = trusted
        self.updates = updates
        self.saved_path = saved_path


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES: tuple[str, ...] = (
    "settings.json",
    "extensions",
    "skills",
    "prompts",
    "themes",
    "SYSTEM.md",
    "APPEND_SYSTEM.md",
)

# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def _normalize_cwd(cwd: str) -> str:
    """规范化工作目录路径。"""
    return str(Path(cwd).resolve())


def _find_nearest_trust_entry(
    data: dict[str, bool | None],
    cwd: str,
) -> ProjectTrustStoreEntry | None:
    """从当前目录向上查找最近的信任条目。"""
    current = _normalize_cwd(cwd)
    while True:
        value = data.get(current)
        if value is True or value is False:
            return ProjectTrustStoreEntry(path=current, decision=value)
        parent = str(Path(current).parent)
        if parent == current:
            return None
        current = parent


def get_project_trust_parent_path(cwd: str) -> str | None:
    """获取项目信任的父目录路径。"""
    trust_path = _normalize_cwd(cwd)
    parent = str(Path(trust_path).parent)
    return None if parent == trust_path else parent


def get_project_trust_options(
    cwd: str,
    include_session_only: bool = False,
) -> list[ProjectTrustOption]:
    """获取项目信任选项列表。"""
    trust_path = _normalize_cwd(cwd)
    options: list[ProjectTrustOption] = [
        ProjectTrustOption(
            label="Trust",
            trusted=True,
            updates=[ProjectTrustUpdate(path=trust_path, decision=True)],
            saved_path=trust_path,
        ),
    ]
    parent_path = get_project_trust_parent_path(cwd)
    if parent_path is not None:
        options.append(
            ProjectTrustOption(
                label=f"Trust parent folder ({parent_path})",
                trusted=True,
                updates=[
                    ProjectTrustUpdate(path=parent_path, decision=True),
                    ProjectTrustUpdate(path=trust_path, decision=None),
                ],
                saved_path=parent_path,
            ),
        )
    if include_session_only:
        options.append(
            ProjectTrustOption(
                label="Trust (this session only)",
                trusted=True,
                updates=[],
            ),
        )
    options.append(
        ProjectTrustOption(
            label="Do not trust",
            trusted=False,
            updates=[ProjectTrustUpdate(path=trust_path, decision=False)],
            saved_path=trust_path,
        ),
    )
    if include_session_only:
        options.append(
            ProjectTrustOption(
                label="Do not trust (this session only)",
                trusted=False,
                updates=[],
            ),
        )
    return options


def has_trust_requiring_project_resources(cwd: str) -> bool:
    """检查项目目录是否包含需要信任的资源。

    检查 ``cwd/.pi/`` 下是否有需要信任的配置文件，以及 ``cwd/.agents/skills/``
    是否存在（排除用户全局 ``~/.agents/skills``）。
    """
    home_dir = str(Path.home())
    user_agents_skills_dir = str(Path(home_dir) / ".agents" / "skills")
    current = _normalize_cwd(cwd)

    config_dir = Path(current) / CONFIG_DIR_NAME
    for entry in TRUST_REQUIRING_PROJECT_CONFIG_RESOURCES:
        if (config_dir / entry).exists():
            return True

    while True:
        agents_skills_dir = str(Path(current) / ".agents" / "skills")
        if (
            agents_skills_dir != user_agents_skills_dir
            and Path(agents_skills_dir).exists()
        ):
            return True
        parent = str(Path(current).parent)
        if parent == current:
            return False
        current = parent


# ---------------------------------------------------------------------------
# ProjectTrustStore
# ---------------------------------------------------------------------------


class ProjectTrustStore:
    """项目信任存储。

    读取/写入 ``trust.json`` 文件，管理项目目录的信任决策。
    """

    def __init__(self, agent_dir: str | None = None) -> None:
        agent_path = Path(agent_dir) if agent_dir else get_agent_dir()
        self._trust_path = agent_path / "trust.json"

    # ------------------------------------------------------------------
    # 内部文件操作
    # ------------------------------------------------------------------

    def _read_trust_file(self) -> dict[str, bool | None]:
        """读取信任文件。"""
        if not self._trust_path.exists():
            return {}
        try:
            raw = self._trust_path.read_bytes()
            data = orjson.loads(raw)
            if not isinstance(data, dict):
                msg = f"Invalid trust store {self._trust_path}: expected an object"
                raise TypeError(msg)
            result: dict[str, bool | None] = {}
            for key, value in data.items():
                if value is True or value is False or value is None:
                    result[str(key)] = value
                else:
                    msg = f"Invalid trust store {self._trust_path}: value for {key!r} must be true, false, or null"
                    raise ValueError(msg)
            return result
        except (FileNotFoundError, orjson.JSONDecodeError) as exc:
            msg = f"Failed to read trust store {self._trust_path}: {exc}"
            raise OSError(msg) from exc

    def _write_trust_file(self, data: dict[str, bool | None]) -> None:
        """写入信任文件。"""
        sorted_data: dict[str, bool | None] = {}
        for key in sorted(data):
            val = data[key]
            if val is True or val is False or val is None:
                sorted_data[key] = val
        self._trust_path.parent.mkdir(parents=True, exist_ok=True)
        raw = orjson.dumps(
            sorted_data, option=orjson.OPT_INDENT_2 | orjson.OPT_APPEND_NEWLINE
        )
        self._trust_path.write_bytes(raw)

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def get(self, cwd: str) -> ProjectTrustDecision:
        """获取项目目录的信任决策。"""
        entry = self.get_entry(cwd)
        return entry.decision if entry else None

    def get_entry(self, cwd: str) -> ProjectTrustStoreEntry | None:
        """获取项目目录的信任条目。"""
        data = self._read_trust_file()
        return _find_nearest_trust_entry(data, cwd)

    def set(self, cwd: str, decision: ProjectTrustDecision) -> None:
        """设置项目目录的信任决策。"""
        self.set_many([ProjectTrustUpdate(path=cwd, decision=decision)])

    def set_many(self, decisions: list[ProjectTrustUpdate]) -> None:
        """批量设置信任决策。"""
        data = self._read_trust_file()
        for update in decisions:
            key = _normalize_cwd(update.path)
            if update.decision is None:
                data.pop(key, None)
            else:
                data[key] = update.decision
        self._write_trust_file(data)


__all__ = [
    "ProjectTrustDecision",
    "ProjectTrustOption",
    "ProjectTrustStore",
    "ProjectTrustStoreEntry",
    "ProjectTrustUpdate",
    "get_project_trust_options",
    "get_project_trust_parent_path",
    "has_trust_requiring_project_resources",
]
