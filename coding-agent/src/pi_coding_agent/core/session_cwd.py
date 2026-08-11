from __future__ import annotations

from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    pass


class SessionCwdIssue:
    def __init__(
        self,
        session_file: Optional[str] = None,
        session_cwd: str = "",
        fallback_cwd: str = "",
    ) -> None:
        self.session_file = session_file
        self.session_cwd = session_cwd
        self.fallback_cwd = fallback_cwd


class SessionCwdSource:
    def get_cwd(self) -> str:
        raise NotImplementedError

    def get_session_file(self) -> Optional[str]:
        raise NotImplementedError


def get_missing_session_cwd_issue(
    session_manager: SessionCwdSource,
    fallback_cwd: str,
) -> Optional[SessionCwdIssue]:
    session_file = session_manager.get_session_file()
    if not session_file:
        return None

    session_cwd = session_manager.get_cwd()
    if not session_cwd or Path(session_cwd).exists():
        return None

    return SessionCwdIssue(
        session_file=session_file,
        session_cwd=session_cwd,
        fallback_cwd=fallback_cwd,
    )


def format_missing_session_cwd_error(issue: SessionCwdIssue) -> str:
    session_file_line = f"\nSession file: {issue.session_file}" if issue.session_file else ""
    return (
        f"Stored session working directory does not exist: {issue.session_cwd}"
        f"{session_file_line}\n"
        f"Current working directory: {issue.fallback_cwd}"
    )


def format_missing_session_cwd_prompt(issue: SessionCwdIssue) -> str:
    return (
        f"cwd from session file does not exist\n"
        f"{issue.session_cwd}\n\n"
        f"continue in current cwd\n"
        f"{issue.fallback_cwd}"
    )


class MissingSessionCwdError(Exception):
    def __init__(self, issue: SessionCwdIssue) -> None:
        self.issue = issue
        super().__init__(format_missing_session_cwd_error(issue))
        self.name = "MissingSessionCwdError"


def assert_session_cwd_exists(session_manager: SessionCwdSource, fallback_cwd: str) -> None:
    issue = get_missing_session_cwd_issue(session_manager, fallback_cwd)
    if issue:
        raise MissingSessionCwdError(issue)