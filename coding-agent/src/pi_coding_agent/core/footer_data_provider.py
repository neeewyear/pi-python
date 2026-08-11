from __future__ import annotations

import asyncio
import os
from collections.abc import Callable
from pathlib import Path

from ..utils.fs_watch import (  # type: ignore[import-untyped]
    FS_WATCH_RETRY_DELAY_MS,
    close_watcher,
    watch_with_error_handler,
)


class GitPaths:
    def __init__(self, repo_dir: str, common_git_dir: str, head_path: str) -> None:
        self.repo_dir = repo_dir
        self.common_git_dir = common_git_dir
        self.head_path = head_path


def find_git_paths(cwd: str) -> GitPaths | None:
    """Find git metadata paths by walking up from cwd."""
    dir_ = Path(cwd).resolve()
    while True:
        git_path = dir_ / ".git"
        if git_path.exists():
            try:
                if git_path.is_file():
                    content = git_path.read_text("utf-8").strip()
                    if content.startswith("gitdir: "):
                        git_dir = (dir_ / content[8:].strip()).resolve()
                        head_path = git_dir / "HEAD"
                        if not head_path.exists():
                            return None
                        common_dir_path = git_dir / "commondir"
                        if common_dir_path.exists():
                            common_git_dir = (
                                git_dir / common_dir_path.read_text("utf-8").strip()
                            ).resolve()
                        else:
                            common_git_dir = git_dir
                        return GitPaths(str(dir_), str(common_git_dir), str(head_path))
                elif git_path.is_dir():
                    head_path = git_path / "HEAD"
                    if not head_path.exists():
                        return None
                    return GitPaths(str(dir_), str(git_path), str(head_path))
            except OSError:
                return None
        parent = dir_.parent
        if parent == dir_:
            return None
        dir_ = parent


def _resolve_branch_with_git_sync(repo_dir: str) -> str | None:
    import subprocess

    try:
        result = subprocess.run(
            [
                "git",
                "--no-optional-locks",
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            cwd=repo_dir,
            capture_output=True,
            text=True,
            timeout=10,
        )
        branch = result.stdout.strip() if result.returncode == 0 else ""
        return branch or None
    except (subprocess.SubprocessError, FileNotFoundError):
        return None


async def _resolve_branch_with_git_async(repo_dir: str) -> str | None:
    import asyncio

    try:
        proc = await asyncio.create_subprocess_exec(
            "git",
            "--no-optional-locks",
            "symbolic-ref",
            "--quiet",
            "--short",
            "HEAD",
            cwd=repo_dir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        branch = stdout.decode("utf-8").strip() if proc.returncode == 0 else ""
        return branch or None
    except (FileNotFoundError, OSError):
        return None


def _is_wsl_environment() -> bool:
    return os.name == "posix" and bool(
        os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP")
    )


def _is_windows_mounted_repo_path(repo_dir: str) -> bool:
    import re

    return bool(re.match(r"^/mnt/[a-z](/|$)", repo_dir, re.IGNORECASE))


def _should_poll_git_head(repo_dir: str) -> bool:
    return _is_wsl_environment() and _is_windows_mounted_repo_path(repo_dir)


class FooterDataProvider:
    """Provides git branch and extension statuses."""

    WATCH_DEBOUNCE_MS = 500

    def __init__(self, cwd: str) -> None:
        self._cwd = cwd
        self._extension_statuses: dict[str, str] = {}
        self._cached_branch: str | None = None
        self._git_paths: GitPaths | None = None
        self._head_watcher = None
        self._head_watch_file_path: str | None = None
        self._head_watch_file_listener: Callable[[], None] | None = None
        self._reftable_watcher = None
        self._reftable_tables_list_watcher = None
        self._reftable_tables_list_path: str | None = None
        self._branch_change_callbacks: set[Callable[[], None]] = set()
        self._available_provider_count = 0
        self._refresh_timer: asyncio.TimerHandle | None = None
        self._git_watcher_retry_timer: asyncio.TimerHandle | None = None
        self._refresh_in_flight = False
        self._refresh_pending = False
        self._disposed = False

        self._git_paths = find_git_paths(cwd)
        self._setup_git_watcher()

    def get_git_branch(self) -> str | None:
        """Current git branch, None if not in repo, 'detached' if detached HEAD."""
        if self._cached_branch is None:
            self._cached_branch = self._resolve_git_branch_sync()
        return self._cached_branch

    def get_extension_statuses(self) -> dict[str, str]:
        """Extension status texts."""
        return dict(self._extension_statuses)

    def on_branch_change(self, callback: Callable[[], None]) -> Callable[[], None]:
        """Subscribe to git branch changes. Returns unsubscribe function."""
        self._branch_change_callbacks.add(callback)
        return lambda: self._branch_change_callbacks.discard(callback)

    def set_extension_status(self, key: str, text: str | None) -> None:
        if text is None:
            self._extension_statuses.pop(key, None)
        else:
            self._extension_statuses[key] = text

    def clear_extension_statuses(self) -> None:
        self._extension_statuses.clear()

    def get_available_provider_count(self) -> int:
        return self._available_provider_count

    def set_available_provider_count(self, count: int) -> None:
        self._available_provider_count = count

    def set_cwd(self, cwd: str) -> None:
        if self._cwd == cwd:
            return
        self._cwd = cwd
        if self._refresh_timer:
            self._refresh_timer.cancel()
            self._refresh_timer = None
        self._clear_git_watchers()
        self._cached_branch = None
        self._git_paths = find_git_paths(cwd)
        self._setup_git_watcher()
        self._notify_branch_change()

    def dispose(self) -> None:
        self._disposed = True
        if self._refresh_timer:
            self._refresh_timer.cancel()
            self._refresh_timer = None
        self._clear_git_watchers()
        self._branch_change_callbacks.clear()

    def _notify_branch_change(self) -> None:
        for cb in list(self._branch_change_callbacks):
            cb()

    def _schedule_refresh(self) -> None:
        if self._disposed or self._refresh_timer:
            return
        if self._refresh_in_flight:
            self._refresh_pending = True
            return
        loop = asyncio.get_event_loop()
        self._refresh_timer = loop.call_later(
            FooterDataProvider.WATCH_DEBOUNCE_MS / 1000,
            lambda: asyncio.ensure_future(self._refresh_git_branch_async()),
        )

    async def _refresh_git_branch_async(self) -> None:
        if self._disposed:
            return
        if self._refresh_in_flight:
            self._refresh_pending = True
            return

        self._refresh_in_flight = True
        try:
            next_branch = await self._resolve_git_branch_async()
            if self._disposed:
                return
            if self._cached_branch is not None and self._cached_branch != next_branch:
                self._cached_branch = next_branch
                self._notify_branch_change()
                return
            self._cached_branch = next_branch
        finally:
            self._refresh_in_flight = False
            if self._refresh_pending and not self._disposed:
                self._refresh_pending = False
                self._schedule_refresh()

    def _resolve_git_branch_sync(self) -> str | None:
        try:
            if not self._git_paths:
                return None
            content = Path(self._git_paths.head_path).read_text("utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[16:]
                if branch == ".invalid":
                    return (
                        _resolve_branch_with_git_sync(self._git_paths.repo_dir)
                        or "detached"
                    )
                return branch
            return "detached"
        except OSError:
            return None

    async def _resolve_git_branch_async(self) -> str | None:
        try:
            if not self._git_paths:
                return None
            content = Path(self._git_paths.head_path).read_text("utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                branch = content[16:]
                if branch == ".invalid":
                    result = await _resolve_branch_with_git_async(
                        self._git_paths.repo_dir
                    )
                    return result or "detached"
                return branch
            return "detached"
        except OSError:
            return None

    def _clear_git_watchers(self) -> None:
        close_watcher(self._head_watcher)
        self._head_watcher = None
        if self._head_watch_file_path and self._head_watch_file_listener:
            self._head_watch_file_path = None
            self._head_watch_file_listener = None
        close_watcher(self._reftable_watcher)
        self._reftable_watcher = None
        close_watcher(self._reftable_tables_list_watcher)
        self._reftable_tables_list_watcher = None
        if self._reftable_tables_list_path:
            self._reftable_tables_list_path = None
        if self._git_watcher_retry_timer:
            self._git_watcher_retry_timer.cancel()
            self._git_watcher_retry_timer = None

    def _schedule_git_watcher_retry(self) -> None:
        if self._disposed or self._git_watcher_retry_timer:
            return
        loop = asyncio.get_event_loop()
        self._git_watcher_retry_timer = loop.call_later(
            FS_WATCH_RETRY_DELAY_MS / 1000,
            lambda: self._setup_git_watcher(),
        )

    def _handle_git_watcher_error(self) -> None:
        self._clear_git_watchers()
        self._schedule_git_watcher_retry()

    def _setup_git_watcher(self) -> None:
        self._clear_git_watchers()
        if not self._git_paths:
            return

        poll_git_head = _should_poll_git_head(self._git_paths.repo_dir)
        watch_dir = str(Path(self._git_paths.head_path).parent)

        self._head_watcher = watch_with_error_handler(
            watch_dir,
            lambda event_type, filename: (
                self._schedule_refresh() if not filename or filename == "HEAD" else None
            ),
            lambda: self._handle_git_watcher_error(),
        )

        if poll_git_head:
            self._head_watch_file_path = self._git_paths.head_path
            self._head_watch_file_listener = lambda: self._schedule_refresh()

        if not self._head_watcher and not poll_git_head:
            return

        reftable_dir = Path(self._git_paths.common_git_dir) / "reftable"
        if reftable_dir.exists():
            self._reftable_watcher = watch_with_error_handler(
                str(reftable_dir),
                lambda *args: self._schedule_refresh(),
                lambda: self._handle_git_watcher_error(),
            )
            if not self._reftable_watcher:
                return

            tables_list_path = reftable_dir / "tables.list"
            if tables_list_path.exists():
                self._reftable_tables_list_path = str(tables_list_path)
                self._reftable_tables_list_watcher = watch_with_error_handler(
                    str(tables_list_path),
                    lambda *args: self._schedule_refresh(),
                    lambda: self._handle_git_watcher_error(),
                )
                if not self._reftable_tables_list_watcher:
                    return
