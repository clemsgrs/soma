"""Git provenance of the soma *code* that produced a run or a recorded benchmark row.

Resolved from the installed package directory, never from the working directory:
a run launched from some project folder must record the soma checkout it imported,
and a wheel install (site-packages, no checkout) records nothing rather than the
state of whatever repository happens to enclose the environment.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class GitState:
    sha: str | None
    dirty: bool | None

    @property
    def short(self) -> str | None:
        return self.sha[:7] if self.sha else None


def _soma_package_dir() -> Path:
    import soma

    return Path(soma.__file__).resolve().parent


def _git(repo: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), *args],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.SubprocessError, OSError):
        return None
    return result.stdout.strip()


def soma_git_state() -> GitState:
    """``GitState`` of the checkout the imported ``soma`` package lives in.

    ``sha`` is the full ``HEAD`` commit; ``dirty`` is ``True`` when *tracked* files under
    the checkout have modifications (untracked scratch such as run outputs does not
    count). Both are ``None`` when the package is not inside a git checkout that owns
    it — a wheel install, or a package directory copied out of its repository.
    """
    package_dir = _soma_package_dir()
    toplevel = _git(package_dir, "rev-parse", "--show-toplevel")
    if not toplevel:
        return GitState(sha=None, dirty=None)
    # The checkout must be soma's own: an environment created inside an unrelated
    # repository would otherwise stamp that repository's commit onto soma runs.
    try:
        package_dir.relative_to(Path(toplevel).resolve())
    except ValueError:
        return GitState(sha=None, dirty=None)
    sha = _git(package_dir, "rev-parse", "HEAD")
    if not sha:
        return GitState(sha=None, dirty=None)
    status = _git(package_dir, "status", "--porcelain", "--untracked-files=no")
    return GitState(sha=sha, dirty=None if status is None else bool(status))
