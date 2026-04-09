import argparse
import re
import subprocess
from pathlib import Path


PYPROJECT_PATH = Path("pyproject.toml")
VERSION_PATTERN = re.compile(r'(?m)^(version\s*=\s*")(\d+)\.(\d+)\.(\d+)(")$')


def run(cmd: str, check: bool = True) -> str:
    result = subprocess.run(cmd, shell=True, check=check, stdout=subprocess.PIPE)
    return result.stdout.decode().strip()


def release_branch(version: str) -> str:
    return f"release-{version}"


def release_tag(version: str) -> str:
    return version


def _parse_version(version: str) -> tuple[int, int, int]:
    match = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", version)
    if match is None:
        raise ValueError(f"Unsupported version format: {version}")
    return tuple(int(part) for part in match.groups())


def bump_version_string(version: str, level: str) -> str:
    major, minor, patch = _parse_version(version)
    if level == "patch":
        patch += 1
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "major":
        major += 1
        minor = 0
        patch = 0
    else:
        raise ValueError(f"Unsupported level: {level}")
    return f"{major}.{minor}.{patch}"


def get_current_version(pyproject_path: Path = PYPROJECT_PATH) -> str:
    match = VERSION_PATTERN.search(pyproject_path.read_text())
    if match is None:
        raise RuntimeError(f"Could not find project version in {pyproject_path}")
    return ".".join(match.groups()[1:4])


def update_pyproject_version(pyproject_path: Path, level: str) -> str:
    current = get_current_version(pyproject_path)
    bumped = bump_version_string(current, level)
    updated, count = VERSION_PATTERN.subn(rf'\g<1>{bumped}\g<5>', pyproject_path.read_text(), count=1)
    if count != 1:
        raise RuntimeError(f"Expected to update exactly one version entry in {pyproject_path}")
    pyproject_path.write_text(updated)
    return bumped


def bump_version(level: str = "patch", pyproject_path: Path = PYPROJECT_PATH) -> str:
    print(f"🔧 Bumping version with level: {level}")
    return update_pyproject_version(pyproject_path, level)


def create_branch(branch: str) -> None:
    print(f"🌿 Creating and switching to branch {branch}...")
    run(f"git checkout -b {branch}")


def commit_bump(version: str) -> None:
    print(f"📦 Committing version bump for {version}...")
    run("git add pyproject.toml")
    run(f'git commit -m "Bump version to {version}"')


def push_branch_and_tag(branch: str, version: str) -> None:
    run(f"git push origin {branch}")

    tag = release_tag(version)
    print(f"🏷️ Creating and pushing tag {tag}...")
    run(f"git tag {tag}")
    run(f"git push origin {tag}")


def create_pull_request(branch: str, version: str) -> None:
    print(f"🔁 Creating pull request for {branch} → main...")
    run(
        f'gh pr create --title "Release {version}" '
        f'--body "This PR bumps the version to {version} and tags the release." '
        f"--base main --head {branch}"
    )


def open_release_draft(tag: str) -> None:
    repo = run("git remote get-url origin")
    match = re.search(r"github\.com[:/](.*?)(\.git)?$", repo)
    if not match:
        print("❌ Could not detect GitHub repo")
        return
    repo_path = match.group(1)
    url = f"https://github.com/{repo_path}/releases/new?tag={tag}&title={tag}"
    print(f"🌐 Open the release page:\n{url}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", choices=["patch", "minor", "major"], default="patch", help="Version bump level")
    parser.add_argument("--no-pr", action="store_true", help="Don't create pull request")
    parser.add_argument("--no-draft", action="store_true", help="Don't open GitHub release page")
    args = parser.parse_args()

    run("git checkout main")
    run("git pull origin main")

    version = bump_version(args.level)
    branch = release_branch(version)

    create_branch(branch)
    commit_bump(version)
    push_branch_and_tag(branch, version)

    if not args.no_pr:
        create_pull_request(branch, version)

    if not args.no_draft:
        open_release_draft(release_tag(version))

    print(f"\n✅ Release flow completed for version {version}!")
