"""Tests for the CI workflows that build Docker images from Docker Hub bases."""

import os
import subprocess
from pathlib import Path

import pytest
import yaml

DOCKERHUB_USERNAME = "${{ secrets.DOCKERHUB_USERNAME }}"
DOCKERHUB_TOKEN = "${{ secrets.DOCKERHUB_TOKEN }}"


def _load_workflow(path: str) -> dict:
    return yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)


def _docker_test_steps() -> list[dict]:
    workflow = _load_workflow(".github/workflows/pr-test.yaml")
    return workflow["jobs"]["docker-test"]["steps"]


def _step_index(steps: list[dict], name: str) -> int:
    return next(i for i, step in enumerate(steps) if step.get("name") == name)


def _is_dockerhub_login(step: dict) -> bool:
    return (
        step.get("uses", "").startswith("docker/login-action@")
        and step.get("with", {}).get("username") == DOCKERHUB_USERNAME
        and step.get("with", {}).get("password") == DOCKERHUB_TOKEN
    )


def test_pr_test_logs_in_to_docker_hub_before_buildx_setup_and_ci_image_build():
    # setup-buildx-action pulls the moby/buildkit builder image from Docker
    # Hub, so the login must come before it, not just before the build.
    steps = _docker_test_steps()
    login = _step_index(steps, "Log in to Docker Hub")

    assert steps[login]["uses"] == "docker/login-action@v3"
    assert _is_dockerhub_login(steps[login])
    assert login < _step_index(steps, "Set up Docker Buildx")
    assert login < _step_index(steps, "Build CI image")


def test_pr_test_docker_hub_login_runs_with_the_suite_and_skips_fork_prs():
    steps = _docker_test_steps()
    condition = steps[_step_index(steps, "Log in to Docker Hub")]["if"]

    # Same gate as the image build: docs-only PRs skip the whole Docker path.
    assert "steps.scope.outputs.run == 'true'" in condition
    # Fork PRs get no secrets: skip the login and fall back to anonymous pulls.
    assert (
        "(github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository)"
    ) in condition
    assert " && " in condition


def test_docker_test_job_still_always_reports_a_status():
    # The main-branch ruleset requires the `docker-test` check by name, so the
    # job keeps its name and its scope step runs unconditionally; only the
    # Docker steps (login included) are gated on the scope output.
    job = _load_workflow(".github/workflows/pr-test.yaml")["jobs"]["docker-test"]
    steps = job["steps"]

    assert job["if"] == "${{ !startsWith(github.head_ref, 'release-') }}"
    assert "if" not in steps[_step_index(steps, "Determine whether code changed")]


def _builds_dockerhub_dockerfile(step: dict) -> bool:
    if step.get("uses", "").startswith("docker/build-push-action@"):
        return step.get("with", {}).get("file", "Dockerfile") in {"Dockerfile", "Dockerfile.ci"}
    return "docker build" in step.get("run", "")


def _pulls_from_docker_hub(step: dict) -> bool:
    # setup-buildx-action pulls moby/buildkit from Docker Hub; Dockerfile and
    # Dockerfile.ci both start FROM a Docker Hub image.
    return step.get("uses", "").startswith(
        "docker/setup-buildx-action@"
    ) or _builds_dockerhub_dockerfile(step)


def test_every_docker_hub_pull_is_preceded_by_a_docker_hub_login():
    pulls = []
    for path in sorted(Path(".github/workflows").glob("*.y*ml")):
        for job_name, job in _load_workflow(str(path))["jobs"].items():
            steps = job.get("steps", [])
            for i, step in enumerate(steps):
                if not _pulls_from_docker_hub(step):
                    continue
                pulls.append((path.name, job_name, step.get("name")))
                assert any(_is_dockerhub_login(prior) for prior in steps[:i]), (
                    f"{path.name}:{job_name} pulls from Docker Hub in step "
                    f"{step.get('name')!r} without logging in to Docker Hub first"
                )

    assert ("pr-test.yaml", "docker-test", "Set up Docker Buildx") in pulls
    assert ("pr-test.yaml", "docker-test", "Build CI image") in pulls
    assert ("pr-test.yaml", "prism-regression", "Set up Docker Buildx") in pulls
    assert ("pr-test.yaml", "prism-regression", "Build CI image") in pulls
    assert ("docker.yaml", "docker", "Set up Docker Buildx") in pulls
    assert ("docker.yaml", "docker", "Build and push Docker image") in pulls


PRISM_TEST = "prism_slide_feature_matches_slide2vec_gt"
SAME_REPO_GUARD = (
    "(github.event_name != 'pull_request' || "
    "github.event.pull_request.head.repo.full_name == github.repository)"
)


def _pr_test_jobs() -> dict:
    return _load_workflow(".github/workflows/pr-test.yaml")["jobs"]


def _prism_steps() -> list[dict]:
    return _pr_test_jobs()["prism-regression"]["steps"]


def test_prism_regression_runs_in_its_own_parallel_job():
    jobs = _pr_test_jobs()
    prism = jobs["prism-regression"]

    # Parallel with the suite: no dependency on docker-test.
    assert "needs" not in prism
    run = _prism_steps()[_step_index(_prism_steps(), "Run PRISM regression in container")]
    assert f"-k {PRISM_TEST}" in run["run"]
    assert "--shm-size=2g" in run["run"]
    assert run["env"]["SOMA_RUN_PRISM_REGRESSION"] == "1"
    assert prism["env"]["HF_TOKEN"] == "${{ secrets.HF_TOKEN }}"


def test_docker_test_no_longer_runs_the_prism_regression():
    steps = _docker_test_steps()
    runs = [step.get("run", "") for step in steps]

    assert not any("SOMA_RUN_PRISM_REGRESSION" in run for run in runs)
    assert all(step.get("name") != "Run PRISM regression in container" for step in steps)
    suite = steps[_step_index(steps, "Run full test suite in container")]["run"]
    assert f"-k 'not {PRISM_TEST}'" in suite


PRISM_SCOPE_STEP = "Determine whether extraction code changed"
SCOPE_ENV = {
    "EVENT_NAME": "${{ github.event_name }}",
    "BASE_SHA": "${{ github.event.pull_request.base.sha }}",
    "HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
    "HEAD_REF": "${{ github.head_ref }}",
    "HEAD_REPO": "${{ github.event.pull_request.head.repo.full_name }}",
    "REPO": "${{ github.repository }}",
}


def _run_prism_scope(
    tmp_path: Path,
    *,
    event: str = "pull_request",
    changed: tuple[str, ...] = (),
    head_ref: str = "some-feature",
    head_repo: str = "clemsgrs/soma",
) -> str:
    """Execute the real scope step with a stub ``git`` that reports ``changed``."""
    steps = _prism_steps()
    script = steps[_step_index(steps, PRISM_SCOPE_STEP)]["run"]
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    diff = tmp_path / "diff.txt"
    diff.write_text("".join(f"{path}\n" for path in changed))
    git = bin_dir / "git"
    git.write_text(f'#!/usr/bin/env bash\n[ "$1" = diff ] && cat "{diff}"\n')
    git.chmod(0o755)
    output = tmp_path / "github_output"
    output.write_text("")
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GITHUB_OUTPUT": str(output),
        "EVENT_NAME": event,
        "BASE_SHA": "base",
        "HEAD_SHA": "head",
        "HEAD_REF": head_ref if event == "pull_request" else "",
        "HEAD_REPO": head_repo if event == "pull_request" else "",
        "REPO": "clemsgrs/soma",
    }
    subprocess.run(["bash", "-c", script], env=env, check=True, capture_output=True)
    outputs = dict(line.split("=", 1) for line in output.read_text().splitlines())
    return outputs["run"]


def test_prism_scope_step_reads_the_event_from_its_env():
    steps = _prism_steps()
    scope = steps[_step_index(steps, PRISM_SCOPE_STEP)]

    assert scope["id"] == "scope"
    assert scope["env"] == SCOPE_ENV
    # Event data reaches the script only through env, never interpolated inline.
    assert "${{" not in scope["run"]


@pytest.mark.parametrize("event", ["push", "workflow_dispatch"])
def test_prism_regression_always_runs_outside_pull_requests(tmp_path, event):
    assert _run_prism_scope(tmp_path, event=event) == "true"


@pytest.mark.parametrize(
    "changed",
    [
        ("pyproject.toml",),
        ("Dockerfile.ci",),
        ("docs/index.md", "soma/extraction/extractor.py"),
        ("soma/encoders/validation.py",),
        ("soma/cache/keys.py",),
        ("soma/preprocessing/supplied_coordinates.py",),
        ("soma/slide2vec_adapter.py",),
        ("tests/fixtures/regression/gt/test-wsi.pt",),
        ("tests/test_regression_fixtures.py",),
        (".github/workflows/pr-test.yaml",),
    ],
)
def test_prism_regression_runs_when_a_pr_touches_the_extraction_path(tmp_path, changed):
    assert _run_prism_scope(tmp_path, changed=changed) == "true"


@pytest.mark.parametrize(
    "changed",
    [
        ("docs/index.md",),
        ("soma/training/trainer.py", "soma/aggregators/mil/abmil.py"),
        ("soma/tasks/segmentation.py", "tests/test_pipeline.py"),
        ("soma/dense_extraction.py", "soma/decoders/heavy_conv.py"),
        # Look-alike names must not match by prefix.
        ("soma/features_extra.py", "soma/configs/eva/bach.yaml", "pyproject.toml.bak"),
    ],
)
def test_prism_regression_skips_prs_off_the_extraction_path(tmp_path, changed):
    assert _run_prism_scope(tmp_path, changed=changed) == "false"


def test_prism_regression_skips_fork_prs_which_get_no_secrets(tmp_path):
    changed = ("soma/extraction/extractor.py",)

    assert _run_prism_scope(tmp_path, changed=changed, head_repo="someone/soma") == "false"


def test_prism_regression_skips_release_version_bump_prs_like_docker_test(tmp_path):
    changed = ("pyproject.toml",)

    assert _run_prism_scope(tmp_path, changed=changed, head_ref="release-1.20.0") == "false"


def test_prism_regression_job_always_reports_and_skips_only_its_docker_steps():
    # Should the ruleset require `prism-regression`, the job must report a status
    # on every PR: no job-level `if`, the scope step always runs, and every other
    # step after checkout is gated on the scope output.
    job = _pr_test_jobs()["prism-regression"]
    steps = job["steps"]
    scope = _step_index(steps, PRISM_SCOPE_STEP)

    assert "if" not in job
    assert "if" not in steps[scope]
    assert scope < _step_index(steps, "Log in to Docker Hub")
    for step in steps[scope + 1 :]:
        assert "steps.scope.outputs.run == 'true'" in step["if"], step["name"]


def test_prism_regression_keeps_the_fork_pr_and_hf_token_guards():
    steps = _prism_steps()
    guard = steps[_step_index(steps, "Guard required secret")]

    assert 'test -n "${HF_TOKEN:-}"' in guard["run"]
    for name in ("Log in to Docker Hub", "Guard required secret", "Run PRISM regression in container"):
        condition = steps[_step_index(steps, name)]["if"]
        assert SAME_REPO_GUARD in condition, name
    assert _step_index(steps, "Guard required secret") < _step_index(
        steps, "Run PRISM regression in container"
    )


def test_prism_regression_reuses_the_shared_gha_build_cache():
    steps = _prism_steps()
    build = steps[_step_index(steps, "Build CI image")]

    assert build["with"]["file"] == "Dockerfile.ci"
    assert build["with"]["cache-from"] == "type=gha"
