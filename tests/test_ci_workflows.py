"""Tests for the CI workflows that build Docker images from Docker Hub bases."""

from pathlib import Path

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


def test_pr_test_logs_in_to_docker_hub_between_buildx_setup_and_ci_image_build():
    steps = _docker_test_steps()
    login = _step_index(steps, "Log in to Docker Hub")

    assert steps[login]["uses"] == "docker/login-action@v3"
    assert _is_dockerhub_login(steps[login])
    assert _step_index(steps, "Set up Docker Buildx") < login
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


def test_every_docker_image_build_is_preceded_by_a_docker_hub_login():
    # Dockerfile and Dockerfile.ci both start FROM a Docker Hub image.
    builds = []
    for path in sorted(Path(".github/workflows").glob("*.y*ml")):
        for job_name, job in _load_workflow(str(path))["jobs"].items():
            steps = job.get("steps", [])
            for i, step in enumerate(steps):
                if not _builds_dockerhub_dockerfile(step):
                    continue
                builds.append((path.name, job_name, step.get("name")))
                assert any(_is_dockerhub_login(prior) for prior in steps[:i]), (
                    f"{path.name}:{job_name} builds an image in step "
                    f"{step.get('name')!r} without logging in to Docker Hub first"
                )

    assert ("pr-test.yaml", "docker-test", "Build CI image") in builds
    assert ("docker.yaml", "docker", "Build and push Docker image") in builds
