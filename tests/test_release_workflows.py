"""Tests for release-triggered publishing workflows."""

from pathlib import Path

import yaml


def _load_workflow(path: str) -> dict:
    return yaml.load(Path(path).read_text(), Loader=yaml.BaseLoader)


def _load_text(path: str) -> str:
    return Path(path).read_text()


def test_release_workflow_publishes_to_pypi_on_published_release():
    workflow = _load_workflow(".github/workflows/release.yaml")

    assert workflow["name"] == "Publish Python Package"
    assert workflow["on"]["release"]["types"] == ["published"]

    steps = workflow["jobs"]["build"]["steps"]
    publish_step = next(step for step in steps if step.get("name") == "Publish to PyPI")

    assert publish_step["env"]["TWINE_USERNAME"] == "__token__"
    assert publish_step["env"]["TWINE_PASSWORD"] == "${{ secrets.PYPI_API_TOKEN }}"
    assert publish_step["run"] == "twine upload --verbose dist/*"


def test_docker_workflow_pushes_version_and_latest_tags_on_published_release():
    workflow = _load_workflow(".github/workflows/docker.yaml")

    assert workflow["name"] == "Docker Hub Release"
    assert workflow["on"]["release"]["types"] == ["published"]

    steps = workflow["jobs"]["docker"]["steps"]
    login_step = next(step for step in steps if step.get("name") == "Log in to Docker Hub")
    push_step = next(step for step in steps if step.get("name") == "Build and push Docker image")

    assert login_step["with"]["username"] == "${{ secrets.DOCKERHUB_USERNAME }}"
    assert login_step["with"]["password"] == "${{ secrets.DOCKERHUB_TOKEN }}"
    assert "docker build -t $IMAGE_BASE:$VERSION -t $IMAGE_BASE:latest ." in push_step["run"]
    assert "docker push $IMAGE_BASE:$VERSION" in push_step["run"]
    assert "docker push $IMAGE_BASE:latest" in push_step["run"]


def test_ci_dockerfile_stays_generic_and_defers_project_install_to_runtime():
    dockerfile = _load_text("Dockerfile.ci")
    workflow = _load_workflow(".github/workflows/pr-test.yaml")

    assert 'COPY --chown=user:user soma /opt/app/soma' not in dockerfile
    assert '"/opt/app[dev]"' not in dockerfile
    assert "--no-build-isolation" not in dockerfile

    test_step = next(
        step for step in workflow["jobs"]["docker-test"]["steps"] if step.get("name") == "Run full test suite in container"
    )
    prism_step = next(
        step for step in workflow["jobs"]["docker-test"]["steps"] if step.get("name") == "Run PRISM regression in container"
    )

    assert "python -m pip install --no-cache-dir -c /tmp/constraints-cu128.txt --no-build-isolation '/opt/app[dev]' pytest pytest-cov" in test_step["run"]
    assert "python -m pip install --no-cache-dir -c /tmp/constraints-cu128.txt --no-build-isolation '/opt/app[dev]' pytest pytest-cov" in prism_step["run"]
