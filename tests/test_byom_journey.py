"""Installed Encoder plugin → soma CLI → measured Leaderboard smoke (#335)."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _run(module: str, *args: str, cwd: Path, env: dict[str, str]):
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )


def _assert_ordinary_name_identity(output_root: Path) -> None:
    experiment_specs = [
        json.loads(path.read_text()) for path in output_root.glob("**/experiment.json")
    ]
    names = {spec["canonical_spec"]["encoder"]["name"] for spec in experiment_specs}
    assert names == {"private-lab-fixture", "dinov2-vitb14"}
    assert len({spec["experiment_id"] for spec in experiment_specs}) == 2
    assert all(
        spec["experiment_id"] in spec["experiment_dirname"]
        or spec["short_hash"] in spec["experiment_dirname"]
        for spec in experiment_specs
    )
    assert all(
        set(spec["canonical_spec"]["encoder"])
        == {
            "name",
            "precision",
            "batch_size",
            "adaptive_batching",
            "output_variant",
            "allow_non_recommended_settings",
            "save_tile_features",
        }
        for spec in experiment_specs
    )

    metadata_paths = list(
        (output_root / "feature_cache").glob("**/cache_metadata.json")
    )
    cache_metadata = [json.loads(path.read_text()) for path in metadata_paths]
    assert {item["encoder_name"] for item in cache_metadata} == names
    assert all(path.parent.name == item["cache_key"] for path, item in zip(metadata_paths, cache_metadata))
    assert all(
        not ({"provider", "distribution", "plugin", "checkpoint"} & set(item))
        for item in cache_metadata
    )


def test_installed_encoder_plugin_completes_byom_benchmark_journey(tmp_path: Path):
    fixture = Path(__file__).parent / "fixtures" / "byom_plugin"
    plugin = tmp_path / "plugin"
    shutil.copytree(fixture, plugin)
    target = tmp_path / "site-packages"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--no-deps",
            "--no-build-isolation",
            "--target",
            str(target),
            str(plugin),
        ],
        check=True,
    )

    repository_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        path
        for path in (str(target), str(repository_root), env.get("PYTHONPATH", ""))
        if path
    )
    env["BYOM_WORKER_SENTINEL"] = str(tmp_path / "workers.log")
    env["BYOM_PUBLIC_FACTORY_SENTINEL"] = str(tmp_path / "public-factory.log")
    env["BYOM_PUBLIC_CLASS_REPORT"] = str(tmp_path / "public-class.log")

    listed = _run("byom_smoke.cli", "list", "encoders", cwd=tmp_path, env=env)
    assert "private-lab-fixture" in listed.stdout
    assert "dinov2-vitb14" in listed.stdout

    worker_env = env.copy()
    worker_env["BYOM_WORKER_REQUEST"] = json.dumps(
        {
            "name": "private-lab-fixture",
            "output_variant": None,
            "allow_non_recommended_settings": False,
        }
    )
    _run("byom_smoke.worker", cwd=tmp_path, env=worker_env)

    output_root = tmp_path / "runs"
    reproduced = _run(
            "byom_smoke.cli",
            "reproduce",
            "byom-smoke",
            "--encoders",
            "private-lab-fixture",
            "dinov2-vitb14",
            "--raw-root",
            str(tmp_path / "raw"),
            "--output-root",
            str(output_root),
            "--seeds",
            "1",
        cwd=tmp_path,
        env=env,
    )

    assert (tmp_path / "workers.log").read_text().splitlines() == [
        "private-lab-fixture:3",
    ]
    assert "[MEASURED] byom-smoke test/accuracy" in reproduced.stdout
    assert "[REFERENCE SKIPPED] byom-smoke test/accuracy" in reproduced.stdout
    assert "[REFERENCE OK] byom-smoke test/accuracy" in reproduced.stdout
    assert "= 0.1230" not in reproduced.stdout
    assert (tmp_path / "public-class.log").read_text() == (
        "slide2vec.encoders.models.dinov2:DINOv2ViTB14\n"
    )
    assert (tmp_path / "public-factory.log").read_text() == (
        "vit_base_patch14_dinov2.lvd142m:pretrained=True\n"
    )

    _assert_ordinary_name_identity(output_root)

    leaderboard = json.loads(
        (output_root / "leaderboards" / "byom-smoke.json").read_text()
    )
    assert leaderboard["vary"] == ["encoder"]
    rows = {row["vary"]["encoder"]: row for row in leaderboard["rows"]}
    assert set(rows) == {"private-lab-fixture", "dinov2-vitb14"}
    assert rows["private-lab-fixture"]["reference"] is None
    assert rows["dinov2-vitb14"]["reference"]["expected"] == 0.123
    assert rows["dinov2-vitb14"]["n"] == 1
    assert rows["dinov2-vitb14"]["mean"] != 0.123
