"""Every registered benchmark's protocol, pinned as the plan its config resolves to.

The benchmarks build their configs from :mod:`soma.presets`; the plans below were
captured from the hand-built configs before that change. A preset default that moves a
benchmark's plan (and so its run identity) fails here. Regenerate the fixture only for a
deliberate protocol change: ``python -m tests.test_benchmark_plans``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from soma.benchmark import get_benchmark
from soma.plan import load_cohort, resolve

FIXTURE = Path(__file__).parent / "fixtures" / "benchmark_plans.yaml"

#: ``(case id, benchmark, build_config kwargs, cohort labels, tune is test)``.
CASES: list[tuple[str, str, dict[str, Any], list[Any] | None, bool]] = [
    ("eva/bach@uni2", "eva/bach", {"encoder": "uni2"}, ["a", "b", "c", "d"] * 2, True),
    (
        "eva/bach@virchow2+knobs",
        "eva/bach",
        {
            "encoder": "virchow2",
            "seed": 3,
            "max_steps": 100,
            "patience": 5,
            "encoder_batch_size": 8,
            "overrides": {"cache": {"root_dir": "/cache", "dtype": "fp16"}},
        },
        ["a", "b", "c", "d"] * 2,
        True,
    ),
    ("eva/patch_camelyon@virchow2", "eva/patch_camelyon", {"encoder": "virchow2"}, [0, 1] * 4, False),
    ("eva/camelyon16_small@uni2", "eva/camelyon16_small", {"encoder": "uni2"}, [0, 1] * 4, False),
    (
        "eva/panda_small@virchow2+knobs",
        "eva/panda_small",
        {"encoder": "virchow2", "seed": 2, "epochs": 3, "patience": 2, "overrides": {"cache": {"root_dir": "/cache"}}},
        [0, 1, 2, 3, 4, 5] * 2,
        False,
    ),
    ("eva/consep@uni2", "eva/consep", {"encoder": "uni2"}, None, True),
    (
        "eva/monusac@virchow2+knobs",
        "eva/monusac",
        {"encoder": "virchow2", "seed": 1, "max_steps": 50, "patience": 4, "encoder_batch_size": 4},
        None,
        True,
    ),
    ("hest/IDC@uni2", "hest/IDC", {"encoder": "uni2"}, None, False),
    (
        "hest/LUNG@virchow2+knobs",
        "hest/LUNG",
        {"encoder": "virchow2", "seed": 1, "pca_components": 64, "overrides": {"cache": {"root_dir": "/cache"}}},
        None,
        False,
    ),
    ("croma/camelyon@uni2", "croma/camelyon", {"encoder": "uni2"}, None, False),
    (
        "croma/tcga-4x4@virchow2+knobs",
        "croma/tcga-4x4",
        {"encoder": "virchow2", "encoder_batch_size": 16, "overrides": {"cache": {"root_dir": "/cache"}}},
        None,
        False,
    ),
]


def _cohort_files(root: Path, labels: list[Any] | None, tune_is_test: bool) -> tuple[Path, Path]:
    labels = labels if labels is not None else [0, 1] * 4
    ids = [f"s{index}" for index in range(len(labels))]
    root.mkdir(parents=True, exist_ok=True)
    records, splits = root / "dataset.csv", root / "splits.csv"
    pd.DataFrame({"sample_id": ids, "image_path": [f"/images/{s}.png" for s in ids], "label": labels}).to_csv(
        records, index=False
    )
    held_out = ["tune", "tune"] if tune_is_test else ["tune", "test"]
    pd.DataFrame({"sample_id": ids, "split": ["train"] * (len(ids) - 2) + held_out}).to_csv(splits, index=False)
    return records, splits


def benchmark_plan(benchmark: str, kwargs: dict[str, Any], labels, tune_is_test: bool, root: Path) -> dict[str, Any]:
    """The comparable plan ``benchmark`` builds, with the cohort files' paths left out."""
    records, splits = _cohort_files(root, labels, tune_is_test)
    config = get_benchmark(benchmark).build_config(
        dataset_csv=records, splits_csv=splits, output_root=root / "out", **kwargs
    )
    trains = config.learner is not None and config.learner.name in ("mil", "dense")
    plan = resolve(config, load_cohort(config) if trains else None).comparable()
    plan["cohort"].update(records="<records>", splits="<splits>")
    return plan


@pytest.mark.parametrize(
    ("case", "benchmark", "kwargs", "labels", "tune_is_test"), CASES, ids=[case[0] for case in CASES]
)
def test_a_benchmark_resolves_to_its_pinned_plan(tmp_path: Path, case, benchmark, kwargs, labels, tune_is_test):
    pinned = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert benchmark_plan(benchmark, kwargs, labels, tune_is_test, tmp_path) == pinned[case]


@pytest.mark.parametrize(
    ("benchmark", "preset", "arguments"),
    [
        ("eva/bach", "tile_classification", ("uni2",)),
        ("eva/camelyon16_small", "mil_classification", ("uni2",)),
        ("eva/monusac", "segmentation", ("uni2",)),
        ("hest/IDC", "tile_classification", ("uni2",)),
        ("croma/camelyon", "tile_classification", ("uni2",)),
    ],
)
def test_a_benchmark_builds_its_config_from_a_preset(monkeypatch, benchmark, preset, arguments):
    from soma import presets

    calls = []
    original = getattr(presets, preset)

    def spy(*args, **kwargs):
        calls.append(args[: len(arguments)])
        return original(*args, **kwargs)

    monkeypatch.setattr(presets, preset, spy)
    get_benchmark(benchmark).build_config(encoder="uni2")
    assert calls == [arguments]


def _write_fixture() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plans = {
            case: benchmark_plan(benchmark, kwargs, labels, tune_is_test, Path(directory) / str(index))
            for index, (case, benchmark, kwargs, labels, tune_is_test) in enumerate(CASES)
        }
    FIXTURE.write_text(yaml.safe_dump(plans, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    _write_fixture()
