from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import pandas as pd
import torch
import yaml

from soma.config import (
    PipelineConfig,
    RepresentationConfig,
    TaskConfig,
    config_yaml_dict,
    load_config,
)
from soma.dataset import Dataset
from soma.dataset import Splits
from soma.features import FeatureStore
from soma.output_layout import (
    build_experiment_spec,
    canonical_experiment_payload,
    create_run_metadata,
)
from soma.pipeline import evaluate_representation


def _representation_yaml(tmp_path: Path, **representation_overrides: object) -> Path:
    representation = {
        "kind": "croma",
        "confounder_column": "medical_center",
        "split": "test",
        "evaluation_design": "all",
        "m": 5,
        "alpha": 0.10,
        **representation_overrides,
    }
    path = tmp_path / "representation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_root": str(tmp_path / "runs")},
                "data": {
                    "dataset_csv": str(tmp_path / "dataset.csv"),
                    "splits_csv": str(tmp_path / "splits.csv"),
                    "dataset_type": "tile",
                },
                "task": None,
                "representation": representation,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_representation_yaml_disables_task_default_and_materializes_protocol(tmp_path: Path):
    config = load_config(_representation_yaml(tmp_path))

    assert config.task is None
    assert config.representation == RepresentationConfig(
        kind="croma",
        confounder_column="medical_center",
        split="test",
        evaluation_design="all",
        m=5,
        alpha=0.10,
    )
    persisted = config_yaml_dict(config)
    assert persisted["task"] is None
    assert persisted["representation"] == {
        "kind": "croma",
        "confounder_column": "medical_center",
        "split": "test",
        "evaluation_design": "all",
        "m": 5,
        "alpha": 0.10,
    }


def test_pipeline_config_requires_exactly_one_consumer():
    common = {
        "dataset_csv": "dataset.csv",
        "splits_csv": "splits.csv",
        "output_root": "runs",
        "dataset_type": "tile",
    }
    representation = RepresentationConfig(
        kind="croma", confounder_column="site", split="test"
    )

    with pytest.raises(TypeError, match="exactly one.*task.*representation"):
        PipelineConfig(**common, task=None, representation=None)
    with pytest.raises(TypeError, match="exactly one.*task.*representation"):
        PipelineConfig(
            **common,
            task=TaskConfig(name="binary_classification"),
            representation=representation,
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "other"}, "kind"),
        ({"evaluation_design": "paired_2x2"}, "evaluation_design"),
        ({"m": 3}, "m"),
        ({"alpha": 0.2}, "alpha"),
        ({"confounder_column": ""}, "confounder_column"),
        ({"split": ""}, "split"),
    ],
)
def test_representation_v1_rejects_noncanonical_protocol(
    tmp_path: Path, overrides: dict[str, object], message: str
):
    with pytest.raises(ValueError, match=message):
        load_config(_representation_yaml(tmp_path, **overrides))


@pytest.mark.parametrize(
    "extra",
    [
        {"normalization": {"method": "l2"}},
        {"projection": {"method": "random", "target_dim": 8}},
        {"augmentation": {"horizontal_flip": 0.5}},
    ],
)
def test_representation_rejects_active_feature_transforms(tmp_path: Path, extra: dict):
    path = _representation_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(raw.get(key), dict):
            raw[key].update(value)
        else:
            raw[key] = value
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="representation.*normalization.*projection.*augmentation"):
        load_config(path)


def test_dataset_populates_literal_optional_group_id(tmp_path: Path):
    with_group = tmp_path / "with_group.csv"
    with_group.write_text(
        "sample_id,image_path,label,group_id,site\n"
        "s0,/slides/s0.png,A,patient-7,north\n",
        encoding="utf-8",
    )
    without_group = tmp_path / "without_group.csv"
    without_group.write_text(
        "sample_id,image_path,label\ns0,/slides/s0.png,A\n", encoding="utf-8"
    )

    grouped = Dataset(with_group).samples["s0"]
    ordinary = Dataset(without_group).samples["s0"]

    assert grouped.group_id == "patient-7"
    assert "group_id" not in grouped.metadata
    assert grouped.metadata == {"site": "north"}
    assert ordinary.group_id is None


def _representation_config_with_manifests(
    tmp_path: Path,
    *,
    dataset_rows: str,
    split_rows: str,
    **representation_overrides: object,
) -> PipelineConfig:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dataset.csv").write_text(
        "sample_id,image_path,label,group_id,site\n" + dataset_rows,
        encoding="utf-8",
    )
    (tmp_path / "splits.csv").write_text(
        "fold,sample_id,split\n" + split_rows,
        encoding="utf-8",
    )
    return load_config(_representation_yaml(tmp_path, **representation_overrides))


def test_representation_identity_uses_only_selected_rows_and_protocol(tmp_path: Path):
    base = _representation_config_with_manifests(
        tmp_path / "base",
        dataset_rows=(
            "train,/images/train.png,A,g0,north\n"
            "held,/images/held.png,B,g1,south\n"
        ),
        split_rows="0,train,train\n0,held,test\n",
    )
    changed_selected = _representation_config_with_manifests(
        tmp_path / "selected",
        dataset_rows=(
            "train,/images/train.png,A,g0,north\n"
            "held,/images/other.png,B,g1,south\n"
        ),
        split_rows="0,train,train\n0,held,test\n",
    )
    changed_unselected = _representation_config_with_manifests(
        tmp_path / "unselected",
        dataset_rows=(
            "train,/images/other.png,A,g0,north\n"
            "held,/images/held.png,B,g1,south\n"
        ),
        split_rows="0,train,train\n0,held,test\n",
    )
    changed_protocol = _representation_config_with_manifests(
        tmp_path / "protocol",
        dataset_rows=(
            "train,/images/train.png,A,g0,north\n"
            "held,/images/held.png,B,g1,south\n"
        ),
        split_rows="0,train,train\n0,held,test\n",
        confounder_column="site_2",
    )

    base_id = build_experiment_spec(base).experiment_id
    assert build_experiment_spec(changed_selected).experiment_id != base_id
    assert build_experiment_spec(changed_unselected).experiment_id == base_id
    assert build_experiment_spec(changed_protocol).experiment_id != base_id
    payload = canonical_experiment_payload(base)
    assert payload["task"] is None
    assert payload["representation"] == {
        "kind": "croma",
        "confounder_column": "medical_center",
        "split": "test",
        "evaluation_design": "all",
        "m": 5,
        "alpha": 0.10,
    }
    assert "training" not in payload
    assert "evaluation" not in payload


def test_representation_slug_and_metadata_use_tagged_comparison_key(tmp_path: Path):
    config = _representation_config_with_manifests(
        tmp_path,
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    experiment = build_experiment_spec(config)
    metadata = create_run_metadata(
        config=config,
        experiment=experiment,
        run_dir=tmp_path / "run",
        run_id="attempt-1",
        status="running",
    )

    assert "croma" in experiment.slug
    assert metadata.comparison_key == {
        "kind": "representation",
        "dataset_checksum": experiment.dataset_checksum,
        "splits_checksum": experiment.splits_checksum,
        "representation": {
            "kind": "croma",
            "confounder_column": "medical_center",
            "split": "test",
            "evaluation_design": "all",
            "m": 5,
            "alpha": 0.10,
        },
    }


def _representation_inputs(
    tmp_path: Path,
    *,
    rows: list[tuple[str, str, str, str]],
    split_rows: list[tuple[int, str, str]],
    tensors: dict[str, torch.Tensor] | None = None,
):
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {
                "sample_id": sample_id,
                "image_path": f"/images/{sample_id}.png",
                "label": label,
                "group_id": group_id,
                "site": site,
            }
            for sample_id, label, group_id, site in rows
        ]
    ).to_csv(dataset_csv, index=False)
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(split_rows, columns=["fold", "sample_id", "split"]).to_csv(
        splits_csv, index=False
    )
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for index, (sample_id, *_rest) in enumerate(rows):
        tensor = (tensors or {}).get(
            sample_id, torch.tensor([float(index), float(index + 10)])
        )
        torch.save(tensor, feature_dir / f"{sample_id}.pt")
    dataset = Dataset(dataset_csv)
    return dataset, Splits(splits_csv, dataset), FeatureStore(feature_dir)


def test_representation_forwards_manifest_order_and_maps_croma_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset, splits, store = _representation_inputs(
        tmp_path,
        rows=[
            ("z", "A", "g0", "north"),
            ("a", "B", "g1", "south"),
            ("m", "A", "g2", "north"),
        ],
        split_rows=[(0, "m", "test"), (0, "z", "test"), (0, "a", "train")],
    )
    captured: dict[str, object] = {}

    def fake_compute(features, manifest, **kwargs):
        captured.update(features=features.copy(), manifest=manifest.copy(), kwargs=kwargs)
        return SimpleNamespace(
            value=0.61,
            f0=0.25,
            ltm_alpha=0.42,
            undefined_frac=0.0,
            sample_values_aligned=np.array([0.5, 0.7]),
        )

    monkeypatch.setattr("croma.CRoMa.compute", staticmethod(fake_compute))
    result = evaluate_representation(
        feature_store=store,
        dataset=dataset,
        splits=splits,
        representation=RepresentationConfig(
            kind="croma", confounder_column="site", split="test"
        ),
        run_dir=tmp_path / "run",
    )

    assert result.fold_results == []
    assert result.summary == {
        "test/croma_median": 0.61,
        "test/croma_f0": 0.25,
        "test/croma_ltm10": 0.42,
    }
    np.testing.assert_array_equal(captured["features"], [[0.0, 10.0], [2.0, 12.0]])
    manifest = captured["manifest"]
    assert isinstance(manifest, pd.DataFrame)
    assert manifest["sample_id"].tolist() == ["z", "m"]
    assert manifest["group_id"].tolist() == ["g0", "g2"]
    assert manifest["site"].tolist() == ["north", "north"]
    assert captured["kwargs"] == {
        "confounder_column": "site",
        "evaluation_design": "all",
        "m": 5,
        "alpha": 0.10,
    }
    assert yaml.safe_load((tmp_path / "run" / "summary.json").read_text()) == result.summary
    samples = pd.read_csv(tmp_path / "run" / "croma_samples.csv")
    assert samples.to_dict("records") == [
        {"sample_id": "z", "croma": 0.5},
        {"sample_id": "m", "croma": 0.7},
    ]


@pytest.mark.parametrize(
    ("result_overrides", "message"),
    [
        ({"undefined_frac": 0.1}, "undefined"),
        ({"value": float("nan")}, "non-finite"),
        ({"f0": float("inf")}, "non-finite"),
        ({"ltm_alpha": float("-inf")}, "non-finite"),
    ],
)
def test_representation_rejects_partial_or_nonfinite_croma_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_overrides: dict[str, float],
    message: str,
):
    dataset, splits, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
    )
    values = {"value": 0.6, "f0": 0.2, "ltm_alpha": 0.4, "undefined_frac": 0.0}
    values.update(result_overrides)
    monkeypatch.setattr(
        "croma.CRoMa.compute",
        lambda *_args, **_kwargs: SimpleNamespace(
            **values, sample_values_aligned=np.array([0.6])
        ),
    )

    with pytest.raises(ValueError, match=message):
        evaluate_representation(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            representation=RepresentationConfig(
                kind="croma", confounder_column="site", split="test"
            ),
            run_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run" / "summary.json").exists()


@pytest.mark.parametrize(
    ("rows", "tensors", "message"),
    [
        ([("s0", "", "g0", "north")], None, "label"),
        ([("s0", "A", "", "north")], None, "group_id"),
        ([("s0", "A", "g0", "")], None, "site"),
        ([("s0", "A", "g0", "north")], {"s0": torch.ones(2, 2)}, "rank-1"),
    ],
)
def test_representation_rejects_invalid_selected_rows_or_features(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows,
    tensors,
    message: str,
):
    dataset, splits, store = _representation_inputs(
        tmp_path,
        rows=rows,
        split_rows=[(0, "s0", "test")],
        tensors=tensors,
    )
    monkeypatch.setattr("croma.CRoMa.compute", lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match=message):
        evaluate_representation(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            representation=RepresentationConfig(
                kind="croma", confounder_column="site", split="test"
            ),
            run_dir=tmp_path / "run",
        )


def test_representation_split_must_be_nonempty_and_in_exactly_one_fold(tmp_path: Path):
    dataset, splits, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north"), ("s1", "B", "g1", "south")],
        split_rows=[(0, "s0", "test"), (1, "s1", "test")],
    )

    with pytest.raises(ValueError, match="exactly one fold"):
        evaluate_representation(
            feature_store=store,
            dataset=dataset,
            splits=splits,
            representation=RepresentationConfig(
                kind="croma", confounder_column="site", split="test"
            ),
            run_dir=tmp_path / "run",
        )


def test_pipeline_representation_run_writes_normal_artifacts_without_task_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    dataset, _splits, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
    )
    del dataset, store
    config = PipelineConfig(
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        output_root=tmp_path / "outputs",
        dataset_type="tile",
        task=None,
        representation=RepresentationConfig(
            kind="croma", confounder_column="site", split="test"
        ),
    )
    monkeypatch.setattr(
        "croma.CRoMa.compute",
        lambda *_args, **_kwargs: SimpleNamespace(
            value=0.6,
            f0=0.2,
            ltm_alpha=0.4,
            undefined_frac=0.0,
            sample_values_aligned=np.array([0.6]),
        ),
    )

    from soma.pipeline import Pipeline

    result = Pipeline(config, feature_dir=tmp_path / "features").run()

    assert result.fold_results == []
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "summary.json").is_file()
    assert (result.run_dir / "run.log").is_file()
    assert not (result.run_dir / "report.html").exists()
    metadata = yaml.safe_load((result.run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["finished_at"]
    assert metadata["seed"] == 0
    assert metadata["representation_provenance"] == {"croma": "0.3.0"}
