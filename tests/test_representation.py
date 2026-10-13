from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from soma.config import CRoMaOptions, EncoderLeaf, ExperimentConfig, FeaturesConfig
from soma.data import Cohort, ImageManifest
from soma.data import CachedSetSource
from soma.pipeline.layout import (
    build_experiment_spec,
    canonical_experiment_payload,
    create_run_metadata,
)
from soma.pipeline.representation import evaluate_representation
from soma.plan import Plan, check
from tests.experiment_configs import experiment

_PROTOCOL = {
    "confounder_column": "medical_center",
    "split": "test",
    "evaluation_design": "all",
    "m": 5,
    "alpha": 0.10,
}


def _representation_yaml(tmp_path: Path, **representation_overrides: object) -> Path:
    path = tmp_path / "representation.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "run": {"output_dir": str(tmp_path / "runs")},
                "cohort": {
                    "records": str(tmp_path / "dataset.csv"),
                    "splits": str(tmp_path / "splits.csv"),
                },
                "evaluation": {"representation": {**_PROTOCOL, **representation_overrides}},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_representation_yaml_is_a_task_free_evaluation(tmp_path: Path):
    config = ExperimentConfig.from_yaml(_representation_yaml(tmp_path))
    check(config)

    assert config.learner is None
    assert config.evaluation.representation == CRoMaOptions(
        confounder_column="medical_center",
        split="test",
        evaluation_design="all",
        m=5,
        alpha=0.10,
    )
    persisted = config.to_dict()
    assert persisted["learner"] is None
    assert persisted["evaluation"]["representation"] == _PROTOCOL


def test_a_run_has_exactly_one_consumer():
    representation = CRoMaOptions(confounder_column="site", split="test")

    with pytest.raises(ValueError, match="needs a learner.*or evaluation.representation"):
        check(experiment(head=None))
    with pytest.raises(ValueError, match="task-free evaluation: drop the learner"):
        check(experiment(evaluation={"representation": representation}))


def test_representation_requires_tile_embeddings():
    with pytest.raises(ValueError, match="tile-level encoder"):
        check(
            experiment(
                head=None,
                features=FeaturesConfig(encoder="titan"),
                evaluation={"representation": CRoMaOptions(confounder_column="site", split="test")},
            )
        )


def test_representation_requires_canonical_seed_zero(tmp_path: Path):
    path = _representation_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw["run"]["seed"] = 7
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match="representation.*seed.*0"):
        check(ExperimentConfig.from_yaml(path))


def test_ordinary_task_identity_matches_semantic_manifest_contract(tmp_path: Path):
    (tmp_path / "dataset.csv").write_text(
        "sample_id,image_path,label\n"
        "train,/images/train.png,A\n"
        "held,/images/held.png,B\n",
        encoding="utf-8",
    )
    (tmp_path / "splits.csv").write_text(
        "fold,sample_id,split\n0,train,train\n0,held,test\n",
        encoding="utf-8",
    )
    config = experiment(tmp_path / "dataset.csv", tmp_path / "splits.csv", output_dir=tmp_path / "runs")

    payload = canonical_experiment_payload(Plan(config))
    spec = build_experiment_spec(Plan(config))

    assert payload["learner"]["params"]["head"] == {"name": "binary_classification", "params": {}}
    assert payload["evaluation"]["representation"] is None
    assert spec.experiment_id == (
        "1e71145b4fae9ddabebd4a3c3898e4948abfcb6a39d5a4ed0675249a0cc9493c"
    )
    assert spec.slug == (
        "dataset-precomputed-slide-binary-classification_1e71145b4fae"
    )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "croma"}, "kind"),
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
        ExperimentConfig.from_yaml(_representation_yaml(tmp_path, **overrides))


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        ({"learner": {"name": "mil", "params": {"normalization": {"method": "l2"}}}}, "drop the learner"),
        ({"learner": {"name": "mil", "params": {"projection": {"method": "random", "target_dim": 8}}}}, "drop the learner"),
        ({"loaders": {"augment": {"name": "segmentation", "params": {"horizontal_flip": 0.5}}}}, "no augmentation"),
    ],
)
def test_representation_rejects_active_feature_transforms(tmp_path: Path, extra: dict, message: str):
    path = _representation_yaml(tmp_path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw.update(extra)
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        check(ExperimentConfig.from_yaml(path))


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

    def record(csv):
        return Cohort.from_frames(pd.read_csv(csv).assign(split="test"), targets=["label"]).record("s0")

    grouped, ordinary = record(with_group), record(without_group)

    # group_id is plain metadata now: the representation protocol reads it from there.
    assert grouped.metadata == {"group_id": "patient-7", "site": "north"}
    assert "group_id" not in ordinary.metadata


def _representation_config_with_manifests(
    tmp_path: Path,
    *,
    dataset_rows: str,
    split_rows: str,
    **representation_overrides: object,
) -> Plan:
    """A representation run's plan over the given cohort files."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "dataset.csv").write_text(
        "sample_id,image_path,label,group_id,site\n" + dataset_rows,
        encoding="utf-8",
    )
    (tmp_path / "splits.csv").write_text(
        "fold,sample_id,split\n" + split_rows,
        encoding="utf-8",
    )
    return Plan(ExperimentConfig.from_yaml(_representation_yaml(tmp_path, **representation_overrides)))


def test_representation_identity_uses_only_selected_rows_and_protocol(tmp_path: Path):
    base = _representation_config_with_manifests(
        tmp_path / "base",
        dataset_rows=(
            "train,/images/train.png,A,g0,north\n"
            "held,/images/held.png,B,g1,south\n"
        ),
        split_rows="0,train,train\n0,held,test\n",
    )
    relocated_selected = _representation_config_with_manifests(
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
    assert build_experiment_spec(relocated_selected).experiment_id == base_id
    assert build_experiment_spec(changed_unselected).experiment_id == base_id
    assert build_experiment_spec(changed_protocol).experiment_id != base_id
    payload = canonical_experiment_payload(base)
    assert payload["representation"] == _PROTOCOL
    for section in ("learner", "trainer", "loaders", "evaluation"):
        assert section not in payload


@pytest.mark.parametrize(
    "selected_row",
    [
        "held,/images/held.png,C,g1,south\n",
        "held,/images/held.png,B,other-group,south\n",
        "held,/images/held.png,B,g1,north\n",
    ],
)
def test_representation_identity_covers_selected_evaluation_values(
    tmp_path: Path, selected_row: str
):
    base = _representation_config_with_manifests(
        tmp_path / "base",
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    changed = _representation_config_with_manifests(
        tmp_path / "changed",
        dataset_rows=selected_row,
        split_rows="0,held,test\n",
    )

    assert build_experiment_spec(changed).experiment_id != build_experiment_spec(
        base
    ).experiment_id


def test_representation_identity_covers_encoder_output_variant(tmp_path: Path):
    base = _representation_config_with_manifests(
        tmp_path,
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    def with_variant(variant: str) -> Plan:
        encoder = EncoderLeaf(name="h0-mini", params={"output_variant": variant})
        return Plan(replace(base.config, features=replace(base.features, encoder=encoder)))

    cls = with_variant("cls")
    patch_mean = with_variant("cls_patch_mean")

    assert build_experiment_spec(cls).experiment_id != build_experiment_spec(
        patch_mean
    ).experiment_id


def test_representation_slug_and_metadata_use_tagged_comparison_key(tmp_path: Path):
    config = _representation_config_with_manifests(
        tmp_path,
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    experiment = build_experiment_spec(config)
    metadata = create_run_metadata(
        plan=config,
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
        "representation": _PROTOCOL,
    }


def test_representation_provenance_does_not_silence_version_lookup_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config = _representation_config_with_manifests(
        tmp_path,
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    experiment = build_experiment_spec(config)

    def fail_version_lookup(_distribution: str) -> str:
        raise RuntimeError("distribution metadata is unreadable")

    monkeypatch.setattr("importlib.metadata.version", fail_version_lookup)

    with pytest.raises(RuntimeError, match="distribution metadata is unreadable"):
        create_run_metadata(
            plan=config,
            experiment=experiment,
            run_dir=tmp_path / "run",
            run_id="attempt-1",
            status="running",
        )


def _representation_inputs(
    tmp_path: Path,
    *,
    rows: list[tuple[str, str, str, str]],
    split_rows: list[tuple[int, str, str]],
    tensors: dict[str, torch.Tensor] | None = None,
    missing_feature_ids: set[str] | None = None,
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
        if sample_id in (missing_feature_ids or set()):
            continue
        tensor = (tensors or {}).get(
            sample_id, torch.tensor([float(index), float(index + 10)])
        )
        torch.save(tensor, feature_dir / f"{sample_id}.pt")
    cohort = Cohort.from_csv(dataset_csv, splits_csv, targets=["label"])
    return cohort, ImageManifest.from_csv(dataset_csv), CachedSetSource(feature_dir)


def _croma_cohort(
    sites: tuple[str, ...] = ("north", "south"), per_cell: int = 6
) -> list[tuple[str, str, str, str]]:
    """Rows with ``per_cell`` samples for every label and site, one group per sample."""
    return [
        (f"{site}-{label}-{index}", label, f"{site}-{label}-{index}", site)
        for site in sites
        for label in ("A", "B")
        for index in range(per_cell)
    ]


def _croma_tensors(rows: list[tuple[str, str, str, str]]) -> dict[str, torch.Tensor]:
    """Orthogonal label/site cells give equal SO/OS distances and zero margins."""
    vectors = {
        ("north", "A"): torch.tensor([1.0, 0.0, 0.0, 0.0]),
        ("north", "B"): torch.tensor([0.0, 1.0, 0.0, 0.0]),
        ("south", "A"): torch.tensor([0.0, 0.0, 1.0, 0.0]),
        ("south", "B"): torch.tensor([0.0, 0.0, 0.0, 1.0]),
    }
    return {
        sample_id: vectors[(site, label)] for sample_id, label, _group_id, site in rows
    }


def test_representation_forwards_manifest_order_and_maps_croma_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cohort, images, store = _representation_inputs(
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
            sample_values_aligned=np.array([0.5, 0.7]),
        )

    monkeypatch.setattr("croma.CRoMa.compute", staticmethod(fake_compute))
    result = evaluate_representation(
        feature_store=store,
        cohort=cohort,
        images=images,
        representation=CRoMaOptions(confounder_column="site", split="test"),
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


def test_representation_runs_the_installed_croma(tmp_path: Path):
    rows = _croma_cohort()
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=rows,
        split_rows=[(0, row[0], "test") for row in rows],
        tensors=_croma_tensors(rows),
    )

    result = evaluate_representation(
        feature_store=store,
        cohort=cohort,
        images=images,
        representation=CRoMaOptions(confounder_column="site", split="test"),
        run_dir=tmp_path / "run",
    )

    assert result.summary == {
        "test/croma_median": 0.0,
        "test/croma_f0": 1.0,
        "test/croma_ltm10": 0.0,
    }
    samples = pd.read_csv(tmp_path / "run" / "croma_samples.csv")
    assert samples.to_dict("records") == [
        {"sample_id": row[0], "croma": 0.0} for row in rows
    ]


@pytest.mark.parametrize(
    ("result_overrides", "message"),
    [
        ({"value": float("nan")}, "non-finite"),
        ({"f0": float("inf")}, "non-finite"),
        ({"ltm_alpha": float("-inf")}, "non-finite"),
    ],
)
def test_representation_rejects_nonfinite_croma_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    result_overrides: dict[str, float],
    message: str,
):
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
    )
    values = {"value": 0.6, "f0": 0.2, "ltm_alpha": 0.4}
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
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test"),
            run_dir=tmp_path / "run",
        )
    assert not (tmp_path / "run" / "summary.json").exists()


def test_invalid_representation_rerun_removes_stale_summary(tmp_path: Path):
    # A single site leaves no other-confounder neighbours, so CRoMa cannot score any sample.
    rows = _croma_cohort(sites=("north",))
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=rows,
        split_rows=[(0, row[0], "test") for row in rows],
        tensors=_croma_tensors(rows),
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(
        '{"test/croma_median": 0.9}\n', encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="scoreable"):
        evaluate_representation(
            feature_store=store,
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test"),
            run_dir=run_dir,
        )

    assert not (run_dir / "summary.json").exists()


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
    monkeypatch.setattr("croma.CRoMa.compute", lambda *_args, **_kwargs: None)

    # A blank label is now rejected by Dataset itself; the other rows reach the
    # representation evaluator, which owns the group/confounder/rank checks.
    with pytest.raises(ValueError, match=message):
        cohort, images, store = _representation_inputs(
            tmp_path,
            rows=rows,
            split_rows=[(0, "s0", "test")],
            tensors=tensors,
        )
        evaluate_representation(
            feature_store=store,
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test"),
            run_dir=tmp_path / "run",
        )


def test_representation_split_must_be_nonempty_and_in_exactly_one_fold(tmp_path: Path):
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north"), ("s1", "B", "g1", "south")],
        split_rows=[(0, "s0", "test"), (1, "s1", "test")],
    )

    with pytest.raises(ValueError, match="exactly one fold"):
        evaluate_representation(
            feature_store=store,
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test"),
            run_dir=tmp_path / "run",
        )


def test_representation_split_must_exist(tmp_path: Path):
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
    )

    with pytest.raises(ValueError, match="non-empty cohort in exactly one fold"):
        evaluate_representation(
            feature_store=store,
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test_external"),
            run_dir=tmp_path / "run",
        )


def test_representation_rejects_duplicate_split_membership(tmp_path: Path):
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text(
        "sample_id,image_path,label,group_id,site\n"
        "s0,/images/s0.png,A,g0,north\n",
        encoding="utf-8",
    )
    splits_csv = tmp_path / "splits.csv"
    splits_csv.write_text(
        "fold,sample_id,split\n0,s0,test\n0,s0,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate sample_id.*s0"):
        Cohort.from_csv(dataset_csv, splits_csv, targets=["label"])


def test_representation_rejects_missing_selected_feature(tmp_path: Path):
    cohort, images, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
        missing_feature_ids={"s0"},
    )

    with pytest.raises(ValueError, match="Missing features.*s0"):
        evaluate_representation(
            feature_store=store,
            cohort=cohort,
            images=images,
            representation=CRoMaOptions(confounder_column="site", split="test"),
            run_dir=tmp_path / "run",
        )


def test_representation_attempt_metadata_retains_attempt_and_completion_time(
    tmp_path: Path,
):
    config = _representation_config_with_manifests(
        tmp_path,
        dataset_rows="held,/images/held.png,B,g1,south\n",
        split_rows="0,held,test\n",
    )
    experiment = build_experiment_spec(config)

    first = create_run_metadata(
        plan=config,
        experiment=experiment,
        run_dir=tmp_path / "attempt-1",
        run_id="attempt-1",
        status="completed",
        finished_at="2026-08-11T12:00:00+00:00",
    )
    second = create_run_metadata(
        plan=config,
        experiment=experiment,
        run_dir=tmp_path / "attempt-2",
        run_id="attempt-2",
        status="completed",
        finished_at="2026-08-11T12:01:00+00:00",
    )

    assert (first.run_id, first.finished_at) == (
        "attempt-1",
        "2026-08-11T12:00:00+00:00",
    )
    assert (second.run_id, second.finished_at) == (
        "attempt-2",
        "2026-08-11T12:01:00+00:00",
    )


def test_pipeline_representation_run_writes_normal_artifacts_without_task_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FrozenTimestamp:
        def astimezone(self) -> FrozenTimestamp:
            return self

        def isoformat(self) -> str:
            return "2026-08-11T12:34:56+00:00"

        def __format__(self, spec: str) -> str:  # the run id's timestamp
            return "2026-08-11_12-34-56"

    class FrozenDateTime:
        @classmethod
        def now(cls) -> FrozenTimestamp:
            return FrozenTimestamp()

    monkeypatch.setattr("soma.pipeline.layout.datetime", FrozenDateTime)
    monkeypatch.setattr("importlib.metadata.version", lambda name: "9.8.7")
    cohort, _images, store = _representation_inputs(
        tmp_path,
        rows=[("s0", "A", "g0", "north")],
        split_rows=[(0, "s0", "test")],
    )
    del cohort, store
    config = experiment(
        tmp_path / 'dataset.csv',
        tmp_path / 'splits.csv',
        output_dir=tmp_path / 'outputs',
        head=None,
        evaluation={"representation": CRoMaOptions(confounder_column='site', split='test')},
    )
    monkeypatch.setattr(
        "croma.CRoMa.compute",
        lambda *_args, **_kwargs: SimpleNamespace(
            value=0.6,
            f0=0.2,
            ltm_alpha=0.4,
            sample_values_aligned=np.array([0.6]),
        ),
    )

    from soma.pipeline import Pipeline

    result = Pipeline(config, source=CachedSetSource(tmp_path / "features")).run()

    assert result.fold_results == []
    assert (result.run_dir / "config.yaml").is_file()
    assert (result.run_dir / "plan.yaml").is_file()
    assert (result.run_dir / "summary.json").is_file()
    assert (result.run_dir / "run.log").is_file()
    assert not (result.run_dir / "report.html").exists()
    metadata = yaml.safe_load((result.run_dir / "run.yaml").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["finished_at"] == "2026-08-11T12:34:56+00:00"
    assert metadata["seed"] == 0
    assert metadata["representation_provenance"] == {"croma": "9.8.7"}
