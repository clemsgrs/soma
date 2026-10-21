"""Tests for soma CLI entrypoint."""

import json
from pathlib import Path
from typing import Literal
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from soma import presets
from soma.config import Component, ExperimentConfig, FeaturesConfig
from soma.plan import check, load_cohort, resolve
from tests.experiment_configs import experiment


def _make_valid_config(tmp_path: Path) -> Path:
    cfg = experiment(
        'data/dataset.csv',
        'data/splits.csv',
        output_dir='runs',
        features=FeaturesConfig(encoder='uni2'),
        aggregator='abmil',
        trainer={"max_epochs": 5},
        scheduler='cosine',
    )
    path = tmp_path / "config.yaml"
    cfg.to_yaml(path)
    return path


def _pipeline_call(args: list[str]):
    """Run main() with Pipeline patched; return the (config, source) it was constructed with."""
    with patch("soma.cli.Pipeline") as MockPipeline:
        MockPipeline.return_value = MagicMock()
        from soma.cli import main

        main(args)
    MockPipeline.return_value.run.assert_called_once()
    return MockPipeline.call_args[0][0], MockPipeline.call_args.kwargs["source"]


def _exit_code(args: list[str], capsys) -> tuple[int, str]:
    from soma.cli import main

    with patch("soma.cli.Pipeline") as MockPipeline, pytest.raises(SystemExit) as exc_info:
        main(args)
    MockPipeline.assert_not_called()
    return exc_info.value.code, capsys.readouterr().err


# --- soma run CONFIG ---


def test_run_takes_a_config_path_and_runs_the_pipeline(tmp_path: Path):
    config, source = _pipeline_call(["run", str(_make_valid_config(tmp_path))])

    assert config.features.encoder.name == "uni2"
    assert config.learner.params["head"].name == "binary_classification"
    assert source is None


def test_a_bare_config_path_is_not_a_command(tmp_path: Path, capsys):
    code, err = _exit_code([str(_make_valid_config(tmp_path))], capsys)
    assert code == 2
    assert "soma run" in err


def test_run_takes_no_overrides(tmp_path: Path, capsys):
    # A config is edited, not overridden: there is no --set.
    code, _ = _exit_code(["run", str(_make_valid_config(tmp_path)), "--set", "trainer.max_epochs=3"], capsys)
    assert code == 2


def test_run_missing_config_file_exits(tmp_path: Path, capsys):
    code, err = _exit_code(["run", str(tmp_path / "nonexistent.yaml")], capsys)
    assert code != 0
    assert "nonexistent.yaml" in err


def test_run_invalid_yaml_exits(tmp_path: Path, capsys):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(": invalid: yaml: [")
    code, _ = _exit_code(["run", str(bad_yaml)], capsys)
    assert code != 0


def test_run_invalid_config_content_exits(tmp_path: Path, capsys):
    # Valid YAML but an unknown section.
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({"cohort": {"records": "data.csv", "splits": "splits.csv"}, "data": {"dataset_type": "case"}})
    )
    code, _ = _exit_code(["run", str(config_path)], capsys)
    assert code != 0


# --- --feature-dir ---


def test_feature_dir_attaches_a_cached_set_source(tmp_path: Path):
    import torch

    from soma.data import CachedSetSource

    config_path = _make_valid_config(tmp_path)
    features = tmp_path / "features"
    features.mkdir()
    torch.save(torch.randn(3, 4), features / "s0.pt")

    _, source = _pipeline_call(["run", str(config_path), "--feature-dir", str(features)])

    assert isinstance(source, CachedSetSource) and source.feature_dir == features


def test_feature_dir_attaches_a_cached_grid_source_for_a_dense_task(tmp_path: Path):
    from soma.extract.dense import CachedGridSource

    config_path = tmp_path / "config.yaml"
    experiment(
        "data/dataset.csv",
        "data/splits.csv",
        decoder="lightweight_conv",
        head="segmentation",
        head_params={"num_classes": 2},
    ).to_yaml(config_path)
    grids = tmp_path / "grids"
    grids.mkdir()

    _, source = _pipeline_call(["run", str(config_path), f"--feature-dir={grids}"])

    assert isinstance(source, CachedGridSource)
    assert source.provenance.to_dict()["kind"] == "dense_cache"


def test_feature_dir_must_be_a_directory(tmp_path: Path, capsys):
    code, err = _exit_code(["run", str(_make_valid_config(tmp_path)), "--feature-dir", str(tmp_path / "missing")], capsys)
    assert code == 1
    assert "--feature-dir" in err


# --- soma run --preset ---


def _preset_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "run",
        "--preset",
        "mil_classification",
        "--cohort",
        str(tmp_path / "dataset.csv"),
        "--splits",
        str(tmp_path / "splits.csv"),
        "--out",
        str(tmp_path / "runs"),
        *extra,
    ]


def test_run_preset_builds_the_config_from_the_preset_and_the_cohort(tmp_path: Path):
    config, source = _pipeline_call(_preset_args(tmp_path, "--encoder", "uni2", "--aggregator", "transmil", "--seed", "3"))

    expected = presets.mil_classification("uni2", aggregator="transmil")
    expected.cohort.records = str(tmp_path / "dataset.csv")
    expected.cohort.splits = str(tmp_path / "splits.csv")
    expected.run.output_dir = str(tmp_path / "runs")
    expected.run.seed = 3
    assert config.to_dict() == expected.to_dict()
    assert source is None


def test_run_preset_takes_the_tiling_as_typed_flags(tmp_path: Path):
    config, _ = _pipeline_call(
        _preset_args(tmp_path, "--encoder", "uni2", "--aggregator", "abmil", "--spacing", "0.5", "--tile-size", "224")
    )
    assert config.features == presets.mil_classification("uni2", "abmil", spacing=0.5, tile_size=224).features


def test_run_preset_takes_the_preset_s_defaults(tmp_path: Path):
    config, _ = _pipeline_call(_preset_args(tmp_path, "--encoder", "uni2", "--aggregator", "abmil"))
    assert config.features.extraction.tiling is None
    assert config.run.seed == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--aggregator", "abmil"), "--encoder"),  # a required preset argument
        (("--encoder", "uni2"), "--aggregator"),  # the aggregator has no default
        (("--encoder", "uni2", "--aggregator", "abmil", "--decoder", "linear"), "--decoder"),  # another preset's argument
    ],
)
def test_run_preset_takes_exactly_its_own_arguments(tmp_path: Path, capsys, extra, message):
    code, err = _exit_code(_preset_args(tmp_path, *extra), capsys)
    assert code == 2
    assert message in err


def test_run_preset_needs_the_cohort_splits_and_output_dir(tmp_path: Path, capsys):
    code, err = _exit_code(["run", "--preset", "tile_classification", "--encoder", "uni2"], capsys)
    assert code == 2
    assert "--cohort" in err and "--splits" in err and "--out" in err


def test_run_takes_a_config_or_a_preset_not_both(tmp_path: Path, capsys):
    config_path = str(_make_valid_config(tmp_path))
    code, _ = _exit_code([*_preset_args(tmp_path, "--encoder", "uni2"), config_path], capsys)
    assert code == 2
    # The preset's flags are not overrides of a config.
    code, err = _exit_code(["run", config_path, "--seed", "3"], capsys)
    assert code == 2 and "--preset" in err
    code, _ = _exit_code(["run"], capsys)
    assert code == 2


def test_run_preset_reports_a_bad_preset_argument(tmp_path: Path, capsys):
    code, err = _exit_code(_preset_args(tmp_path, "--encoder", "not-an-encoder", "--aggregator", "abmil"), capsys)
    assert code == 2
    assert "not-an-encoder" in err


# --- soma init ---


def _init(args: list[str], capsys) -> str:
    from soma.cli import main

    main(["init", *args])
    return capsys.readouterr().out


def test_init_prints_an_editable_yaml_with_every_default(capsys):
    from soma import components

    data = yaml.safe_load(_init(["mil_classification", "--encoder", "uni2", "--aggregator", "abmil"], capsys))

    config = ExperimentConfig.from_dict(data)
    check(config)
    assert data["cohort"]["records"] == "dataset.csv" and data["cohort"]["splits"] == "splits.csv"
    model = data["learner"]["params"]
    abmil = next(entry for entry in components("aggregators") if entry["name"] == "abmil")
    assert model["aggregator"] == {"name": "abmil", "params": abmil["params"]}
    # What the cohort decides is left for the run.
    assert model["head"]["params"]["num_classes"] is None and model["head"]["params"]["label_map"] is None
    assert model["optimizer"]["params"]["lr"] == 1e-4 and model["optimizer"]["params"]["eps"] == 1e-8
    assert data["trainer"]["callbacks"][1]["params"]["patience"] == 10


def test_init_output_does_not_depend_on_the_machine(monkeypatch, capsys):
    """The GPU and tiling-worker counts are left unset: slide2vec resolves them on the
    machine a run starts on, so the printed YAML is the same wherever ``soma init`` runs."""
    import os

    import torch

    for env_name in ("SLURM_CPUS_PER_TASK", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE"):
        monkeypatch.delenv(env_name, raising=False)
    outputs = []
    for cpus, gpus in ((4, 1), (32, 8)):
        monkeypatch.setattr(os, "cpu_count", lambda cpus=cpus: cpus)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "device_count", lambda gpus=gpus: gpus)
        outputs.append(_init(["mil_classification", "--encoder", "uni2", "--aggregator", "abmil"], capsys))

    assert outputs[0] == outputs[1]
    execution = yaml.safe_load(outputs[0])["features"]["execution"]
    assert (execution["num_gpus"], execution["num_preprocessing_workers"]) == (None, None)


def test_init_output_resolves_to_the_preset_s_plan(tmp_path: Path, capsys):
    ids = [f"s{index}" for index in range(8)]
    records, splits = tmp_path / "dataset.csv", tmp_path / "splits.csv"
    pd.DataFrame({"sample_id": ids, "image_path": [f"/x/{s}.tif" for s in ids], "label": ["a", "b"] * 4}).to_csv(records, index=False)
    pd.DataFrame({"sample_id": ids, "split": ["train"] * 5 + ["tune"] + ["test"] * 2}).to_csv(splits, index=False)
    path = tmp_path / "exp.yaml"
    path.write_text(_init(["mil_classification", "--encoder", "uni2", "--aggregator", "transmil"], capsys))

    # The user points the cohort at their files; everything else is the preset's.
    edited = ExperimentConfig.from_yaml(path)
    edited.cohort.records, edited.cohort.splits = str(records), str(splits)
    preset = presets.mil_classification("uni2", aggregator="transmil")
    preset.cohort.records, preset.cohort.splits = str(records), str(splits)

    assert resolve(edited, load_cohort(edited)).comparable() == resolve(preset, load_cohort(preset)).comparable()


@pytest.mark.parametrize(
    "name",
    [
        "tile_classification", "tile_regression", "mil_classification", "mil_regression", "mil_survival",
        "slide_classification", "slide_regression", "slide_survival", "segmentation", "detection",
    ],
)
def test_init_output_of_every_preset_passes_check(capsys, name):
    arguments = {"aggregator": "abmil", "loss": "cox", "decoder": "lightweight_conv", "num_classes": "2"}
    from soma import components

    entry = next(entry for entry in components("presets") if entry["name"] == name)
    flags = ["--encoder", "titan" if name.startswith("slide_") else "uni2"] + [
        token for key in entry["params"] if key in arguments for token in (f"--{key.replace('_', '-')}", arguments[key])
    ]
    check(ExperimentConfig.from_dict(yaml.safe_load(_init([name, *flags], capsys))))


def test_init_takes_the_tiling_as_typed_flags(capsys):
    # The preset annotates them ``float | None`` and ``int | None``.
    out = _init(["mil_classification", "--encoder", "uni2", "--aggregator", "abmil", "--spacing", "0.5"], capsys)
    assert "requested_spacing_um: 0.5\n" in out
    tiling = yaml.safe_load(out)["features"]["extraction"]["tiling"]
    assert (tiling["requested_spacing_um"], tiling["requested_tile_size_px"]) == (0.5, 224)
    out = _init(
        ["mil_survival", "--encoder", "uni2", "--aggregator", "abmil", "--loss", "nll", "--num-bins", "4", "--tile-size", "224"],
        capsys,
    )
    tiling = yaml.safe_load(out)["features"]["extraction"]["tiling"]
    assert (tiling["requested_spacing_um"], tiling["requested_tile_size_px"]) == (0.5, 224)


def test_init_refuses_a_spacing_the_encoder_does_not_support(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "mil_classification", "--encoder", "uni2", "--aggregator", "abmil", "--spacing", "1.0"])
    assert exc_info.value.code == 2
    assert "encoder 'uni2' supports spacing 0.5 µm" in capsys.readouterr().err


def test_init_takes_the_preset_s_flags_before_its_name(capsys):
    data = yaml.safe_load(_init(["--encoder", "uni2", "--aggregator", "transmil", "mil_classification"], capsys))
    assert data["learner"]["params"]["aggregator"]["name"] == "transmil"


def test_init_requires_the_preset_s_arguments(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "segmentation", "--encoder", "uni2"])
    assert exc_info.value.code == 2
    assert "preset segmentation needs --decoder, --num-classes" in capsys.readouterr().err


def test_init_takes_the_survival_loss_from_its_choices(capsys):
    from soma.cli import main

    head = yaml.safe_load(_init(["mil_survival", "--encoder", "uni2", "--aggregator", "abmil", "--loss", "cox"], capsys))[
        "learner"
    ]["params"]["head"]
    assert (head["name"], head["params"]["cox_window"]) == ("cox_survival", 32)
    with pytest.raises(SystemExit) as exc_info:
        main(["init", "mil_survival", "--encoder", "uni2", "--aggregator", "abmil", "--loss", "coxph"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "argument --loss: invalid choice: 'coxph'" in err and "nll" in err and "cox" in err
    with pytest.raises(SystemExit) as exc_info:
        main(["init", "mil_survival", "--help"])
    assert exc_info.value.code == 0
    assert "--loss {nll,cox}" in capsys.readouterr().out


def test_init_refuses_arguments_that_make_an_invalid_config(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "mil_survival", "--encoder", "uni2", "--aggregator", "clam_sb", "--loss", "nll", "--num-bins", "4"])
    assert exc_info.value.code == 2
    assert "not supported for survival" in capsys.readouterr().err


def test_init_takes_the_survival_bin_count_as_a_typed_flag(capsys):
    from soma.cli import main

    args = ["mil_survival", "--encoder", "uni2", "--aggregator", "abmil", "--loss", "nll"]
    head = yaml.safe_load(_init([*args, "--num-bins", "4"], capsys))["learner"]["params"]["head"]
    assert (head["name"], head["params"]["binning"], head["params"]["num_bins"]) == ("discrete_survival", "train_quantiles", 4)
    with pytest.raises(SystemExit) as exc_info:
        main(["init", *args])
    assert exc_info.value.code == 2
    assert "preset mil_survival: loss 'nll' needs num_bins" in capsys.readouterr().err


# --- a preset registered through soma.register ---


def _my_preset(encoder: str, aggregator: str = "abmil") -> ExperimentConfig:
    """A user's preset: the MIL preset under another name."""
    return presets.mil_classification(encoder, aggregator=aggregator)


def _bare_preset() -> ExperimentConfig:
    """A user's preset without arguments."""
    return presets.tile_classification("uni2")


@pytest.fixture()
def user_presets():
    # The public soma.register supplies no ``required``/``description`` metadata; one
    # entry has a params dataclass, the other none.
    from soma import register
    from soma.registry import signature_params

    register("presets", "my_preset", _my_preset, params=signature_params(_my_preset, name="MyPresetParams"))
    register("presets", "bare_preset", _bare_preset)
    try:
        yield
    finally:
        presets.preset_registry._entries.pop("my_preset", None)
        presets.preset_registry._entries.pop("bare_preset", None)


def test_a_publicly_registered_preset_inits_and_runs(tmp_path: Path, capsys, user_presets):
    data = yaml.safe_load(_init(["my_preset", "--encoder", "uni2"], capsys))
    check(ExperimentConfig.from_dict(data))
    assert data["learner"]["params"]["aggregator"]["name"] == "abmil"
    assert yaml.safe_load(_init(["bare_preset"], capsys))["features"]["encoder"]["name"] == "uni2"

    run = ["run", "--preset", "my_preset", "--cohort", "d.csv", "--splits", "s.csv", "--out", str(tmp_path)]
    config, _ = _pipeline_call([*run, "--encoder", "uni2", "--aggregator", "transmil"])
    assert config.learner.params["aggregator"].name == "transmil"
    code, err = _exit_code(run, capsys)  # its required argument missing: a usage error
    assert code == 2 and "--encoder" in err


def test_a_preset_registered_without_params_leaves_run_working(tmp_path: Path, user_presets):
    config, _ = _pipeline_call(["run", str(_make_valid_config(tmp_path))])
    assert config.features.encoder.name == "uni2"


def _seeded_preset(encoder: str, seed: int = 3) -> ExperimentConfig:
    """A user's preset whose argument is named like soma run's own --seed."""
    config = presets.mil_classification(encoder, "abmil")
    config.run.seed = seed
    return config


def _typed_preset(
    encoder: str,
    epochs: int = 50,
    lr: "float" = 1e-4,
    batch_size: Literal[1, 8] = 1,
    workers: Literal["auto", 4] = "auto",
    mode: Literal["auto", None] = "auto",
) -> ExperimentConfig:
    """A user's preset with typed arguments (``lr`` annotated as under postponed annotations)."""
    config = presets.mil_classification(encoder, "abmil")
    config.trainer.max_epochs = epochs
    config.learner.params["optimizer"] = Component("adamw", {"lr": lr})
    config.loaders.batch_size = batch_size
    if workers != "auto":
        config.loaders.num_workers = workers
    return config


def _unevaluable_preset(encoder: "NotAType", epochs: int = 50) -> ExperimentConfig:  # noqa: F821
    """A user's preset with an annotation that does not evaluate."""
    config = presets.mil_classification(encoder, "abmil")
    config.trainer.max_epochs = epochs
    return config


@pytest.fixture()
def typed_presets():
    from soma import register
    from soma.registry import signature_params

    typed = (("seeded", _seeded_preset), ("typed", _typed_preset), ("unevaluable", _unevaluable_preset))
    for name, preset in typed:
        register("presets", name, preset, params=signature_params(preset, name=f"{name.title()}Params"))
    try:
        yield
    finally:
        for name, _ in typed:
            presets.preset_registry._entries.pop(name, None)


def test_a_preset_argument_named_like_a_run_flag_leaves_other_runs_working(tmp_path: Path, capsys, typed_presets):
    config, _ = _pipeline_call(["run", str(_make_valid_config(tmp_path))])
    assert config.features.encoder.name == "uni2"
    assert _exit_code(["run", "--help"], capsys)[0] == 0
    # soma init owns no --seed: there the argument is the preset's flag.
    assert yaml.safe_load(_init(["seeded", "--encoder", "uni2", "--seed", "4"], capsys))["run"]["seed"] == 4
    # soma run does: selecting the preset there is a usage error naming the clash.
    run = ["run", "--preset", "seeded", "--encoder", "uni2", "--cohort", "d.csv", "--splits", "s.csv", "--out", str(tmp_path)]
    code, err = _exit_code(run, capsys)
    assert code == 2 and "preset seeded: argument seed clashes with a soma run flag" in err


def test_preset_arguments_take_their_declared_types(tmp_path: Path, capsys, typed_presets):
    data = yaml.safe_load(_init(["typed", "--encoder", "uni2", "--epochs", "10", "--lr", "0.001"], capsys))
    assert data["trainer"]["max_epochs"] == 10 and data["learner"]["params"]["optimizer"]["params"]["lr"] == 0.001

    run = ["run", "--preset", "typed", "--encoder", "uni2", "--cohort", "d.csv", "--splits", "s.csv", "--out", str(tmp_path)]
    config, _ = _pipeline_call([*run, "--epochs", "10", "--lr", "0.001"])
    assert config.trainer.max_epochs == 10 and config.learner.params["optimizer"].params["lr"] == 0.001
    code, err = _exit_code([*run, "--epochs", "ten"], capsys)
    assert code == 2 and "--epochs" in err
    # A Literal's values are the flag's choices, each matched by its text.
    config, _ = _pipeline_call([*run, "--batch-size", "8"])
    assert config.loaders.batch_size == 8
    config, _ = _pipeline_call([*run, "--workers", "4"])
    assert config.loaders.num_workers == 4
    code, err = _exit_code([*run, "--batch-size", "4"], capsys)
    assert code == 2 and "--batch-size: invalid choice: '4'" in err
    # None among a Literal's values means the flag may be left out, not a choice.
    code, err = _exit_code([*run, "--mode", "None"], capsys)
    assert code == 2 and "--mode: invalid choice: 'None'" in err


def test_a_preset_annotation_that_does_not_evaluate_leaves_the_flags_working(capsys, typed_presets):
    assert yaml.safe_load(_init(["unevaluable", "--encoder", "uni2", "--epochs", "10"], capsys))["trainer"]["max_epochs"] == 10
    data = yaml.safe_load(_init(["mil_classification", "--encoder", "uni2", "--aggregator", "abmil", "--spacing", "0.5"], capsys))
    assert data["features"]["extraction"]["tiling"]["requested_spacing_um"] == 0.5


def _rows(out: str) -> list[list[str]]:
    """The cells of each table row ``soma list`` printed, whatever the terminal width.

    A cell too long for the width wraps onto continuation lines whose name cell is empty
    (the name column never wraps); each one is joined back onto the row it continues.
    """
    rows: list[list[str]] = []
    for line in out.splitlines():
        if not line.startswith("│"):
            continue
        cells = [cell.strip() for cell in line.split("│")[1:-1]]
        if rows and not cells[0]:
            rows[-1] = [" ".join(part for part in pair if part) for pair in zip(rows[-1], cells)]
        else:
            rows.append(cells)
    return rows


def test_list_without_a_kind_prints_the_kinds(capsys):
    from soma import components
    from soma.cli import main

    main(["list"])

    names = [row[0] for row in _rows(capsys.readouterr().out)]
    assert names == components()


def test_list_a_kind_prints_its_components_and_params(capsys):
    from soma.cli import main

    main(["list", "aggregators"])

    out = capsys.readouterr().out
    assert "Aggregators" in out
    rows = {row[0]: row[1] for row in _rows(out)}
    assert {"abmil", "clam_sb", "mean_pool", "transmil"} <= set(rows)
    assert "hidden_dim=128" in rows["abmil"]


def test_list_heads_prints_the_registered_heads(capsys):
    from soma.cli import main

    main(["list", "heads"])

    names = [row[0] for row in _rows(capsys.readouterr().out)]
    assert {"binary_classification", "segmentation", "discrete_survival", "cox_survival"} <= set(names)
    assert "survival" not in names


def test_list_presets_prints_the_registered_presets(capsys):
    from soma.cli import main

    main(["list", "presets"])

    names = [row[0] for row in _rows(capsys.readouterr().out)]
    # Grouped by what the head trains on: tile images, a slide's pooled tiles, a slide
    # encoder's embedding, then the dense presets.
    assert names == [
        "tile_classification", "tile_regression",
        "mil_classification", "mil_regression", "mil_survival",
        "slide_classification", "slide_regression", "slide_survival",
        "segmentation", "detection",
    ]


def test_list_encoders_filters_by_level(capsys):
    from soma import components
    from soma.cli import main

    main(["list", "encoders", "--level", "slide"])

    out = capsys.readouterr().out
    assert "Encoders (slide)" in out
    printed = {row[0] for row in _rows(out)}
    expected = {entry["name"] for entry in components("encoders") if entry["level"] == "slide"}
    assert printed == expected


def test_list_level_applies_to_encoders_only(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["list", "aggregators", "--level", "tile"])

    assert exc_info.value.code == 2
    assert "encoders only" in capsys.readouterr().err


def test_list_help_prints_structured_block(capsys):
    from soma.cli import main

    main(["list", "--help"])

    out = capsys.readouterr().out
    assert "usage: soma list" in out
    assert "commands:" in out
    assert "options:" in out
    assert "examples:" in out


# --- soma evaluate ---


def _zero_padded_cohort(root: Path) -> tuple[Path, Path]:
    """All-numeric, zero-padded sample ids: type inference would read 001 as 1."""
    root.mkdir()
    for sample_id, (x, y) in {"001": (2, 3), "002": (5, 7)}.items():
        (root / f"{sample_id}.csv").write_text(f"x,y,class\n{x},{y},0\n")
    (root / "dataset.csv").write_text(
        "sample_id,value,points_path\n"
        f"001,0.0,{root / '001.csv'}\n002,1.0,{root / '002.csv'}\n"
        f"003,2.0,{root / '001.csv'}\n004,3.0,{root / '002.csv'}\n"
    )
    (root / "splits.csv").write_text("sample_id,split\n001,train\n002,train\n003,test\n004,test\n")
    return root / "dataset.csv", root / "splits.csv"


@pytest.mark.parametrize(("splits", "mae"), [(True, 1.0), (False, 0.5)], ids=["splits", "unsplit"])
def test_evaluate_scores_zero_padded_sample_ids(tmp_path: Path, splits, mae):
    from soma.cli import main

    dataset_csv, splits_csv = _zero_padded_cohort(tmp_path / "cohort")
    # The test split errs by 0 and 2 (MAE 1.0); over every row, by 0, 0, 0, 2 (MAE 0.5).
    predictions = tmp_path / "predictions.parquet"
    pd.DataFrame(
        {"sample_id": ["001", "002", "003", "004"], "predicted_value": [0.0, 1.0, 2.0, 5.0]}
    ).iloc[2 if splits else 0 :].to_parquet(predictions, index=False)
    out = tmp_path / "evaluation.json"
    split_args = ["--splits", str(splits_csv)] if splits else []
    main(
        ["evaluate", str(predictions), "--cohort", str(dataset_csv), *split_args,
         "--metrics", "mae", "--out", str(out)]
    )

    assert json.loads(out.read_text())["metrics"]["mae"] == mae


@pytest.mark.parametrize("splits", [True, False], ids=["splits", "unsplit"])
def test_grid_evaluate_scores_zero_padded_sample_ids(tmp_path: Path, splits):
    from soma.cli import main
    from soma.evaluation import PointPredictions

    dataset_csv, splits_csv = _zero_padded_cohort(tmp_path / "cohort")
    ids = ["003", "004"] if splits else ["001", "002", "003", "004"]
    xy = {"001": (2, 3), "002": (5, 7), "003": (2, 3), "004": (5, 7)}
    frame = pd.DataFrame(
        [(sample_id, *xy[sample_id], 0, 0.9) for sample_id in ids],
        columns=["sample_id", "x", "y", "class", "score"],
    )
    predictions = tmp_path / "points.parquet"
    PointPredictions(frame, classes=["cell"], sample_ids=ids, match_distance_px=2.0).to_parquet(
        predictions
    )
    out = tmp_path / "evaluation.json"
    split_args = ["--splits", str(splits_csv)] if splits else []
    main(
        ["evaluate", str(predictions), "--cohort", str(dataset_csv), *split_args,
         "--metrics", "mean_f1", "--out", str(out)]
    )

    scored = json.loads(out.read_text())
    assert (scored["num_samples"], scored["metrics"]["mean_f1"]) == (len(ids), 1.0)


# --- Help / no-args ---


def test_no_args_exits_with_usage(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([])

    # argparse exits with code 2 (usage error) when no subcommand given
    assert exc_info.value.code != 0


def test_help_prints_structured_block(capsys):
    from soma.cli import main

    main(["--help"])

    out = capsys.readouterr().out
    assert "commands:" in out
    assert "examples:" in out
    assert "soma run /path/to/config.yaml" in out
    assert "soma init" in out
