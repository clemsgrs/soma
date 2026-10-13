"""Tests for soma CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import yaml

from soma import presets
from soma.config import Component, ExperimentConfig, FeaturesConfig
from soma.plan import MACHINE_RESOLVED_EXECUTION_FIELDS, check, load_cohort, resolve
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


def test_run_preset_takes_the_preset_s_defaults(tmp_path: Path):
    config, _ = _pipeline_call(_preset_args(tmp_path, "--encoder", "uni2"))
    assert config.learner.params["aggregator"].name == "abmil"
    assert config.run.seed == 0


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--aggregator", "abmil"), "--encoder"),  # a required preset argument
        (("--encoder", "uni2", "--decoder", "linear"), "--decoder"),  # another preset's argument
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
    code, err = _exit_code(_preset_args(tmp_path, "--encoder", "not-an-encoder"), capsys)
    assert code == 2
    assert "not-an-encoder" in err


# --- soma init ---


def _init(args: list[str], capsys) -> str:
    from soma.cli import main

    main(["init", *args])
    return capsys.readouterr().out


def test_init_prints_an_editable_yaml_with_every_default(capsys):
    from soma import components

    data = yaml.safe_load(_init(["mil_classification", "--encoder", "uni2"], capsys))

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
    # slide2vec resolves these for the machine a run starts on.
    assert all(data["features"]["execution"][name] is None for name in MACHINE_RESOLVED_EXECUTION_FIELDS)


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


@pytest.mark.parametrize("name", ["mil_classification", "tile_classification", "survival", "segmentation", "detection"])
def test_init_output_of_every_preset_passes_check(capsys, name):
    arguments = {"aggregator": "abmil", "decoder": "lightweight_conv"}
    from soma import components

    entry = next(entry for entry in components("presets") if entry["name"] == name)
    flags = ["--encoder", "uni2"] + [
        token for key in entry["params"] if key in arguments for token in (f"--{key}", arguments[key])
    ]
    check(ExperimentConfig.from_dict(yaml.safe_load(_init([name, *flags], capsys))))


def test_init_takes_the_preset_s_flags_before_its_name(capsys):
    data = yaml.safe_load(_init(["--encoder", "uni2", "--aggregator", "transmil", "mil_classification"], capsys))
    assert data["learner"]["params"]["aggregator"]["name"] == "transmil"


def test_init_requires_the_preset_s_arguments(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "segmentation", "--encoder", "uni2"])
    assert exc_info.value.code == 2
    assert "--decoder" in capsys.readouterr().err


def test_init_refuses_arguments_that_make_an_invalid_config(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["init", "survival", "--encoder", "uni2", "--aggregator", "clam_sb"])
    assert exc_info.value.code == 2
    assert "not supported for survival" in capsys.readouterr().err


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
    config = presets.mil_classification(encoder)
    config.run.seed = seed
    return config


def _typed_preset(encoder: str, epochs: int = 50, lr: "float" = 1e-4) -> ExperimentConfig:
    """A user's preset with typed arguments (``lr`` annotated as under postponed annotations)."""
    config = presets.mil_classification(encoder)
    config.trainer.max_epochs = epochs
    config.learner.params["optimizer"] = Component("adamw", {"lr": lr})
    return config


@pytest.fixture()
def typed_presets():
    from soma import register
    from soma.registry import signature_params

    for name, preset in (("seeded", _seeded_preset), ("typed", _typed_preset)):
        register("presets", name, preset, params=signature_params(preset, name=f"{name.title()}Params"))
    try:
        yield
    finally:
        for name in ("seeded", "typed"):
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


def _my_task(task: str) -> ExperimentConfig:
    """A user's preset whose argument can take another preset's name as its value."""
    config = presets.mil_classification("uni2")
    config.run.run_id = task
    return config


def test_init_tells_a_preset_name_from_a_flag_value_naming_another_preset(capsys):
    from soma import register
    from soma.cli import _selected_preset

    register("presets", "my_task", _my_task)
    try:
        assert _selected_preset(["--task", "survival", "my_task"], positional=True) == "my_task"
        assert yaml.safe_load(_init(["--task", "survival", "my_task"], capsys))["run"]["run_id"] == "survival"
    finally:
        presets.preset_registry._entries.pop("my_task", None)


def _rows(out: str) -> list[list[str]]:
    """The cells of each table row ``soma list`` printed."""
    return [[cell.strip() for cell in line.split("│")[1:-1]] for line in out.splitlines() if line.startswith("│")]


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
    rows = {row[0]: row[1] for row in _rows(out) if row[0]}
    assert {"abmil", "clam_sb", "mean_pool", "transmil"} <= set(rows)
    assert "hidden_dim=128" in rows["abmil"]


def test_list_heads_prints_the_registered_heads(capsys):
    from soma.cli import main

    main(["list", "heads"])

    names = [row[0] for row in _rows(capsys.readouterr().out)]
    assert {"binary_classification", "segmentation", "survival", "cox_survival"} <= set(names)


def test_list_encoders_filters_by_level(capsys):
    from soma import components
    from soma.cli import main

    main(["list", "encoders", "--level", "slide"])

    out = capsys.readouterr().out
    assert "Encoders (slide)" in out
    printed = {row[0] for row in _rows(out) if row[0]}
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
