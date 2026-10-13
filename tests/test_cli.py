"""Tests for soma CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from soma.config import FeaturesConfig
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


def test_run_shorthand_passes_config_and_runs_pipeline(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)

    with patch("soma.cli.Pipeline") as MockPipeline:
        from soma.cli import main
        main([str(config_path)])

    MockPipeline.assert_called_once()
    MockPipeline.return_value.run.assert_called_once()
    config_arg = MockPipeline.call_args[0][0]
    assert config_arg.features.encoder.name == "uni2"
    assert config_arg.learner.params["head"].name == "binary_classification"


# --- --set overrides ---


def _run_with_args(config_args: list[str]):
    """Run main() with Pipeline patched; return the ExperimentConfig it received."""
    with patch("soma.cli.Pipeline") as MockPipeline:
        MockPipeline.return_value = MagicMock()
        from soma.cli import main
        main(config_args)
    return MockPipeline.call_args[0][0]


def test_set_override_repoints_scalar(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)
    config_arg = _run_with_args([str(config_path), "--set", "features.encoder.name=phikon"])
    assert config_arg.features.encoder.name == "phikon"


def test_set_override_parses_value_type(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)
    config_arg = _run_with_args([str(config_path), "--set", "trainer.max_epochs=3"])
    # YAML-parsed, so it lands as an int, not the string "3".
    assert config_arg.trainer.max_epochs == 3
    assert isinstance(config_arg.trainer.max_epochs, int)


def test_set_override_equals_form_and_paths(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)
    config_arg = _run_with_args(
        [str(config_path), "--set=cohort.records=/elsewhere/dataset.csv"]
    )
    assert str(config_arg.cohort.records) == "/elsewhere/dataset.csv"


def test_multiple_set_overrides(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)
    config_arg = _run_with_args(
        [str(config_path), "--set", "features.encoder.name=phikon", "--set", "run.output_dir=/out"]
    )
    assert config_arg.features.encoder.name == "phikon"
    assert str(config_arg.run.output_dir) == "/out"


def test_parse_set_overrides_builds_nested_dict():
    from soma.cli import _parse_set_overrides

    out = _parse_set_overrides(["a.b.c=1", "a.b.d=hello", "x=true"])
    assert out == {"a": {"b": {"c": 1, "d": "hello"}}, "x": True}


def test_set_without_value_exits(tmp_path: Path, capsys):
    config_path = _make_valid_config(tmp_path)
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--set"])
    assert exc_info.value.code == 2
    assert "--set requires" in capsys.readouterr().err


def test_set_malformed_pair_exits(tmp_path: Path, capsys):
    config_path = _make_valid_config(tmp_path)
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--set", "noequalssign"])
    assert exc_info.value.code == 2
    assert "key=value" in capsys.readouterr().err


# --- --feature-dir ---


def _source_for(config_args: list[str]):
    """Run main() with Pipeline patched; return the ``source`` it was constructed with."""
    with patch("soma.cli.Pipeline") as MockPipeline:
        from soma.cli import main

        main(config_args)
    return MockPipeline.call_args.kwargs["source"]


def test_feature_dir_attaches_a_cached_set_source(tmp_path: Path):
    import torch

    from soma.data import CachedSetSource

    config_path = _make_valid_config(tmp_path)
    features = tmp_path / "features"
    features.mkdir()
    torch.save(torch.randn(3, 4), features / "s0.pt")

    source = _source_for([str(config_path), "--feature-dir", str(features)])

    assert isinstance(source, CachedSetSource) and source.feature_dir == features
    assert _source_for([str(config_path)]) is None


def test_feature_dir_attaches_a_cached_grid_source_for_a_dense_task(tmp_path: Path):
    from soma.dense import CachedGridSource

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

    source = _source_for([str(config_path), f"--feature-dir={grids}"])

    assert isinstance(source, CachedGridSource)
    assert source.provenance.to_dict()["kind"] == "dense_cache"


def test_feature_dir_must_be_a_directory(tmp_path: Path, capsys):
    config_path = _make_valid_config(tmp_path)
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path), "--feature-dir", str(tmp_path / "missing")])
    assert exc_info.value.code == 1
    assert "--feature-dir" in capsys.readouterr().err


# --- Error cases ---


def test_run_missing_config_file_exits(tmp_path: Path, capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(tmp_path / "nonexistent.yaml")])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "nonexistent.yaml" in captured.err


def test_run_invalid_yaml_exits(tmp_path: Path, capsys):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(": invalid: yaml: [")

    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(bad_yaml)])

    assert exc_info.value.code != 0


def test_run_invalid_config_content_exits(tmp_path: Path, capsys):
    # Valid YAML but an unknown section.
    raw = {
        "cohort": {"records": "data.csv", "splits": "splits.csv"},
        "data": {"dataset_type": "case"},
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(raw, f)

    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([str(config_path)])

    assert exc_info.value.code != 0


def test_run_subcommand_is_rejected(tmp_path: Path, capsys):
    config_path = _make_valid_config(tmp_path)

    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(config_path)])

    assert exc_info.value.code == 2
    captured = capsys.readouterr()
    assert "pass the config path directly" in captured.err


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
    assert "soma /path/to/config.yaml" in out
