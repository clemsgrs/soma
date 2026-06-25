"""Tests for soma CLI entrypoint."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from soma.config import (
    AggregatorConfig,
    CacheConfig,
    EncoderConfig,
    PipelineConfig,
    TaskConfig,
    TrainingConfig,
    save_config,
)


def _make_valid_config(tmp_path: Path) -> Path:
    cfg = PipelineConfig(
        dataset_csv="data/dataset.csv",
        splits_csv="data/splits.csv",
        output_root="runs",
        dataset_type="slide",
        cache=CacheConfig(),
        encoder=EncoderConfig(name="uni2"),
        aggregator=AggregatorConfig(name="abmil"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=5),
    )
    path = tmp_path / "config.yaml"
    save_config(cfg, path)
    return path


def test_run_shorthand_calls_pipeline_run(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)

    with patch("soma.cli.Pipeline") as MockPipeline:
        mock_instance = MagicMock()
        MockPipeline.return_value = mock_instance

        from soma.cli import main
        main([str(config_path)])

    MockPipeline.assert_called_once()
    mock_instance.run.assert_called_once()


def test_run_shorthand_passes_correct_config_to_pipeline(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)

    with patch("soma.cli.Pipeline") as MockPipeline:
        mock_instance = MagicMock()
        MockPipeline.return_value = mock_instance

        from soma.cli import main
        main([str(config_path)])

    config_arg = MockPipeline.call_args[0][0]
    assert config_arg.encoder.name == "uni2"
    assert config_arg.task.name == "binary_classification"


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
    # Valid YAML but invalid dataset_type.
    raw = {
        "data": {
            "dataset_csv": "data.csv",
            "splits_csv": "splits.csv",
            "dataset_type": "case",
        }
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


def test_list_encoders_uses_level_filter_and_prints_results(capsys):
    from soma.cli import main

    with patch("soma.cli.list_models", return_value=["alpha", "beta"]) as mock_list_models:
        main(["list", "encoders", "--level", "tile"])

    mock_list_models.assert_called_once_with(level="tile")
    out = capsys.readouterr().out
    assert "Encoders" in out
    assert "tile" in out
    assert "alpha" in out
    assert "beta" in out


def test_list_aggregators_prints_results(capsys):
    from soma.cli import main

    with patch("soma.cli.list_aggregators", return_value=["abmil", "clam_sb"]) as mock_list_aggregators:
        main(["list", "aggregators"])

    mock_list_aggregators.assert_called_once_with()
    out = capsys.readouterr().out
    assert "Aggregators" in out
    assert "abmil" in out
    assert "clam_sb" in out


def test_list_tasks_prints_results(capsys):
    from soma.cli import main

    with patch("soma.cli.list_task_heads", return_value=["binary_classification"]) as mock_list_task_heads:
        main(["list", "tasks"])

    mock_list_task_heads.assert_called_once_with()
    out = capsys.readouterr().out
    assert "Task Heads" in out
    assert "binary_classification" in out


@pytest.mark.parametrize(
    ("kind", "helper", "title", "values"),
    [
        ("decoders", "list_decoders", "Decoders", ["linear", "lightweight_conv"]),
        (
            "pixel-classifiers",
            "list_pixel_classifiers",
            "Pixel Classifiers",
            ["logistic", "xgboost"],
        ),
    ],
)
def test_list_dense_registries_prints_results(kind, helper, title, values, capsys):
    from soma.cli import main

    with patch(f"soma.cli.{helper}", return_value=values) as mock_helper:
        main(["list", kind])

    mock_helper.assert_called_once_with()
    out = capsys.readouterr().out
    assert title in out
    for value in values:
        assert value in out


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
