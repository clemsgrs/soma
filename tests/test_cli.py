"""Tests for soma CLI entrypoint."""

import sys
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


# --- `soma run config.yaml` launches pipeline ---


def test_run_calls_pipeline_run(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)

    with patch("soma.cli.Pipeline") as MockPipeline:
        mock_instance = MagicMock()
        MockPipeline.return_value = mock_instance

        from soma.cli import main
        main(["run", str(config_path)])

    MockPipeline.assert_called_once()
    mock_instance.run.assert_called_once()


def test_run_passes_correct_config_to_pipeline(tmp_path: Path):
    config_path = _make_valid_config(tmp_path)

    with patch("soma.cli.Pipeline") as MockPipeline:
        mock_instance = MagicMock()
        MockPipeline.return_value = mock_instance

        from soma.cli import main
        main(["run", str(config_path)])

    config_arg = MockPipeline.call_args[0][0]
    assert config_arg.encoder.name == "uni2"
    assert config_arg.task.name == "binary_classification"


# --- Error cases ---


def test_run_missing_config_file_exits(tmp_path: Path, capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(tmp_path / "nonexistent.yaml")])

    assert exc_info.value.code != 0
    captured = capsys.readouterr()
    assert "nonexistent.yaml" in captured.err


def test_run_invalid_yaml_exits(tmp_path: Path, capsys):
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text(": invalid: yaml: [")

    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(bad_yaml)])

    assert exc_info.value.code != 0


def test_run_invalid_config_content_exits(tmp_path: Path, capsys):
    # Valid YAML but missing required fields (no task)
    raw = {
        "dataset_csv": "data.csv",
        "splits_csv": "splits.csv",
        "output_root": "out",
        "dataset_type": "slide",
    }
    config_path = tmp_path / "config.yaml"
    with config_path.open("w") as f:
        yaml.safe_dump(raw, f)

    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main(["run", str(config_path)])

    assert exc_info.value.code != 0


# --- Help / no-args ---


def test_no_args_exits_with_usage(capsys):
    from soma.cli import main

    with pytest.raises(SystemExit) as exc_info:
        main([])

    # argparse exits with code 2 (usage error) when no subcommand given
    assert exc_info.value.code != 0
