"""soma's default log output when the caller has not configured Python logging."""

from __future__ import annotations

import contextlib
import io
import logging
from pathlib import Path
from unittest.mock import patch

from soma import (

    AggregatorConfig,

    EncoderConfig,

    FeatureExtractor,

    Pipeline,

    PipelineConfig,

    TaskConfig,

    TrainingConfig,

)
from soma.data._legacy import legacy_samples_from_csv
from soma.cli import main
from soma.config import save_config
from tests.test_pipeline import FIXED_RUN_ID, _setup_synthetic_data


@contextlib.contextmanager
def no_logging_setup():
    """Run the block in a process with no logging configured, then restore pytest's.

    pytest attaches its capture handlers to the root logger for each test phase, which
    is exactly the "caller configured logging" case. They are detached here (inside the
    test body, after pytest attached them) so soma sees a bare process, and every
    handler, level and propagate flag is put back afterwards so no handler leaks into
    other tests.
    """
    root = logging.getLogger()
    soma_logger = logging.getLogger("soma")
    saved = (
        root.handlers[:],
        root.level,
        soma_logger.handlers[:],
        soma_logger.level,
        soma_logger.propagate,
    )
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    soma_logger.handlers.clear()
    soma_logger.setLevel(logging.NOTSET)
    soma_logger.propagate = True
    try:
        yield
    finally:
        root.handlers[:] = saved[0]
        root.setLevel(saved[1])
        soma_logger.handlers[:] = saved[2]
        soma_logger.setLevel(saved[3])
        soma_logger.propagate = saved[4]


def _slide_config(tmp_path: Path) -> tuple[PipelineConfig, Path]:
    dataset_csv, splits_csv, feature_dir = _setup_synthetic_data(tmp_path)
    config = PipelineConfig(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "output",
        dataset_type="slide",
        aggregator=AggregatorConfig(name="mean_pool"),
        task=TaskConfig(name="binary_classification"),
        training=TrainingConfig(epochs=1, patience=10, batch_size=2),
    )
    return config, feature_dir


def test_pipeline_run_prints_the_fold_summary_to_stderr_once_without_logging_setup(
    tmp_path: Path, capsys
):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup(), patch(
        "soma.output_layout.make_run_id", return_value=FIXED_RUN_ID
    ):
        Pipeline(config, feature_dir=feature_dir).run()

    err = capsys.readouterr().err
    assert err.count("Fold 0: train=6 tune=1 test=1") == 1


def test_feature_extractor_turns_on_soma_info_output_without_logging_setup(
    tmp_path: Path, capsys
):
    dataset_csv = tmp_path / "dataset.csv"
    dataset_csv.write_text("sample_id,image_path,label\ns0,tile.png,1\n", encoding="utf-8")

    with no_logging_setup():
        FeatureExtractor(
            legacy_samples_from_csv(dataset_csv),
            EncoderConfig(name="phikon"),
            output_root=tmp_path / "output",
        )
        logging.getLogger("soma.extraction").info("feature cache resolved")

    assert capsys.readouterr().err.count("feature cache resolved") == 1


def test_a_caller_logging_setup_made_first_is_left_alone(tmp_path: Path, capsys):
    config, feature_dir = _slide_config(tmp_path)
    caller_stream = io.StringIO()

    with no_logging_setup():
        logging.basicConfig(stream=caller_stream, level=logging.INFO, format="%(message)s")
        Pipeline(config, feature_dir=feature_dir)
        soma_handlers = list(logging.getLogger("soma").handlers)
        logging.getLogger("soma.pipeline").info("fold summary")

    assert soma_handlers == []
    assert caller_stream.getvalue().count("fold summary\n") == 1
    assert "fold summary" not in capsys.readouterr().err


def test_constructing_several_pipelines_prints_each_record_once(tmp_path: Path, capsys):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup():
        for _ in range(3):
            Pipeline(config, feature_dir=feature_dir)
        logging.getLogger("soma.pipeline").info("fold summary")

    assert capsys.readouterr().err.count("fold summary") == 1


def test_a_caller_logging_setup_made_after_soma_does_not_duplicate_records(
    tmp_path: Path, capsys
):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup():
        Pipeline(config, feature_dir=feature_dir)
        logging.basicConfig(level=logging.INFO, format="caller: %(message)s")
        logging.getLogger("soma.pipeline").info("fold summary")

    err = capsys.readouterr().err
    assert err.count("fold summary") == 1
    assert "caller: fold summary" in err


def test_the_soma_cli_prints_info_logs(tmp_path: Path, capsys):
    config, _ = _slide_config(tmp_path)
    config_path = tmp_path / "config.yaml"
    save_config(config, config_path)

    class LoggingPipeline:
        """Stands in for the real run: it only logs, so the CLI alone owns the setup."""

        def __init__(self, config):
            pass

        def run(self):
            logging.getLogger("soma.pipeline").info("fold summary")

    with no_logging_setup(), patch("soma.cli.Pipeline", LoggingPipeline):
        main([str(config_path)])

    assert capsys.readouterr().err.count("fold summary") == 1


def test_an_explicit_debug_level_on_the_soma_logger_reaches_the_default_handler(
    tmp_path: Path, capsys
):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup():
        logging.getLogger("soma").setLevel(logging.DEBUG)
        Pipeline(config, feature_dir=feature_dir)
        logging.getLogger("soma.cache").debug("set before construction")
        logging.getLogger("soma").setLevel(logging.INFO)
        logging.getLogger("soma.cache").debug("hidden at info")
        logging.getLogger("soma").setLevel(logging.DEBUG)
        logging.getLogger("soma.cache").debug("set after construction")

    err = capsys.readouterr().err
    assert err.count("set before construction") == 1
    assert "hidden at info" not in err
    assert err.count("set after construction") == 1


def test_a_caller_root_level_set_without_a_handler_is_kept(tmp_path: Path, capsys):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup():
        logging.getLogger().setLevel(logging.ERROR)
        Pipeline(config, feature_dir=feature_dir)
        logging.getLogger("soma.pipeline").info("fold summary")
        logging.getLogger("soma.pipeline").warning("a warning")
        logging.getLogger("soma.pipeline").error("an error")

    err = capsys.readouterr().err
    assert "fold summary" not in err
    assert "a warning" not in err
    assert err.count("an error") == 1


def test_records_still_print_when_soma_does_not_propagate_to_a_later_root_setup(
    tmp_path: Path, capsys
):
    config, feature_dir = _slide_config(tmp_path)

    with no_logging_setup():
        logging.getLogger("soma").propagate = False
        Pipeline(config, feature_dir=feature_dir)
        logging.basicConfig(level=logging.INFO, format="caller: %(message)s")
        logging.getLogger("soma.pipeline").error("an error")

    err = capsys.readouterr().err
    assert err.count("an error") == 1
    assert "caller: an error" not in err


def test_a_caller_handler_added_to_soma_after_soma_does_not_duplicate_records(
    tmp_path: Path, capsys
):
    config, feature_dir = _slide_config(tmp_path)
    caller_stream = io.StringIO()

    with no_logging_setup():
        Pipeline(config, feature_dir=feature_dir)
        caller_handler = logging.StreamHandler(caller_stream)
        caller_handler.setFormatter(logging.Formatter("caller: %(message)s"))
        logging.getLogger("soma").addHandler(caller_handler)
        logging.getLogger("soma.pipeline").info("fold summary")

    assert caller_stream.getvalue().count("caller: fold summary") == 1
    assert "fold summary" not in capsys.readouterr().err
