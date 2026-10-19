"""Extension seam: a user-registered task head, decoder, aggregator and learner train through
the real CLI.

The customisation API is subclass-and-override: subclass the family's built-in head (or
``Decoder``, or the shape's ``Learner``), override the methods that should differ, and
reference the subclass from the config, either under a name registered with its params
dataclass (and, for a head whose class count the cohort decides, the family's ``infer``
hook) or by its dotted import path, which resolves with a warning and passes its params
through unvalidated (ADR 0018). Validation keys on the registered class's
``task_family``, so the custom head passes everywhere the built-in one does. A custom set
head passes the ``soma.testing.check_task_head`` conformance suite, and a learner's
``predict_step`` override adds a column to the run's predictions. A custom aggregator
trains on padded batches and passes ``soma.testing.check_aggregator``.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

import soma
from soma.cli import main as soma_main
from soma.models.decoders.base import Decoder
from soma.models.decoders.registry import decoder_registry
from soma.evaluation import Predictions
from soma.learn import DenseLearner, MILLearner, PredictOutput, learner_registry
from soma.models.aggregators import Aggregator, AggregatorOutput, aggregator_registry
from soma.models.heads import ClassificationParams, SegmentationParams, infer_label_map
from soma.models.heads.classification import BinaryClassificationHead
from soma.models.heads.dense_metrics import soft_dice_loss
from soma.models.heads.registry import task_registry
from soma.models.heads.segmentation import SegmentationHead
from soma.registry import NoParams
from soma.testing import check_aggregator, check_task_head
from soma.learn.sample_dataset import SampleBatch
from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, learner_section, run_soma, trainer_section
from tests.e2e.synthetic import SPACING_UM, make_dense_dataset, make_marker_cohort, make_slide_cohort

CUSTOM_HEAD = "e2e_dice_only_segmentation"
CUSTOM_DECODER = "e2e_two_stage_conv"
CUSTOM_SET_HEAD = "e2e_label_smoothing_classification"
CUSTOM_AGGREGATOR = "e2e_projected_mean"
EDITABLE_AGGREGATOR = "e2e_editable_mean_pool"

#: A user's aggregator in a module file of its own, which the code-guard scenario edits.
USER_MODULE = "e2e_user_aggregators"
USER_SOURCE = '''"""A user's aggregator, in a module file the scenario edits between runs."""

from soma.models import MeanPool


class EditableMeanPool(MeanPool):
    def forward(self, X, mask=None):
        return super().forward(X, mask)
'''
#: The same file after the user edits ``forward``. It is longer, so Python cannot load
#: stale bytecode for it.
USER_SOURCE_EDITED = USER_SOURCE.replace("super().forward(X, mask)", "super().forward(X * 1.0, mask)")

# Call counters: the scenario proves the overrides actually ran, not just registered.
CALLS = {
    "loss": 0, "decoder": 0, "entropy": 0, "smoothing": 0, "predict": 0, "smooth_grid": 0,
    "aggregator": 0, "padded_bags": 0,
}


class AttentionEntropyLearner(MILLearner):
    """``MILLearner`` plus an attention-entropy regulariser: one ``training_step`` override.

    Referenced from the config by dotted path (``learner.name``), so nothing is
    registered; ``weight`` arrives through the learner's ``params``.
    """

    def __init__(self, model, *, weight: float, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.weight = float(weight)

    def training_step(self, batch) -> Tensor:
        output = self.forward_batch(batch)
        attention = output.tile_attention.masked_fill(~batch.mask, float("-inf")).softmax(-1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1).mean()
        CALLS["entropy"] += 1
        return self.compute_loss(output, batch) + self.weight * entropy

    def predict_step(self, batch) -> PredictOutput:
        """The head's columns plus each bag's attention entropy (one extra column)."""
        step = super().predict_step(batch)
        attention = step.raw.tile_attention.masked_fill(~batch.mask, float("-inf")).softmax(-1)
        entropy = -(attention * attention.clamp_min(1e-8).log()).sum(-1)
        CALLS["predict"] += 1
        return PredictOutput(columns={**step.columns, "attention_entropy": entropy}, raw=step.raw)


class SmoothGridLearner(DenseLearner):
    """``DenseLearner`` plus a total-variation penalty on the predicted class probabilities:
    one ``training_step`` override on the grid shape, named by dotted path in the config."""

    def __init__(self, model, *, weight: float, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.weight = float(weight)

    def training_step(self, batch) -> Tensor:
        output = self.forward_batch(batch)
        probabilities = output.logits.softmax(dim=1)
        variation = (probabilities[..., 1:, :] - probabilities[..., :-1, :]).abs().mean() + (
            probabilities[..., 1:] - probabilities[..., :-1]
        ).abs().mean()
        CALLS["smooth_grid"] += 1
        return self.compute_loss(output, batch) + self.weight * variation


class LabelSmoothingHead(BinaryClassificationHead):
    """``BinaryClassificationHead`` trained with label-smoothed cross-entropy."""

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        CALLS["smoothing"] += 1
        return F.cross_entropy(predictions, targets["label"], label_smoothing=0.1)


class DiceOnlySegmentationHead(SegmentationHead):
    """``SegmentationHead`` with the cross-entropy term dropped: pure soft Dice."""

    def compute_loss(self, predictions: Tensor, targets: dict[str, Tensor]) -> Tensor:
        CALLS["loss"] += 1
        return soft_dice_loss(
            predictions,
            targets["mask"],
            num_classes=self.num_classes,
            ignore_index=self.ignore_index,
        )


@dataclass
class TwoStageConvParams:
    """``TwoStageConvDecoder``'s params: registering them validates a config's values."""

    hidden_dim: int = 16

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")


class TwoStageConvDecoder(Decoder):
    """Upsample, 3x3 conv, upsample, 3x3 conv to class logits — no norm, no activation."""

    def __init__(self, *, input_dim: int, num_classes: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self._num_classes = int(num_classes)
        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(hidden_dim, num_classes, kernel_size=3, padding=1),
        )

    def forward(self, X: Tensor) -> Tensor:
        CALLS["decoder"] += 1
        return self.layers(X)

    @property
    def num_classes(self) -> int:
        return self._num_classes


@dataclass
class ProjectedMeanParams:
    """``ProjectedMeanAggregator``'s params: registering them validates a config's values."""

    hidden_dim: int = 16

    def __post_init__(self) -> None:
        if self.hidden_dim < 1:
            raise ValueError(f"hidden_dim must be >= 1, got {self.hidden_dim}")


class ProjectedMeanAggregator(Aggregator):
    """Project each tile, then average the bag's real tiles: the mask drops the padding,
    and ``mask=None`` means every tile is real (the docs' worked aggregator)."""

    def __init__(self, input_dim: int, hidden_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.project = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())

    def forward(self, X: Tensor, mask: Tensor | None = None) -> AggregatorOutput:
        CALLS["aggregator"] += 1
        if mask is None:
            mask = torch.ones(X.shape[:2], dtype=torch.bool, device=X.device)
        CALLS["padded_bags"] += int((~mask).any(dim=1).sum())
        H = self.project(X).masked_fill(~mask.unsqueeze(-1), 0.0)
        z = H.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        return AggregatorOutput(bag_representation=z)

    @property
    def output_dim(self) -> int:
        return self.hidden_dim


@pytest.fixture
def registered_extensions(monkeypatch):
    """Register the custom heads, decoder and aggregator with their params for one test,
    then restore the registries.

    The set head inherits nothing from its base's registration, so it names the
    categorical heads' params and their ``infer`` hook (the cohort decides its class
    count and label map). Without the restore, the names would leak into every later test
    that lists the registries (CI runs the unit and E2E suites in a single pytest session).
    """
    monkeypatch.setattr(task_registry, "_entries", dict(task_registry._entries))
    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    monkeypatch.setattr(aggregator_registry, "_entries", dict(aggregator_registry._entries))
    monkeypatch.setattr(learner_registry, "_warned", set())
    soma.register("heads", CUSTOM_HEAD, DiceOnlySegmentationHead, params=SegmentationParams)
    soma.register(
        "heads", CUSTOM_SET_HEAD, LabelSmoothingHead, params=ClassificationParams, infer=infer_label_map
    )
    soma.register("decoders", CUSTOM_DECODER, TwoStageConvDecoder, params=TwoStageConvParams)
    soma.register("aggregators", CUSTOM_AGGREGATOR, ProjectedMeanAggregator, params=ProjectedMeanParams)


def test_custom_head_and_decoder_train_through_cli(
    tmp_path, encoder_name, encoded_images, new_artifact, registered_extensions
):
    CALLS["loss"] = CALLS["decoder"] = 0
    artifact = new_artifact("extension_custom_head_decoder")

    roi_px = 96
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=roi_px)
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": roi_px,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "dense",
            {
                "decoder": {"name": CUSTOM_DECODER, "params": {"hidden_dim": 16}},
                "head": {"name": CUSTOM_HEAD, "params": {"num_classes": 3}},
            },
            learning_rate=3e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(20),
        "evaluation": {"metrics": ["mean_dice", "mean_iou"]},
    }
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("rois_encoded", sum(encoded_images), 12)
    artifact.check_at_least("custom_loss_calls", CALLS["loss"], 1)
    artifact.check_at_least("custom_decoder_calls", CALLS["decoder"], 1)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.assert_passed()


def test_custom_aggregator_trains_through_cli(tmp_path, encoder_name, new_artifact, registered_extensions):
    """A user-registered ``Aggregator`` trains a slide classifier through the CLI under its
    registered name. The marker cohort's bags hold 4 to 64 tiles and a batch holds four,
    so bags are padded and the aggregator must read the mask. ``plan.yaml`` records the
    validated params, and the aggregator passes ``check_aggregator``."""
    CALLS["aggregator"] = CALLS["padded_bags"] = 0
    artifact = new_artifact("extension_custom_aggregator")
    cohort = make_marker_cohort(tmp_path / "cohort")
    epochs = 30
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(cohort.manifest()), "splits": str(cohort.splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": 32,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
                "segmentation": {"method": "otsu", "downsample": 16},
                "filter": {"ref_tile_size": 32, "a_t": 1},
                "masks": {"pixel_mapping": {"background": 0, "tissue": 1}, "min_coverage": {"tissue": 0.5}},
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 256},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "mil",
            {
                "aggregator": {"name": CUSTOM_AGGREGATOR, "params": {"hidden_dim": 16}},
                "head": {"name": "binary_classification"},
            },
            learning_rate=5e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(epochs),
        "evaluation": {"metrics": ["accuracy", "auroc"]},
    }
    run = run_soma(config, tmp_path / "config.yaml")

    plan = yaml.safe_load((run.run_dir / "plan.yaml").read_text())
    artifact.check_equal(
        "plan/aggregator",
        plan["learner"]["params"]["aggregator"],
        {"name": CUSTOM_AGGREGATOR, "params": {"hidden_dim": 16}},
    )
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("custom_aggregator_calls", CALLS["aggregator"], epochs)
    artifact.check_at_least("custom_aggregator_padded_bags", CALLS["padded_bags"], 1)
    artifact.check_equal("test/accuracy", run.summary["test/accuracy"], 1.0)

    check_aggregator(ProjectedMeanAggregator(input_dim=8), input_dim=8)
    artifact.check_equal("custom_aggregator/check_aggregator", "passed", "passed")
    artifact.assert_passed()


def test_custom_learner_by_dotted_path_trains_through_cli(
    tmp_path, encoder_name, new_artifact, registered_extensions
):
    """A ``MILLearner`` subclass adding an intermediate loss trains through the CLI as
    the ``learner``, named by dotted import path and given its params; its
    ``predict_step`` adds a column to the predictions. ``plan.yaml`` records the dotted
    path with its params verbatim, and every bundle's ``provenance.json`` records the
    class it resolved to and the digest of the file that defines it, which is its
    fingerprint (the file belongs to no installed distribution). The registered custom
    head it trains passes ``check_task_head``."""
    CALLS["entropy"] = CALLS["smoothing"] = CALLS["predict"] = 0
    artifact = new_artifact("extension_custom_learner")
    cohort = make_slide_cohort(tmp_path / "cohort")
    epochs = 25
    learner = "tests.e2e.test_extension.AttentionEntropyLearner"
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(cohort.manifest("binary_classification")), "splits": str(cohort.splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": 32,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
                "segmentation": {"method": "otsu", "downsample": 16},
                "filter": {"ref_tile_size": 32, "a_t": 1},
                "masks": {"pixel_mapping": {"background": 0, "tissue": 1}, "min_coverage": {"tissue": 0.5}},
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 256},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 1, **CPU_LOADER},
        "learner": learner_section(
            learner,
            {"aggregator": {"name": "abmil"}, "head": {"name": CUSTOM_SET_HEAD}},
            learning_rate=5e-3,
            normalization={"method": "zscore"},
            weight=0.01,
        ),
        "trainer": trainer_section(epochs),
        "evaluation": {"metrics": ["auroc", "balanced_accuracy"]},
    }
    with pytest.warns(UserWarning, match=r"learners 'tests\.e2e\.test_extension\.AttentionEntropyLearner' is not registered"):
        run = run_soma(config, tmp_path / "config.yaml")

    plan = yaml.safe_load((run.run_dir / "plan.yaml").read_text())
    artifact.check_equal("plan/learner_name", plan["learner"]["name"], learner)
    artifact.check_equal("plan/learner_params_verbatim", plan["learner"]["params"]["weight"], 0.01)
    provenance = json.loads((run.run_dir / "bundle" / "provenance.json").read_text())
    recorded = _component(provenance["components"], "learner")
    source_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    artifact.check_equal("provenance/learner/object", recorded["object"], learner)
    artifact.check_equal("provenance/learner/source_sha256", recorded["source_sha256"], source_sha256)
    artifact.check_equal("provenance/learner/fingerprint", recorded["fingerprint"], source_sha256)

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("custom_training_steps", CALLS["entropy"], epochs)
    artifact.check_at_least("test/auroc", run.summary["test/auroc"], 0.95)
    history = run.histories()[""]
    artifact.check_equal("history/columns_without_tune_prefix", "tune_loss" in history[0], False)
    artifact.check_equal("history/loss_present", "loss" in history[0], True)

    artifact.check_at_least("custom_head_loss_calls", CALLS["smoothing"], epochs)
    artifact.check_at_least("custom_predict_steps", CALLS["predict"], 1)
    predictions = Predictions.from_parquet(run.run_dir / "predictions_test.parquet")
    entropy = predictions.frame["attention_entropy"].to_numpy()
    artifact.check_equal("predictions/extra_column_role", predictions.column_roles["attention_entropy"], "extra")
    artifact.check_equal("predictions/extra_column_finite", bool(np.isfinite(entropy).all()), True)
    artifact.check_equal("predictions/declared_columns", predictions.prediction_columns, ("prob_0", "prob_1", "predicted_label"))

    torch.manual_seed(0)
    check_task_head(
        LabelSmoothingHead(input_dim=8, num_classes=2),
        SampleBatch(
            features=torch.randn(8, 8),
            targets={"label": torch.tensor([0, 1] * 4)},
            sample_ids=tuple(f"s{i}" for i in range(8)),
        ),
    )
    artifact.check_equal("custom_head/check_task_head", "passed", "passed")
    artifact.assert_passed()


def test_custom_grid_learner_by_dotted_path_trains_through_cli(
    tmp_path, encoder_name, new_artifact, registered_extensions
):
    """A ``DenseLearner`` subclass adding a regulariser trains a cached segmentation run
    through the CLI as the ``learner``, named by dotted import path."""
    CALLS["smooth_grid"] = 0
    artifact = new_artifact("extension_custom_grid_learner")
    roi_px = 96
    dataset_csv, splits_csv = make_dense_dataset(tmp_path / "data", "segmentation", size=roi_px)
    epochs = 20
    config = {
        "run": {"output_dir": str(tmp_path / "runs"), "seed": 0},
        "cohort": {"records": str(dataset_csv), "splits": str(splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": roi_px,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 8},
            "cache": {"enabled": True, "root_dir": str(tmp_path / "cache")},
        },
        "loaders": {"batch_size": 4, **CPU_LOADER},
        "learner": learner_section(
            "tests.e2e.test_extension.SmoothGridLearner",
            {
                "decoder": {"name": "lightweight_conv"},
                "head": {"name": "segmentation", "params": {"num_classes": 3}},
            },
            learning_rate=3e-3,
            normalization={"method": "zscore"},
            weight=0.1,
        ),
        "trainer": trainer_section(epochs),
        "evaluation": {"metrics": ["mean_dice", "mean_iou"]},
    }
    run = run_soma(config, tmp_path / "config.yaml")

    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_at_least("custom_training_steps", CALLS["smooth_grid"], epochs)
    artifact.check_at_least("test/mean_dice", run.summary["test/mean_dice"], 0.8)
    artifact.assert_passed()


# --- the code a run records ------------------------------------------------------------


@pytest.fixture
def user_aggregator(tmp_path, monkeypatch):
    """``EditableMeanPool`` from a module file on the path, registered as
    ``EDITABLE_AGGREGATOR``; ``edit()`` rewrites ``forward`` in the file, reloads the module
    and registers the new class under the same name, as a new session would."""
    directory = tmp_path / "user_code"
    directory.mkdir()
    path = directory / f"{USER_MODULE}.py"
    path.write_text(USER_SOURCE)
    monkeypatch.syspath_prepend(str(directory))
    entries = dict(aggregator_registry._entries)

    def load() -> None:
        importlib.invalidate_caches()
        module = sys.modules.get(USER_MODULE)
        module = importlib.reload(module) if module is not None else importlib.import_module(USER_MODULE)
        monkeypatch.setattr(aggregator_registry, "_entries", dict(entries))
        soma.register("aggregators", EDITABLE_AGGREGATOR, module.EditableMeanPool, params=NoParams)

    def edit() -> None:
        path.write_text(USER_SOURCE_EDITED)
        load()

    load()
    yield SimpleNamespace(path=path, edit=edit)
    sys.modules.pop(USER_MODULE, None)


def _slide_config(cohort, workdir: Path, encoder_name: str, *, aggregator: str, seed: int = 0, resume: bool = False) -> dict:
    """A slide-level binary classifier over ``cohort`` with ``aggregator``, its features
    cached under ``workdir``."""
    return {
        "run": {"output_dir": str(workdir / "runs"), "seed": seed, "resume": resume},
        "cohort": {"records": str(cohort.manifest("binary_classification")), "splits": str(cohort.splits_csv)},
        "features": {
            "encoder": {"name": encoder_name},
            "extraction": {
                "tiling": {
                    "requested_spacing_um": SPACING_UM,
                    "requested_tile_size_px": 32,
                    "tolerance": 0.05,
                    "overlap": 0.0,
                    "backend": "openslide",
                },
                "segmentation": {"method": "otsu", "downsample": 16},
                "filter": {"ref_tile_size": 32, "a_t": 1},
                "masks": {"pixel_mapping": {"background": 0, "tissue": 1}, "min_coverage": {"tissue": 0.5}},
            },
            "execution": {**CPU_EXECUTION, "precision": "fp32", "batch_size": 256},
            "cache": {"enabled": True, "root_dir": str(workdir / "cache")},
        },
        "loaders": {"batch_size": 1, **CPU_LOADER},
        "learner": learner_section(
            "mil",
            {"aggregator": {"name": aggregator}, "head": {"name": "binary_classification"}},
            learning_rate=5e-3,
            normalization={"method": "zscore"},
        ),
        "trainer": trainer_section(25),
        "evaluation": {"metrics": ["auroc"]},
    }


def _run_yaml(run_dir: Path) -> dict:
    return yaml.safe_load((run_dir / "run.yaml").read_text())


def _component(record: list[dict], kind: str) -> dict:
    (entry,) = [entry for entry in record if entry["kind"] == kind]
    return entry


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_edited_component_code_is_recorded_and_guards_resume_and_leaderboard(
    tmp_path, encoder_name, new_artifact, user_aggregator
):
    """A registered aggregator in a module file, edited between two seeds of one
    experiment: each run records the file's digest, a resume across the edit is refused,
    and the leaderboard refuses to pool the two seeds until the stale one is rerun."""
    artifact = new_artifact("extension_component_code_guards")
    cohort = make_slide_cohort(tmp_path / "cohort")
    config_path = tmp_path / "config.yaml"

    first = run_soma(_slide_config(cohort, tmp_path, encoder_name, aggregator=EDITABLE_AGGREGATOR), config_path)
    artifact.record_run("seed_0", first)
    artifact.check_training_reduced_loss("seed_0", first)
    record = _run_yaml(first.run_dir)["components"]
    aggregator = _component(record, "aggregator")
    artifact.check_equal("seed_0/aggregator/name", aggregator["name"], EDITABLE_AGGREGATOR)
    artifact.check_equal("seed_0/aggregator/object", aggregator["object"], f"{USER_MODULE}.EditableMeanPool")
    artifact.check_equal("seed_0/aggregator/source_sha256", aggregator["source_sha256"], _sha256(USER_SOURCE))
    artifact.check_equal("seed_0/aggregator/fingerprint", aggregator["fingerprint"], _sha256(USER_SOURCE))
    artifact.check_equal("seed_0/aggregator/not_installed", "distribution" in aggregator, False)
    provenance = json.loads((first.run_dir / "bundle" / "provenance.json").read_text())
    artifact.check_equal("seed_0/bundle_records_the_same_components", provenance["components"], record)
    artifact.check_equal(
        "seed_0/head/object",
        _component(record, "head")["object"],
        "soma.models.heads.classification.BinaryClassificationHead",
    )

    # A resume onto the seed-0 run passes while the code is unchanged...
    resume_path = tmp_path / "resume.yaml"
    resume = _slide_config(cohort, tmp_path, encoder_name, aggregator=EDITABLE_AGGREGATOR, resume=True)
    resume_path.write_text(yaml.safe_dump(resume, sort_keys=False))
    soma_main(["run", str(resume_path)])
    run_dirs = sorted((tmp_path / "runs").glob("experiments/*/runs/*"))
    artifact.check_equal("resume_unchanged/reused_the_run", run_dirs == [first.run_dir], True)
    artifact.check_equal("resume_unchanged/status", _run_yaml(first.run_dir)["status"], "completed")

    # ... and is refused once `forward` is edited, naming the aggregator and both fingerprints.
    user_aggregator.edit()
    with pytest.raises(ValueError, match="Refusing to resume") as refused:
        soma_main(["run", str(resume_path)])
    message = str(refused.value)
    artifact.check_equal("resume_edited/names_the_aggregator", f"aggregator '{EDITABLE_AGGREGATOR}'" in message, True)
    artifact.check_equal("resume_edited/names_the_old_fingerprint", _sha256(USER_SOURCE)[:12] in message, True)
    artifact.check_equal("resume_edited/names_the_new_fingerprint", _sha256(USER_SOURCE_EDITED)[:12] in message, True)
    artifact.check_equal(
        "resume_edited/run_keeps_its_record",
        _component(_run_yaml(first.run_dir)["components"], "aggregator")["fingerprint"],
        _sha256(USER_SOURCE),
    )
    artifact.assert_passed()
