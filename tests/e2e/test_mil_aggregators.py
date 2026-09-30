"""Every registered MIL aggregator on a task with rigged features it must solve perfectly.

A slide is positive iff it contains marker tiles. Tiles are flat colours, so the
features are exact and noise-free, and slides come in bag-size-matched pairs, so
nothing but the tiles carries the label. A correct aggregator therefore classifies
every held-out slide correctly. On top of that, the trained model is re-scored under
conditions the run never saw; a correct aggregator's prediction for a bag must not
depend on:

- the batch it is scored in (alone, or padded next to larger bags);
- what the padding contains (masked positions must be ignored);
- the order of its tiles (for aggregators that are set functions by design).

A control run trains on labels made independent of the markers and must stay near
chance: if it does not, labels are reaching the model some other way.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from soma.aggregators import list_aggregators

from tests.e2e.harness import CPU_EXECUTION, CPU_LOADER, run_soma
from tests.e2e.synthetic import SPACING_UM, MarkerCohort, make_marker_cohort

#: Aggregators whose output depends on tile order by design, with the reason.
ORDER_DEPENDENT = {
    "transmil": "PPEG convolves the tile sequence laid out as a square grid",
    "dtfdmil": "evaluation splits the bag into contiguous pseudo-bags",
}
#: Largest logit difference tolerated between two scorings of the same bag (float
#: reductions over a different number of padded positions are not bit-identical).
LOGIT_ATOL = 1e-4


@pytest.fixture(scope="module")
def cohort(tmp_path_factory) -> MarkerCohort:
    return make_marker_cohort(tmp_path_factory.mktemp("marker-cohort"))


@pytest.fixture(scope="module")
def shared_cache(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("marker-cache")


#: Constructor overrides: HIPT reads 2x2-tile regions (the cohort's tissue blocks are
#: aligned to them); TransMIL is shrunk from its 512-wide default to keep it CPU-sized.
AGGREGATOR_PARAMS = {
    "hipt": {"tile_multiple": 2},
    "transmil": {"att_dim": 64},
}
#: clam_mb has one attention branch per class and refuses binary_classification.
TASK = {"clam_mb": "multiclass_classification"}
AUROC = {"binary_classification": "auroc", "multiclass_classification": "auroc_macro"}
#: z-score normalization is not supported on hierarchical (HIPT) feature streams.
NORMALIZATION = {"hipt": "none"}
#: Training overrides. The scenario trains at 5e-3 (50x soma's 1e-4 default) so every
#: aggregator converges in a few CPU epochs; max pooling (one feature per channel into
#: a linear head) needs more epochs, and HIPT's transformers collapse to a constant
#: at that rate.
TRAINING = {
    "max_pool": {"epochs": 100},
    "hipt": {"epochs": 60, "learning_rate": 3e-4},
}


def _auroc(aggregator: str) -> str:
    return AUROC[TASK.get(aggregator, "binary_classification")]


def _config(cohort, workdir, encoder_name, cache_root, aggregator, *, control=False) -> dict:
    aggregation = {"name": aggregator, "params": AGGREGATOR_PARAMS.get(aggregator, {})}
    training = {"epochs": 30, "learning_rate": 5e-3, **TRAINING.get(aggregator, {})}
    return {
        "run": {"output_root": str(workdir / "runs"), "seed": 0},
        "data": {
            "dataset_csv": str(cohort.manifest(control=control)),
            "splits_csv": str(cohort.splits_csv),
            "dataset_type": "slide",
        },
        "preprocessing": {
            "backend": "openslide",
            "requested_tile_size_px": 32,
            "requested_spacing_um": SPACING_UM,
            "tissue_method": "otsu",
            "seg_downsample": 16,
            "a_t": 1,
            "min_coverage": {"tissue": 0.5},
        },
        "execution": CPU_EXECUTION,
        "cache": {"enabled": True, "root_dir": str(cache_root)},
        "encoder": {"name": encoder_name, "precision": "fp32", "batch_size": 256},
        "aggregation": aggregation,
        "task": {"name": TASK.get(aggregator, "binary_classification")},
        "evaluation": {"metrics": ["accuracy", _auroc(aggregator)]},
        "normalization": {"method": NORMALIZATION.get(aggregator, "zscore")},
        "training": {
            **training,
            # Several bags per batch, so bags are padded and the mask is exercised.
            "batch_size": 4,
            "patience": training["epochs"],
            **CPU_LOADER,
        },
    }


@torch.inference_mode()
def _logits(
    scored, *, batch_size: int, shuffle_tiles: bool = False, garbage_padding: bool = False
) -> dict[str, torch.Tensor]:
    """Re-score the run's test split with the run's own model, dataset and collation."""
    generator = torch.Generator().manual_seed(0)
    dataset, collate = scored.loader.dataset, scored.loader.collate_fn
    items = []
    for index in range(len(dataset)):
        features, targets, sample_id = dataset[index]
        if shuffle_tiles:
            # Axis 0 is the tile axis (the region axis for hierarchical features).
            features = features[torch.randperm(features.shape[0], generator=generator)]
        items.append((features, targets, sample_id))

    model = scored.model.eval()
    logits = {}
    for start in range(0, len(items), batch_size):
        batch = collate(items[start : start + batch_size])
        features = batch.features
        if garbage_padding:
            features = features.clone()
            padded = ~batch.mask
            noise_shape = (int(padded.sum()), *features.shape[2:])
            features[padded] = 1e3 * torch.randn(noise_shape, generator=generator)
        out = model(features, mask=batch.mask)
        logits.update(zip(batch.sample_ids, out.logits))
    return logits


def _max_abs_diff(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    assert a.keys() == b.keys()
    return max(float((a[k] - b[k]).abs().max()) for k in a)


@pytest.mark.parametrize("aggregator", list_aggregators())
def test_aggregator_solves_rigged_task_and_scores_bags_independently(
    aggregator, cohort, shared_cache, tmp_path, encoder_name, scored_models, new_artifact
):
    artifact = new_artifact(f"mil_aggregator_{aggregator}")
    config = _config(cohort, tmp_path, encoder_name, shared_cache, aggregator)
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_training_reduced_loss("run", run)
    artifact.check_equal("test/num_samples", run.summary["test/num_samples"], 12.0)
    artifact.check_equal("test/accuracy", run.summary["test/accuracy"], 1.0)
    auroc = f"test/{_auroc(aggregator)}"
    artifact.check_equal(auroc, run.summary[auroc], 1.0)

    scored = scored_models["test"]
    # Tiling must yield exactly the designed bags: one feature row per tissue tile.
    dataset = scored.loader.dataset
    designed = dict(zip(cohort.sample_ids, cohort.bag_size.tolist()))
    tiles = {}
    for index in range(len(dataset)):
        features, _, sample_id = dataset[index]
        tiles[sample_id] = int(features.shape[:-1].numel())
    artifact.check_equal("test/tiles_per_bag", tiles, {k: designed[k] for k in tiles})
    batch_size = scored.loader.batch_size
    reference = _logits(scored, batch_size=batch_size)
    # The re-scoring harness itself must reproduce the probabilities the run wrote.
    head = scored.model.task_head
    reported = {p.sample_id: torch.tensor(p.probabilities) for p in scored.report.predictions}
    rescored = {
        k: torch.as_tensor(head.postprocess(v[None])["probabilities"][0])
        for k, v in reference.items()
    }
    artifact.check_at_most(
        "rescore/matches_run_probabilities", _max_abs_diff(reported, rescored), 1e-6
    )

    variants = {
        "batch_of_1": _logits(scored, batch_size=1),
        "one_batch": _logits(scored, batch_size=len(reference)),
        "one_batch_garbage_padding": _logits(
            scored, batch_size=len(reference), garbage_padding=True
        ),
        "tiles_shuffled": _logits(scored, batch_size=batch_size, shuffle_tiles=True),
    }
    for name, logits in variants.items():
        diff = _max_abs_diff(reference, logits)
        if name == "tiles_shuffled" and aggregator in ORDER_DEPENDENT:
            artifact.facts[f"invariance/{name}"] = {
                "max_logit_diff": round(diff, 6),
                "not_checked": ORDER_DEPENDENT[aggregator],
            }
            continue
        artifact.check_at_most(f"invariance/{name}/max_logit_diff", diff, LOGIT_ATOL)
    artifact.assert_passed()


def test_labels_independent_of_the_tiles_are_not_learned(
    cohort, shared_cache, tmp_path, encoder_name, new_artifact
):
    """Negative control: the rigged task's perfect scores must come from the tiles."""
    artifact = new_artifact("mil_aggregator_control")
    config = _config(cohort, tmp_path, encoder_name, shared_cache, "abmil", control=True)
    run = run_soma(config, tmp_path / "config.yaml")
    artifact.record_run("run", run)
    artifact.check_at_most("test/accuracy", run.summary["test/accuracy"], 0.75)
    artifact.assert_passed()
