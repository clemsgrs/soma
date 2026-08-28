from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

import soma.training.segmentation_roi_population as population_module
from soma.dataset import SampleRecord
from soma.training.segmentation_roi_population import (
    resolve_segmentation_roi_population,
)


def _record(sample_id: str) -> SampleRecord:
    return SampleRecord(sample_id=sample_id, image_path=Path(f"{sample_id}.tif"), label=None)


def test_population_is_computed_once_and_reused_exactly(tmp_path: Path) -> None:
    records = [_record("a"), _record("b")]
    calls: list[str] = []
    masks = {
        "a": torch.tensor([[0, 0], [1, 255]]),
        "b": torch.tensor([[1, 1], [1, 0]]),
    }

    def target_fn(record: SampleRecord) -> dict[str, torch.Tensor]:
        calls.append(record.sample_id)
        return {"mask": masks[record.sample_id]}

    cache_path = tmp_path / "population.json"
    cold = resolve_segmentation_roi_population(
        cache_path,
        records,
        target_fn,
        num_classes=2,
        target_identity={"ordered_labels": ["negative", "positive"]},
        workers=2,
    )

    def fail_if_read(_record: SampleRecord) -> dict[str, torch.Tensor]:
        raise AssertionError("a warm population must not reread masks")

    warm = resolve_segmentation_roi_population(
        cache_path,
        records,
        fail_if_read,
        num_classes=2,
        target_identity={"ordered_labels": ["negative", "positive"]},
    )

    assert sorted(calls) == ["a", "b"]
    assert cold == warm
    assert (warm.sample_ids, warm.class_pixel_counts) == (
        ("a", "b"),
        ((2, 1), (1, 3)),
    )
    assert cold.provenance() == warm.provenance() == {
        "artifact_kind": "segmentation_roi_population",
        "cache_key": cold.cache_key,
        "cache_path": str(cold.cache_path),
        "payload_sha256": cold.payload_sha256,
        "roi_count": 2,
        "num_classes": 2,
        "class_pixel_totals": [3, 4],
    }
    assert len(cold.cache_key) == 16
    assert len(cold.payload_sha256) == 64
    assert cold.cache_path.is_file()


def test_population_subset_aligns_counts_to_requested_sample_order(tmp_path: Path) -> None:
    records = [_record("a"), _record("b")]
    masks = {
        "a": torch.tensor([[0, 0], [1, 255]]),
        "b": torch.tensor([[1, 1], [1, 0]]),
    }
    population = resolve_segmentation_roi_population(
        tmp_path / "population.json",
        records,
        lambda record: {"mask": masks[record.sample_id]},
        num_classes=2,
        target_identity={"ordered_labels": ["negative", "positive"]},
    )

    subset = population.subset(["b", "a"])

    assert (subset.sample_ids, subset.class_pixel_counts) == (
        ("b", "a"),
        ((1, 3), (2, 1)),
    )


def test_concurrent_cold_resolvers_compute_each_roi_once(tmp_path: Path) -> None:
    records = [_record("a"), _record("b")]
    masks = {
        "a": torch.tensor([[0, 0], [1, 255]]),
        "b": torch.tensor([[1, 1], [1, 0]]),
    }
    barrier = threading.Barrier(2)
    calls: list[str] = []
    calls_lock = threading.Lock()

    def target_fn(record: SampleRecord) -> dict[str, torch.Tensor]:
        with calls_lock:
            calls.append(record.sample_id)
        time.sleep(0.01)
        return {"mask": masks[record.sample_id]}

    cache_path = tmp_path / "population.json"

    def resolve():
        barrier.wait()
        return resolve_segmentation_roi_population(
            cache_path,
            records,
            target_fn,
            num_classes=2,
            target_identity={"ordered_labels": ["negative", "positive"]},
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        populations = list(executor.map(lambda _: resolve(), range(2)))

    assert populations[0] == populations[1]
    assert sorted(calls) == ["a", "b"]


def test_changed_target_identity_does_not_reuse_stale_counts(tmp_path: Path) -> None:
    records = [_record("a")]
    calls: list[str] = []

    def resolve(identity: str, mask: torch.Tensor):
        def target_fn(record: SampleRecord) -> dict[str, torch.Tensor]:
            calls.append(f"{identity}:{record.sample_id}")
            return {"mask": mask}

        return resolve_segmentation_roi_population(
            tmp_path / "populations",
            records,
            target_fn,
            num_classes=2,
            target_identity={"ordered_labels": [identity, "positive"]},
        )

    first = resolve("negative", torch.tensor([[0, 0], [1, 255]]))
    changed = resolve("other", torch.tensor([[1, 1], [1, 0]]))

    assert calls == ["negative:a", "other:a"]
    assert first.class_pixel_counts == ((2, 1),)
    assert changed.class_pixel_counts == ((1, 3),)


def test_warm_population_realigns_to_new_fold_record_order(tmp_path: Path) -> None:
    records = [_record("a"), _record("b")]
    masks = {
        "a": torch.tensor([[0, 0], [1, 255]]),
        "b": torch.tensor([[1, 1], [1, 0]]),
    }
    identity = {"ordered_labels": ["negative", "positive"]}
    resolve_segmentation_roi_population(
        tmp_path / "populations",
        records,
        lambda record: {"mask": masks[record.sample_id]},
        num_classes=2,
        target_identity=identity,
    )

    warm = resolve_segmentation_roi_population(
        tmp_path / "populations",
        list(reversed(records)),
        lambda _record: (_ for _ in ()).throw(AssertionError("must reuse")),
        num_classes=2,
        target_identity=identity,
    )

    assert (warm.sample_ids, warm.class_pixel_counts) == (
        ("b", "a"),
        ((1, 3), (2, 1)),
    )


def test_changed_mask_source_does_not_reuse_stale_counts(tmp_path: Path) -> None:
    mask_path = tmp_path / "mask.tif"
    mask_path.write_bytes(b"first")
    original_mtime_ns = mask_path.stat().st_mtime_ns
    records = [replace(_record("a"), label_mask_path=mask_path)]
    calls: list[str] = []

    def resolve(mask: torch.Tensor):
        def target_fn(record: SampleRecord) -> dict[str, torch.Tensor]:
            calls.append(record.sample_id)
            return {"mask": mask}

        return resolve_segmentation_roi_population(
            tmp_path / "populations",
            records,
            target_fn,
            num_classes=2,
            target_identity={"ordered_labels": ["negative", "positive"]},
        )

    first = resolve(torch.tensor([[0, 0], [1, 255]]))
    mask_path.write_bytes(b"other")
    os.utime(mask_path, ns=(original_mtime_ns, original_mtime_ns))
    corrected = resolve(torch.tensor([[1, 1], [1, 0]]))

    assert calls == ["a", "a"]
    assert first.class_pixel_counts == ((2, 1),)
    assert corrected.class_pixel_counts == ((1, 3),)


def test_source_fingerprints_are_reused_within_one_training_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mask_path = tmp_path / "mask.tif"
    mask_path.write_bytes(b"mask bytes")
    records = [replace(_record("a"), label_mask_path=mask_path)]
    fingerprint_calls: list[str] = []
    real_fingerprint = population_module._mask_source_fingerprint

    def track_fingerprint(path: str):
        fingerprint_calls.append(path)
        return real_fingerprint(path)

    monkeypatch.setattr(population_module, "_mask_source_fingerprint", track_fingerprint)
    source_fingerprint_cache: dict[str, dict[str, object]] = {}
    for identity in ("first", "second"):
        resolve_segmentation_roi_population(
            tmp_path / "populations",
            records,
            lambda _record: {"mask": torch.tensor([[0, 1]])},
            num_classes=2,
            target_identity={"identity": identity},
            source_fingerprint_cache=source_fingerprint_cache,
        )

    assert fingerprint_calls == [str(mask_path)]


def test_resolved_mask_backend_changes_population_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    mask_path = tmp_path / "mask.tif"
    mask_path.write_bytes(b"mask bytes")
    records = [
        replace(
            _record("a"),
            label_mask_path=mask_path,
            region=(0, 0, 2, 1),
            spacing_at_level_0=0.5,
        )
    ]
    selected_backend = "openslide"

    def resolve_backend(*_args, **_kwargs):
        return type("Selection", (), {"backend": selected_backend})()

    monkeypatch.setattr(population_module, "resolve_backend", resolve_backend)
    kwargs = {
        "cache_root": tmp_path / "populations",
        "records": records,
        "target_fn": lambda _record: {"mask": torch.tensor([[0, 1]])},
        "num_classes": 2,
        "target_identity": {"backend": "auto", "spacing_um": 0.5},
    }

    openslide = resolve_segmentation_roi_population(**kwargs)
    selected_backend = "asap"
    asap = resolve_segmentation_roi_population(**kwargs)

    assert openslide.cache_key != asap.cache_key
