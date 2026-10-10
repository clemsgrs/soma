from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

import soma.training.segmentation_roi_population as population_module
from soma.data._legacy import LegacyRecord
from soma.training.segmentation_roi_population import (
    resolve_segmentation_roi_population,
)


def _record(sample_id: str) -> LegacyRecord:
    return LegacyRecord(sample_id=sample_id, image_path=Path(f"{sample_id}.tif"), )


def test_population_is_computed_once_and_reused_exactly(tmp_path: Path) -> None:
    records = [_record("a"), _record("b")]
    calls: list[str] = []
    masks = {
        "a": torch.tensor([[0, 0], [1, 255]]),
        "b": torch.tensor([[1, 1], [1, 0]]),
    }

    def target_fn(record: LegacyRecord) -> dict[str, torch.Tensor]:
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

    def fail_if_read(_record: LegacyRecord) -> dict[str, torch.Tensor]:
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

    def target_fn(record: LegacyRecord) -> dict[str, torch.Tensor]:
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
        def target_fn(record: LegacyRecord) -> dict[str, torch.Tensor]:
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
        def target_fn(record: LegacyRecord) -> dict[str, torch.Tensor]:
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
    records = [replace(_record("a"), label_mask_path=mask_path, spacing_at_level_0=0.5)]
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


def test_mask_backend_resolves_like_hs2p_mask_without_requiring_spacing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """hs2p 5 opens a mask without spacing metadata (untagged TIFF), so the population
    key must resolve its backend the same way instead of failing on missing spacing."""
    seen = []

    def resolve_backend(backend, **kwargs):
        seen.append((backend, kwargs))
        return type("Selection", (), {"backend": "openslide"})()

    monkeypatch.setattr(population_module, "resolve_backend", resolve_backend)
    mask_path = tmp_path / "untagged_mask.tif"

    backend = population_module._resolved_mask_backend(
        str(mask_path), requested_backend="auto", spacing_aware=True
    )

    assert backend == "openslide"
    assert seen == [("auto", {"wsi_path": mask_path, "require_spacing": False})]


def test_mask_reader_schema_tracks_grid_registered_roi_mask_reads() -> None:
    # hs2p 5 aligns masks to their slide and resamples by coordinate (2); ROI masks are
    # then read at their grid's recorded spacing, with any overhang ignored (3), and
    # finally from the crops ROI sampling stores (4). A population counted by an
    # earlier reader must not be reused.
    assert population_module._MASK_READER_SCHEMA_VERSION == 4


def _roi_record(tmp_path: Path) -> LegacyRecord:
    raster = tmp_path / "annotation.tif"
    raster.write_bytes(b"raster")
    crop = tmp_path / "masks" / "s0" / "0_0.png"
    crop.parent.mkdir(parents=True)
    crop.write_bytes(b"crop")
    return replace(
        _record("s0__x0_y0"),
        label_mask_path=raster,
        region=(0, 0),
        spacing_at_level_0=0.5,
        slide_id="s0",
        label_mask_crop_path=crop,
    )


def _counting_resolve(tmp_path: Path, records, calls: list[str]):
    def target_fn(record: LegacyRecord) -> dict[str, torch.Tensor]:
        calls.append(record.sample_id)
        return {"mask": torch.tensor([[0, 1]])}

    return resolve_segmentation_roi_population(
        tmp_path / "populations",
        records,
        target_fn,
        num_classes=2,
        target_identity={"backend": "auto", "spacing_um": 0.5},
    )


def test_replaced_roi_mask_crop_does_not_reuse_stale_counts(tmp_path: Path) -> None:
    """A slide-manifest ROI's target is its stored crop: replacing that crop (e.g. one a
    hand-built ROI dataset supplies) recounts it."""
    record = _roi_record(tmp_path)
    calls: list[str] = []

    first = _counting_resolve(tmp_path, [record], calls)
    record.label_mask_crop_path.write_bytes(b"replaced crop")
    second = _counting_resolve(tmp_path, [record], calls)

    assert calls == [record.sample_id, record.sample_id]
    assert first.cache_key != second.cache_key


def test_roi_crop_path_is_part_of_the_population_identity(tmp_path: Path) -> None:
    record = _roi_record(tmp_path)
    other_crop = tmp_path / "other" / "0_0.png"
    other_crop.parent.mkdir()
    other_crop.write_bytes(b"crop")
    calls: list[str] = []

    first = _counting_resolve(tmp_path, [record], calls)
    second = _counting_resolve(
        tmp_path, [replace(record, label_mask_crop_path=other_crop)], calls
    )

    assert first.cache_key != second.cache_key


def test_roi_population_identity_never_reads_the_annotation_raster(
    tmp_path: Path, monkeypatch
) -> None:
    """ROI targets never open the annotation raster, so neither does their population
    key: a warm launch does not hash a multi-GB raster or resolve its reader backend."""
    record = _roi_record(tmp_path)
    hashed: list[Path] = []
    real_sha256 = population_module._sha256_file

    def tracking_sha256(path: Path) -> str:
        hashed.append(Path(path))
        return real_sha256(path)

    def no_backend(*_args, **_kwargs):
        raise AssertionError("a ROI population must not resolve the raster's backend")

    monkeypatch.setattr(population_module, "_sha256_file", tracking_sha256)
    monkeypatch.setattr(population_module, "resolve_backend", no_backend)
    calls: list[str] = []

    _counting_resolve(tmp_path, [record], calls)
    record.label_mask_path.write_bytes(b"edited raster")
    _counting_resolve(tmp_path, [record], calls)

    assert calls == [record.sample_id]
    assert record.label_mask_path.resolve() not in {path.resolve() for path in hashed}
