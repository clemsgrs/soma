from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch


COARSE_IDS = (
    "TCGA-OL-A66I-01Z-00-DX1.8CE9DCAB-98D3-4163-94AC-1557D86C1E25",
    "TCGA-OL-A66P-01Z-00-DX1.5ADD0D6D-37C6-4BC9-8C2B-64DB18BE99B3",
    "TCGA-OL-A6VO-01Z-00-DX1.291D54D6-EBAF-4622-BD42-97AA5997F014",
)


def _write_manifest(tmp_path: Path) -> tuple[Path, Path]:
    rows = []
    for sample_id, read_policy, spacing in (
        ("ordinary-empty", "spacing_aware", None),
        ("ordinary-eligible", "spacing_aware", None),
        *((sample_id, "native_level_0_no_upsample", 0.657476) for sample_id in COARSE_IDS),
    ):
        rows.append(
            {
                "sample_id": sample_id,
                "image_path": str(tmp_path / f"{sample_id}.tif"),
                "label_mask_path": str(tmp_path / f"{sample_id}.mask.tif"),
                "patient_id": sample_id,
                "spacing_at_level_0": spacing,
                "read_policy": read_policy,
            }
        )
    dataset_csv = tmp_path / "dataset.csv"
    pd.DataFrame(rows).to_csv(dataset_csv, index=False)
    splits_csv = tmp_path / "splits.csv"
    pd.DataFrame(
        [
            {"sample_id": row["sample_id"], "split": "train", "fold": 0}
            for row in rows
        ]
    ).to_csv(splits_csv, index=False)
    return dataset_csv, splits_csv


class _FakeExtractor:
    calls: list[dict] = []
    negate_fp16 = False

    def __init__(self, roi_dataset, encoder, **kwargs):
        self.roi_dataset = roi_dataset
        self.encoder = encoder
        self.kwargs = kwargs

    def run(self, feature_dir):
        dtype = torch.float16 if self.kwargs["cache"].dtype == "fp16" else torch.float32
        payload = Path(feature_dir) / "dense_embeddings"
        for record in self.roi_dataset.samples.values():
            x, y = record.region
            sample_dir = payload / record.slide_id
            sample_dir.mkdir(parents=True, exist_ok=True)
            base = torch.arange(1280 * 37 * 37, dtype=torch.float32).reshape(1280, 37, 37)
            grid = (base / 4096).to(dtype)
            if dtype == torch.float16 and self.negate_fp16:
                grid = -grid
            torch.save(grid, sample_dir / f"{x}_{y}.pt")
            declared_spacing = max(0.5, record.spacing_at_level_0 or 0.5)
            effective_spacing = record.spacing_at_level_0 or 0.485
            metadata = {
                "artifact_type": "dense_embeddings",
                "sample_id": record.slide_id,
                "x": x,
                "y": y,
                "dtype": str(dtype).removeprefix("torch."),
                "feature_dim": 1280,
                "grid_shape": [37, 37],
                "target_size": [512, 512],
                "patch_size": [14, 14],
                "encoded_size": [518, 518],
                "pad": [6, 6],
                "declared_spacing_um": declared_spacing,
                "source_spacing_um": record.spacing_at_level_0 or 0.485,
                "read_spacing_um": effective_spacing,
                "effective_spacing_um": effective_spacing,
                "spacing_at_level_0": record.spacing_at_level_0,
                "read_level": 0,
                "read_size": [512, 512],
                "output_size": [512, 512],
                "is_within_tolerance": True,
                "window_size": 224,
                "overlap": 0.5,
                "feature_kind": "patch_features",
            }
            (sample_dir / f"{x}_{y}.meta.json").write_text(json.dumps(metadata))
        self.calls.append(
            {
                "sample_ids": list(self.roi_dataset.sample_ids),
                "encoder": self.encoder,
                **self.kwargs,
            }
        )
        return SimpleNamespace(feature_dir=payload)


def test_representative_extraction_uses_exact_coarse_slides_and_first_eligible_ordinary(
    tmp_path: Path, monkeypatch
) -> None:
    from examples.beetle import preflight_extraction as module

    dataset_csv, splits_csv = _write_manifest(tmp_path)
    sampled: list[list[str]] = []

    def fake_sample(dataset, *, sample_ids, **_kwargs):
        ids = list(sample_ids)
        sampled.append(ids)
        return {
            sample_id: (
                []
                if sample_id == "ordinary-empty"
                else (
                    [(96, 64), (32, 64)]
                    if sample_id in COARSE_IDS
                    else [(32, 64), (0, 128), (0, 0)]
                )
            )
            for sample_id in ids
        }

    class FakeHead:
        def __init__(self, **_kwargs):
            pass

        def extract_targets(self, record):
            return {"mask": torch.full((512, 512), 2, dtype=torch.long)}

    monkeypatch.setattr(module, "sample_slide_rois", fake_sample)
    monkeypatch.setattr(module, "SlideManifestDenseExtractor", _FakeExtractor)
    monkeypatch.setattr(module, "SegmentationHead", FakeHead)
    _FakeExtractor.calls = []
    _FakeExtractor.negate_fp16 = False

    result = module.run_representative_extraction_parity(
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_dir=tmp_path / "preflight",
        minimum_cosine_similarity=0.9999,
    ).to_dict()

    assert sampled == [
        list(COARSE_IDS),
        ["ordinary-empty"],
        ["ordinary-eligible"],
    ]
    assert result["status"] == "passed"
    assert [row["sample_id"] for row in result["representatives"]] == [
        "ordinary-eligible",
        *COARSE_IDS,
    ]
    assert result["representatives"][0]["region"] == [0, 0]
    assert all(row["mask_shape"] == [512, 512] for row in result["representatives"])
    assert all(row["mask_labels"] == [2] for row in result["representatives"])
    assert all(row["geometry"]["grid_shape"] == [37, 37] for row in result["representatives"])
    assert all(row["geometry"]["encoded_size"] == [518, 518] for row in result["representatives"])
    coarse = [row for row in result["representatives"] if row["kind"] == "native_spacing_exception"]
    assert all(row["region"] == [32, 64] for row in coarse)
    assert all(row["selection"] == "annotation_eligible_roi" for row in coarse)
    ordinary = [row for row in result["representatives"] if row["kind"] == "ordinary"]
    assert ordinary[0]["selection"] == "annotation_eligible_roi"
    assert all(row["geometry"]["read_level"] == 0 for row in coarse)
    assert all(row["geometry"]["read_size"] == [512, 512] for row in coarse)
    assert all(row["geometry"]["output_size"] == [512, 512] for row in coarse)
    assert result["resume"] == {
        "fp16": {"payloads_verified": 8, "unchanged": True},
        "fp32": {"payloads_verified": 8, "unchanged": True},
    }
    for row in result["representatives"]:
        for field in ("fp16_tensor", "fp32_tensor", "fp16_sidecar", "fp32_sidecar"):
            assert row[field]["bytes"] > 0
            assert len(row[field]["sha256"]) == 64
    assert result["minimum_cosine_similarity"] >= 0.9999
    assert len(_FakeExtractor.calls) == 4
    assert [_FakeExtractor.calls[i]["cache"].dtype for i in range(4)] == [
        "fp16",
        "fp32",
        "fp16",
        "fp32",
    ]
    assert all(
        call["preprocessing"].tolerance == pytest.approx(0.1)
        for call in _FakeExtractor.calls
    )
    assert all(
        call["preprocessing"].mask_backend == "openslide"
        for call in _FakeExtractor.calls
    )
    assert _FakeExtractor.calls[0]["cache"].root_dir != _FakeExtractor.calls[1]["cache"].root_dir


def test_representative_extraction_rejects_fp16_cache_below_cosine_contract(
    tmp_path: Path, monkeypatch
) -> None:
    from examples.beetle import preflight_extraction as module

    dataset_csv, splits_csv = _write_manifest(tmp_path)

    def fake_sample(dataset, *, sample_ids, **_kwargs):
        return {sample_id: [(0, 0)] for sample_id in sample_ids}

    class FakeHead:
        def __init__(self, **_kwargs):
            pass

        def extract_targets(self, record):
            return {"mask": torch.zeros((512, 512), dtype=torch.long)}

    monkeypatch.setattr(module, "sample_slide_rois", fake_sample)
    monkeypatch.setattr(module, "SlideManifestDenseExtractor", _FakeExtractor)
    monkeypatch.setattr(module, "SegmentationHead", FakeHead)
    _FakeExtractor.calls = []
    _FakeExtractor.negate_fp16 = True

    with pytest.raises(ValueError, match=r"fp16/fp32 cosine similarity .* is below 0\.9999"):
        module.run_representative_extraction_parity(
            dataset_csv=dataset_csv,
            splits_csv=splits_csv,
            output_dir=tmp_path / "preflight",
            minimum_cosine_similarity=0.9999,
        )


def test_parity_cosine_is_bounded_by_its_mathematical_range() -> None:
    from examples.beetle.preflight_extraction import _parity

    fp16 = torch.ones(1280 * 37 * 37, dtype=torch.float16)
    fp32 = torch.ones(1280 * 37 * 37, dtype=torch.float32)

    assert _parity(fp16, fp32).cosine_similarity == 1.0

    _FakeExtractor.negate_fp16 = False
