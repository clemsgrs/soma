from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from examples.beetle import extract_cache


def test_extract_cache_prepares_features_without_starting_training(
    tmp_path: Path, monkeypatch
) -> None:
    payload_dir = tmp_path / "cache" / "dense_embeddings"
    payload_dir.mkdir(parents=True)
    for stem in ("slide-a/0_0", "slide-b/4_8"):
        path = payload_dir / f"{stem}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"tensor")
        path.with_suffix(".meta.json").write_text("{}", encoding="utf-8")

    class FakeStore:
        feature_dir = payload_dir
        available_samples = ["roi-a", "roi-b"]
        feature_dim = 1280
        grid_shape = (37, 37)

        def validate_coverage(self, sample_ids):
            assert sample_ids == ["roi-a", "roi-b"]

    class FakeFeatureExtractor:
        def __init__(self, dataset, encoder, preprocessing, **kwargs):
            assert dataset.sample_ids == ["slide-a", "slide-b", "slide-c"]

        def extract(self):
            return SimpleNamespace(
                source=FakeStore(),
                dataset=SimpleNamespace(
                    sample_ids=["roi-a", "roi-b"],
                    samples={
                        "roi-a": SimpleNamespace(slide_id="slide-a"),
                        "roi-b": SimpleNamespace(slide_id="slide-b"),
                    },
                ),
                provenance=SimpleNamespace(zero_roi_sample_ids=("slide-c",)),
            )

    output = tmp_path / "cache_extraction.json"
    config_path = tmp_path / "uniform.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")
    config = SimpleNamespace(
        dataset_csv=tmp_path / "dataset.csv",
        dataset_type="segmentation",
        encoder=object(),
        preprocessing=object(),
        execution=object(),
        cache=SimpleNamespace(root_dir=tmp_path / "cache"),
        output_root=tmp_path / "output",
    )
    parent_dataset = SimpleNamespace(sample_ids=["slide-a", "slide-b", "slide-c"])
    monkeypatch.setattr(extract_cache, "load_config", lambda _path: config)
    monkeypatch.setattr(extract_cache, "load_manifest", lambda _path, _type: parent_dataset)
    monkeypatch.setattr(extract_cache, "resolve_pipeline_preprocessing", lambda _config: object())
    monkeypatch.setattr(extract_cache, "FeatureExtractor", FakeFeatureExtractor)

    payload = extract_cache.extract_verified_cache(
        config_path=config_path,
        work_dir=tmp_path / "work",
        output_path=output,
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["status"] == "completed"
    assert payload["parent_slides"] == 3
    assert payload["roi_grids"] == 2
    assert payload["slides_with_rois"] == 2
    assert payload["zero_roi_slides"] == ["slide-c"]
    assert payload["tensor_files"] == 2
    assert payload["sidecar_files"] == 2
    assert payload["payload_bytes"] == 12
    assert payload["feature_dim"] == 1280
    assert payload["grid_shape"] == [37, 37]
