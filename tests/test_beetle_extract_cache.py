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

    class FakePipeline:
        def __init__(self, config):
            self.dataset = SimpleNamespace(sample_ids=["slide-a", "slide-b", "slide-c"])

        def _build_slide_manifest_dense_context(self, *, run_dir):
            return SimpleNamespace(
                feature_store=FakeStore(),
                dataset=SimpleNamespace(
                    sample_ids=["roi-a", "roi-b"],
                    samples={
                        "roi-a": SimpleNamespace(slide_id="slide-a"),
                        "roi-b": SimpleNamespace(slide_id="slide-b"),
                    },
                ),
            )

    monkeypatch.setattr(extract_cache, "load_config", lambda _path: object())
    monkeypatch.setattr(extract_cache, "Pipeline", FakePipeline)
    output = tmp_path / "cache_extraction.json"
    config_path = tmp_path / "uniform.yaml"
    config_path.write_text("fixture: true\n", encoding="utf-8")

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
