from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.beetle import preflight
from examples.beetle.launch import FrozenEncoderSource


def test_campaign_preflight_writes_launcher_schema_and_observations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    revision = "a" * 40
    snapshot = tmp_path / "models--paige-ai--Virchow2" / "snapshots" / revision
    source = FrozenEncoderSource(
        snapshot_path=snapshot,
        revision=revision,
        weight_sha256="b" * 64,
    )
    encoder = {
        "repository": "paige-ai/Virchow2",
        "revision": revision,
        "weight_file": "model.safetensors",
        "weight_sha256": "b" * 64,
        "patch_size": 14,
        "feature_channels": 1280,
        "weight_checksum_verified": True,
        "snapshot_path": str(snapshot),
    }
    monkeypatch.setattr(
        preflight, "validate_frozen_encoder_snapshot", lambda _path: (encoder, source)
    )
    monkeypatch.setattr(
        preflight,
        "bind_frozen_encoder",
        lambda _source, _output: tmp_path / "bound-hub",
    )
    monkeypatch.setattr(
        preflight,
        "collect_node_observations",
        lambda **_kwargs: {
            "cuda_available": True,
            "gpu_count": 2,
            "gpus": [{"index": 0}, {"index": 1}],
            "storage": {"free_bytes": 10_000},
        },
    )
    monkeypatch.setattr(
        preflight,
        "run_representative_extraction_parity",
        lambda **_kwargs: SimpleNamespace(
            to_dict=lambda: {
                "status": "passed",
                "minimum_cosine_similarity": 0.99999,
                "representatives": [{"sample_id": "ordinary"}],
            }
        ),
    )
    monkeypatch.setattr(
        preflight,
        "probe_decoder_batch_candidates",
        lambda candidates, **_kwargs: {
            "batch_size_candidates": candidates,
            "batch_size_attempts": [
                {"batch_size": value, "passed": value <= 32} for value in candidates
            ],
            "selected_batch_size": 32,
        },
    )
    output = tmp_path / "hardware_preflight.json"

    payload = preflight.run_campaign_preflight(
        snapshot_path=snapshot,
        dataset_csv=tmp_path / "dataset.csv",
        splits_csv=tmp_path / "splits.csv",
        batch_size_candidates=[64, 32, 16, 8, 4],
        output_path=output,
        device="cuda:0",
    )

    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert payload["schema_version"] == 1
    assert payload["status"] == "completed"
    assert payload["scope"] == "campaign"
    assert payload["batch_size_candidates"] == [64, 32, 16, 8, 4]
    assert payload["selected_batch_size"] == 32
    assert payload["same_batch_every_arm_and_fold"] is True
    assert payload["encoder"] == encoder
    assert payload["node"]["gpu_count"] == 2
    assert payload["representative_extraction"]["status"] == "passed"
    assert payload["decoder_probe_device"] == "cuda:0"
    assert payload["encoder_batch_size"] == 8


def test_production_cli_accepts_the_approved_candidate_list(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def fake_run(**kwargs):
        captured.update(kwargs)
        return {"status": "completed"}

    monkeypatch.setattr(preflight, "run_campaign_preflight", fake_run)

    assert preflight.main(
        [
            "--snapshot-path",
            "/locked/snapshot",
            "--batch-size-candidates",
            "64",
            "32",
            "16",
            "8",
            "4",
            "--output",
            str(tmp_path / "hardware_preflight.json"),
        ]
    ) == 0
    assert captured["batch_size_candidates"] == [64, 32, 16, 8, 4]
    assert captured["device"] == "cuda:0"
    assert captured["dataset_csv"] == Path(
        "data/beetle/curated_slide_manifest/dataset.csv"
    )
    assert captured["splits_csv"] == Path(
        "data/beetle/curated_slide_manifest/splits.csv"
    )
