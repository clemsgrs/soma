"""Migrating a legacy dense cache (keyed on the registry ``input_size``) to its new key.

Fully offline: the legacy cache is synthetic, and the migration reads recorded metadata
only. The legacy key is a literal recorded on main (9f4e589), not recomputed here.
"""

from __future__ import annotations

import importlib.util
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest
import torch

from soma.cache import (
    record_feature_dim,
    record_sample_identity_signatures,
    resolve_dense_cache,
)
from soma.cache.keys import _sample_stems_for_kind
from soma.cache.migration import MIGRATION_NOTE_NAME, migrate_legacy_dense_caches
from soma.config import EncoderConfig, PreprocessingConfig
from soma.dataset import Dataset
from soma.dense import compute_dense_geometry, dense_grid_metadata, write_dense_grid

# build_dense_cache_key on main for the configuration below, registry input_size=224.
LEGACY_KEY = "9ab79e2cb15c8c49"
LEGACY_INPUT_SIZE = 224
CACHE_KIND = "dense_image"

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "migrate_dense_cache_keys.py"


class _StubRegistry:
    def __init__(self, input_size: int) -> None:
        self._input_size = input_size

    def info(self, name: str) -> dict:
        return {"input_size": self._input_size, "supported_spacing_um": 0.5, "precision": "fp16"}


def _stub_registry(monkeypatch: pytest.MonkeyPatch, input_size: int) -> None:
    monkeypatch.setattr("soma.cache.keys.encoder_registry", _StubRegistry(input_size))


def _dataset(tmp_path: Path) -> Dataset:
    csv_path = tmp_path / "dataset.csv"
    pd.DataFrame(
        [
            {"sample_id": "s1", "image_path": "/tiles/s1.png", "mask_path": "/masks/s1.png", "label": "a"},
            {"sample_id": "s2", "image_path": "/tiles/s2.png", "mask_path": None, "label": "b"},
        ]
    ).to_csv(csv_path, index=False)
    return Dataset(csv_path)


def _resolve(cache_root: Path, dataset: Dataset):
    return resolve_dense_cache(
        cache_root=cache_root,
        dataset=dataset,
        tile_encoder_name="stub-encoder",
        target_size=(64, 64),
        patch_size=(16, 16),
        pad_mode="reflect",
        execution=EncoderConfig(name="stub-encoder", precision="fp16"),
        preprocessing=PreprocessingConfig(requested_spacing_um=0.5),
        dense_input_mode="sliding_window",
        window_size=32,
        overlap=0.5,
        dtype="fp16",
        cache_kind=CACHE_KIND,
    )


def _make_legacy_cache(cache_root: Path, dataset: Dataset) -> Path:
    """A complete dense cache as main wrote it: legacy folder name, key, execution, identities."""
    written = _resolve(cache_root, dataset)
    geometry = compute_dense_geometry(target_size=(64, 64), patch_size=16)
    for sample_id in dataset.sample_ids:
        sidecar = dense_grid_metadata(geometry, feature_dim=8, pad_mode="reflect")
        write_dense_grid(written.features_dir, sample_id, torch.randn(8, 4, 4), sidecar)
    record_feature_dim(written, 8)
    record_sample_identity_signatures(written, list(dataset.sample_ids))

    metadata = json.loads(written.metadata_path.read_text())
    metadata["cache_key"] = LEGACY_KEY
    metadata["execution"]["input_size"] = LEGACY_INPUT_SIZE
    metadata["sample_identity_signature_by_id"] = _sample_stems_for_kind(
        dataset=dataset,
        cache_kind=CACHE_KIND,
        static_identity_payload={"cache_key": LEGACY_KEY},
    )
    written.metadata_path.write_text(json.dumps(metadata))
    legacy_dir = written.cache_dir.with_name(LEGACY_KEY)
    written.cache_dir.rename(legacy_dir)
    return legacy_dir


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def dataset(tmp_path: Path) -> Dataset:
    return _dataset(tmp_path)


@pytest.fixture
def legacy_dir(cache_root: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch) -> Path:
    _stub_registry(monkeypatch, LEGACY_INPUT_SIZE)
    return _make_legacy_cache(cache_root, dataset)


def test_migrated_cache_sits_at_the_key_new_code_resolves_and_loads_as_a_hit(
    cache_root: Path, dataset: Dataset, legacy_dir: Path, monkeypatch: pytest.MonkeyPatch
):
    [result] = migrate_legacy_dense_caches(cache_root, apply=True)

    # The registry default moved, as it did in slide2vec 6.0.0: the migrated cache must
    # resolve regardless.
    _stub_registry(monkeypatch, 518)
    resolved = _resolve(cache_root, dataset)
    assert result.status == "migrated"
    assert result.old_key == LEGACY_KEY
    assert result.target == resolved.cache_dir
    assert resolved.cache_dir.name == resolved.key == result.new_key
    assert resolved.complete is True
    assert resolved.missing_sample_ids() == []
    assert not legacy_dir.exists()


def test_migration_patches_the_recorded_key_and_execution(cache_root: Path, legacy_dir: Path):
    [result] = migrate_legacy_dense_caches(cache_root, apply=True)

    metadata = json.loads((result.target / "cache_metadata.json").read_text())
    assert metadata["cache_key"] == result.new_key
    assert metadata["execution"] == {"precision": "fp16", "spacing_um": 0.5}


def test_migration_writes_a_provenance_note(cache_root: Path, legacy_dir: Path):
    [result] = migrate_legacy_dense_caches(cache_root, apply=True, today=date(2026, 9, 29))

    note = json.loads((result.target / MIGRATION_NOTE_NAME).read_text())
    assert note["old_key"] == LEGACY_KEY
    assert note["new_key"] == result.new_key
    assert note["migrated_on"] == "2026-09-29"


def test_migration_is_a_dry_run_by_default(cache_root: Path, legacy_dir: Path):
    before = _snapshot(cache_root)

    [result] = migrate_legacy_dense_caches(cache_root)

    assert result.status == "planned"
    assert result.old_key == LEGACY_KEY
    assert result.target == legacy_dir.with_name(result.new_key)
    assert _snapshot(cache_root) == before
    assert not result.target.exists()


def test_migration_refuses_to_overwrite_an_existing_target(
    cache_root: Path, dataset: Dataset, legacy_dir: Path
):
    # New code already ran once and created (empty) the folder the migration targets.
    occupied = _resolve(cache_root, dataset)
    before = _snapshot(cache_root)

    [result] = migrate_legacy_dense_caches(cache_root, apply=True)

    assert result.status == "refused"
    assert result.target == occupied.cache_dir
    assert "exists" in result.reason
    assert legacy_dir.is_dir()
    assert _snapshot(cache_root) == before


def test_migration_refuses_a_cache_whose_recorded_key_it_cannot_reproduce(
    cache_root: Path, legacy_dir: Path
):
    # Metadata that does not hash to the recorded key is not a cache this migration
    # understands: renaming it would assign a key to features of unknown origin.
    metadata_path = legacy_dir / "cache_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["overlap"] = 0.25
    metadata_path.write_text(json.dumps(metadata))
    before = _snapshot(cache_root)

    [result] = migrate_legacy_dense_caches(cache_root, apply=True)

    assert result.status == "refused"
    assert _snapshot(cache_root) == before


def test_migration_leaves_current_caches_alone(
    cache_root: Path, dataset: Dataset, monkeypatch: pytest.MonkeyPatch
):
    _stub_registry(monkeypatch, 518)
    _resolve(cache_root, dataset)
    before = _snapshot(cache_root)

    assert migrate_legacy_dense_caches(cache_root, apply=True) == []
    assert _snapshot(cache_root) == before


def _load_script():
    spec = importlib.util.spec_from_file_location("migrate_dense_cache_keys", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_script_is_a_dry_run_unless_told_to_apply(
    cache_root: Path, legacy_dir: Path, capsys: pytest.CaptureFixture
):
    script = _load_script()

    assert script.main([str(cache_root)]) == 0
    assert legacy_dir.is_dir()
    assert LEGACY_KEY in capsys.readouterr().out

    assert script.main([str(cache_root), "--apply"]) == 0
    assert not legacy_dir.exists()


def test_script_exits_nonzero_when_a_cache_is_refused(
    cache_root: Path, dataset: Dataset, legacy_dir: Path
):
    _resolve(cache_root, dataset)

    assert _load_script().main([str(cache_root), "--apply"]) == 1
