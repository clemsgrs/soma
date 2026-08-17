from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch


def _load_parity_module():
    # Import by file path: sibling checkouts (../hs2p) put a regular top-level
    # `scripts` package on sys.path that shadows soma's namespace `scripts/`.
    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_dense_multigpu_parity.py"
    spec = importlib.util.spec_from_file_location("soma_verify_dense_multigpu_parity", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_parity = _load_parity_module()
Comparison = _parity.Comparison
_compare_gpu_runs = _parity._compare_gpu_runs


@dataclass(frozen=True)
class _Artifact:
    sample_id: str
    path: Path
    metadata_path: Path


def _artifact(
    root: Path,
    run: str,
    *,
    sample_id: str = "sample-a",
    grid: torch.Tensor | None = None,
    metadata: dict | None = None,
) -> _Artifact:
    artifact_root = root / run
    artifact_root.mkdir()
    path = artifact_root / "grid.pt"
    metadata_path = artifact_root / "grid.meta.json"
    torch.save(torch.ones((1, 1, 1)) if grid is None else grid, path)
    metadata_path.write_text(
        json.dumps({"spacing_um": 0.5} if metadata is None else metadata),
        encoding="utf-8",
    )
    return _Artifact(sample_id=sample_id, path=path, metadata_path=metadata_path)


def test_compare_gpu_runs_returns_exact_semantic_and_numerical_summary(tmp_path: Path):
    one_gpu = [_artifact(tmp_path, "one-gpu")]
    two_gpu = [_artifact(tmp_path, "two-gpu")]

    actual = _compare_gpu_runs(
        one_gpu,
        two_gpu,
        semantic_key=lambda artifact: artifact.sample_id,
    )

    assert actual == Comparison(
        semantic_order=["sample-a"],
        shapes=[[1, 1, 1]],
        dtype="float32",
        minimum_cosine=1.0,
        maximum_cosine_distance=0.0,
        maximum_absolute_delta=0.0,
        mean_absolute_delta=0.0,
    )


def test_compare_gpu_runs_rejects_semantic_order_mismatch(tmp_path: Path):
    one_gpu = [_artifact(tmp_path, "one-gpu", sample_id="sample-a")]
    two_gpu = [_artifact(tmp_path, "two-gpu", sample_id="sample-b")]

    with pytest.raises(AssertionError):
        _compare_gpu_runs(one_gpu, two_gpu, semantic_key=lambda artifact: artifact.sample_id)


@pytest.mark.parametrize(
    "two_gpu_grid",
    [torch.ones((1, 1, 2)), torch.ones((1, 1, 1), dtype=torch.float64)],
    ids=["shape", "dtype"],
)
def test_compare_gpu_runs_rejects_tensor_contract_mismatch(
    tmp_path: Path, two_gpu_grid: torch.Tensor
):
    one_gpu = [_artifact(tmp_path, "one-gpu")]
    two_gpu = [_artifact(tmp_path, "two-gpu", grid=two_gpu_grid)]

    with pytest.raises(AssertionError):
        _compare_gpu_runs(one_gpu, two_gpu, semantic_key=lambda artifact: artifact.sample_id)


def test_compare_gpu_runs_rejects_metadata_mismatch(tmp_path: Path):
    one_gpu = [_artifact(tmp_path, "one-gpu")]
    two_gpu = [_artifact(tmp_path, "two-gpu", metadata={"spacing_um": 1.0})]

    with pytest.raises(AssertionError):
        _compare_gpu_runs(one_gpu, two_gpu, semantic_key=lambda artifact: artifact.sample_id)


def test_compare_gpu_runs_rejects_nonfinite_values(tmp_path: Path):
    one_gpu = [_artifact(tmp_path, "one-gpu")]
    two_gpu = [_artifact(tmp_path, "two-gpu", grid=torch.full((1, 1, 1), float("nan")))]

    with pytest.raises(AssertionError):
        _compare_gpu_runs(one_gpu, two_gpu, semantic_key=lambda artifact: artifact.sample_id)


def test_compare_gpu_runs_rejects_cosine_below_contract(tmp_path: Path):
    one_gpu = [_artifact(tmp_path, "one-gpu", grid=torch.tensor([[[1.0, 0.0]]]))]
    two_gpu = [_artifact(tmp_path, "two-gpu", grid=torch.tensor([[[0.0, 1.0]]]))]

    with pytest.raises(AssertionError):
        _compare_gpu_runs(one_gpu, two_gpu, semantic_key=lambda artifact: artifact.sample_id)
