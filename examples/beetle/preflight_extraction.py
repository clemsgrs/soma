"""Representative real-extraction gate for the BEETLE hardware preflight."""

from __future__ import annotations

import gc
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from soma.config import (
    CacheConfig,
    EncoderConfig,
    ExecutionConfig,
    MasksConfig,
    PreprocessingConfig,
    SamplingConfig,
)
from soma import FeatureExtractor
from soma.dataset import SegmentationManifest
from soma.dense.geometry import compute_dense_geometry
from soma.dense.reader import build_label_remap
from soma.dense_slide_extraction import (
    build_roi_dataset,
    sample_slide_rois,
)
from soma.tasks.segmentation import SegmentationHead

from examples.beetle.curate import _NATIVE_LEVEL_0_EXCEPTIONS

COARSE_READ_POLICY = "native_level_0_no_upsample"
EXPECTED_COARSE_SLIDES = 3
DEFAULT_MINIMUM_COSINE_SIMILARITY = 0.9999

_PIXEL_MAPPING = {
    "background": 0,
    "other": 1,
    "non_invasive_epithelium": 2,
    "invasive_epithelium": 3,
    "necrosis": 4,
}
_MIN_COVERAGE = {
    "other": 0.05,
    "non_invasive_epithelium": 0.05,
    "invasive_epithelium": 0.05,
    "necrosis": 0.05,
}


@dataclass(frozen=True)
class ParityStatistics:
    cosine_similarity: float
    relative_l2: float
    maximum_absolute_delta: float
    mean_absolute_delta: float


@dataclass(frozen=True)
class FileEvidence:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class RepresentativeExtraction:
    sample_id: str
    kind: str
    selection: str
    roi_id: str
    region: list[int]
    mask_shape: list[int]
    mask_labels: list[int]
    geometry: dict
    fp16_tensor: FileEvidence
    fp32_tensor: FileEvidence
    fp16_sidecar: FileEvidence
    fp32_sidecar: FileEvidence
    parity: ParityStatistics


@dataclass(frozen=True)
class RepresentativeExtractionResult:
    status: str
    minimum_required_cosine_similarity: float
    minimum_cosine_similarity: float
    resume: dict[str, dict[str, int | bool]]
    representatives: tuple[RepresentativeExtraction, ...]

    def to_dict(self) -> dict:
        """Return the JSON-ready payload embedded in the hardware-preflight record."""
        return asdict(self)


def _recipe() -> tuple[PreprocessingConfig, MasksConfig, SamplingConfig]:
    masks = MasksConfig(pixel_mapping=_PIXEL_MAPPING, min_coverage=_MIN_COVERAGE)
    sampling = SamplingConfig(strategy="joint", output_mode="merged")
    preprocessing = PreprocessingConfig(
        backend="auto",
        mask_backend="openslide",
        requested_tile_size_px=512,
        requested_spacing_um=0.5,
        spacing_policy="native_if_coarser",
        tolerance=0.1,
        feature_kind="patch_features",
        dense_window_size=224,
        dense_window_overlap=0.5,
        masks=masks,
        sampling=sampling,
    )
    return preprocessing, masks, sampling


def _one_coordinate(coords: list[tuple[int, int]], *, sample_id: str) -> tuple[int, int]:
    if not coords:
        raise ValueError(f"BEETLE representative slide {sample_id!r} has no eligible ROI")
    return min((int(x), int(y)) for x, y in coords)


def _representative_coordinates(
    dataset: SegmentationManifest,
    *,
    masks: MasksConfig,
    sampling: SamplingConfig,
    preprocessing: PreprocessingConfig,
) -> tuple[dict[str, list[tuple[int, int]]], set[str]]:
    coarse_ids = [
        sample_id
        for sample_id, record in dataset.samples.items()
        if record.metadata.get("read_policy") == COARSE_READ_POLICY
    ]
    expected_ids = set(_NATIVE_LEVEL_0_EXCEPTIONS)
    if len(coarse_ids) != EXPECTED_COARSE_SLIDES or set(coarse_ids) != expected_ids:
        raise ValueError(
            "BEETLE representative extraction requires the exact three curated "
            f"native-spacing exceptions; expected {sorted(expected_ids)}, "
            f"found {sorted(coarse_ids)}"
        )
    coarse_coords = sample_slide_rois(
        dataset,
        masks=masks,
        sampling=sampling,
        preprocessing=preprocessing,
        sample_ids=coarse_ids,
    )
    missing_eligible = [sample_id for sample_id, coords in coarse_coords.items() if not coords]
    if missing_eligible:
        raise ValueError(
            "BEETLE native-spacing exceptions have no annotation-eligible ROI with the "
            "locked OpenSlide mask backend; re-audit mask decoding before proceeding: "
            f"{sorted(missing_eligible)}"
        )
    selected = {
        sample_id: [_one_coordinate(coords, sample_id=sample_id)]
        for sample_id, coords in coarse_coords.items()
    }

    ordinary_id = None
    ordinary_coord = None
    for sample_id, record in dataset.samples.items():
        if record.metadata.get("read_policy") == COARSE_READ_POLICY:
            continue
        candidate = sample_slide_rois(
            dataset,
            masks=masks,
            sampling=sampling,
            preprocessing=preprocessing,
            sample_ids=[sample_id],
        )[sample_id]
        if candidate:
            ordinary_id = sample_id
            ordinary_coord = _one_coordinate(candidate, sample_id=sample_id)
            break
    if ordinary_id is None or ordinary_coord is None:
        raise ValueError("BEETLE representative extraction found no eligible ordinary slide")

    return ({ordinary_id: [ordinary_coord], **selected}, set(coarse_ids))


def _run_precision(
    *,
    precision: str,
    roi_dataset: SegmentationManifest,
    output_dir: Path,
    preprocessing: PreprocessingConfig,
    masks: MasksConfig,
    sampling: SamplingConfig,
):
    result = FeatureExtractor(
        roi_dataset,
        EncoderConfig(
            name="virchow2",
            precision=precision,
            batch_size=8,
            adaptive_batching=True,
            output_variant="cls",
        ),
        preprocessing=preprocessing,
        execution=ExecutionConfig(
            num_gpus=1,
            num_workers_per_gpu=0,
            precision=precision,
        ),
        cache=CacheConfig(
            enabled=True,
            root_dir=output_dir / f"cache_{precision}",
            reuse_policy="strict",
            validate_payloads=True,
            dtype=precision,
        ),
        output_root=output_dir / f"features_{precision}",
    ).extract()
    store = result.source
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return store


def _payload_paths(store, record) -> tuple[Path, Path]:
    x, y = record.region
    tensor_path = Path(store.feature_dir) / str(record.slide_id) / f"{x}_{y}.pt"
    return tensor_path, tensor_path.with_suffix(".meta.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_evidence(path: Path) -> FileEvidence:
    return FileEvidence(path=str(path), bytes=path.stat().st_size, sha256=_sha256(path))


def _snapshot_payloads(
    stores: dict, roi_dataset: SegmentationManifest
) -> dict[str, dict[str, tuple[int, str]]]:
    snapshots: dict[str, dict[str, tuple[int, str]]] = {}
    for precision, store in stores.items():
        files: dict[str, tuple[int, str]] = {}
        for record in roi_dataset.samples.values():
            for path in _payload_paths(store, record):
                files[str(path)] = (path.stat().st_size, _sha256(path))
        snapshots[precision] = files
    return snapshots


def _validate_sidecar(
    metadata: dict,
    *,
    record,
    precision: str,
    preprocessing: PreprocessingConfig,
) -> None:
    x, y = record.region
    expected = {
        "artifact_type": "dense_embeddings",
        "sample_id": record.slide_id,
        "x": x,
        "y": y,
        "dtype": f"float{16 if precision == 'fp16' else 32}",
        "feature_dim": 1280,
        "grid_shape": [37, 37],
        "target_size": [512, 512],
        "patch_size": [14, 14],
        "encoded_size": [518, 518],
        "pad": [6, 6],
        "window_size": 224,
        "overlap": 0.5,
        "feature_kind": "patch_features",
    }
    mismatches = {
        key: {"expected": value, "observed": metadata.get(key)}
        for key, value in expected.items()
        if metadata.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"BEETLE representative sidecar mismatch for {record.sample_id}: {mismatches}"
        )

    expected_spacing = preprocessing.effective_spacing_um(record.spacing_at_level_0)
    declared = metadata.get("declared_spacing_um")
    if declared is None or not math.isclose(
        float(declared), expected_spacing, rel_tol=0.0, abs_tol=1e-5
    ):
        raise ValueError(
            f"BEETLE representative {record.sample_id} has declared_spacing_um={declared!r}; "
            f"expected {expected_spacing}"
        )
    effective = metadata.get("effective_spacing_um")
    if effective is None or not math.isfinite(float(effective)) or float(effective) <= 0:
        raise ValueError(
            f"BEETLE representative {record.sample_id} has invalid "
            f"effective_spacing_um={effective!r}"
        )
    # A source level already within the configured tolerance is deliberately read native,
    # so an ordinary slide may record (for example) 0.485 rather than exactly 0.5.
    relative_mismatch = abs(float(effective) - expected_spacing) / expected_spacing
    if relative_mismatch > float(preprocessing.tolerance) + 1e-9:
        raise ValueError(
            f"BEETLE representative {record.sample_id} effective spacing {effective!r} "
            f"is outside tolerance of its declared spacing {expected_spacing}"
        )
    if record.spacing_at_level_0 is not None:
        source = metadata.get("source_spacing_um")
        if source is None or not math.isclose(
            float(source), float(record.spacing_at_level_0), rel_tol=0.0, abs_tol=1e-5
        ):
            raise ValueError(
                f"BEETLE representative {record.sample_id} source spacing {source!r} "
                f"does not match manifest {record.spacing_at_level_0}"
            )
        if metadata.get("read_level") != 0:
            raise ValueError(
                f"BEETLE native-spacing representative {record.sample_id} was not read at level 0"
            )
        if metadata.get("read_size") != [512, 512] or metadata.get("output_size") != [
            512,
            512,
        ]:
            raise ValueError(f"BEETLE native-spacing representative {record.sample_id} was resized")


def _parity(fp16: torch.Tensor, fp32: torch.Tensor) -> ParityStatistics:
    left = fp16.float().reshape(-1)
    right = fp32.float().reshape(-1)
    if left.shape != right.shape:
        raise ValueError(
            "BEETLE representative fp16/fp32 shapes differ: "
            f"{tuple(left.shape)} vs {tuple(right.shape)}"
        )
    if not torch.isfinite(left).all() or not torch.isfinite(right).all():
        raise ValueError("BEETLE representative cache contains non-finite values")
    delta = (left - right).abs()
    denominator = torch.linalg.vector_norm(right).clamp_min(torch.finfo(torch.float32).eps)
    relative_l2 = torch.linalg.vector_norm(left - right) / denominator
    cosine = float(F.cosine_similarity(left, right, dim=0))
    return ParityStatistics(
        cosine_similarity=max(-1.0, min(1.0, cosine)),
        relative_l2=float(relative_l2),
        maximum_absolute_delta=float(delta.max()),
        mean_absolute_delta=float(delta.mean()),
    )


def run_representative_extraction_parity(
    *,
    dataset_csv: str | Path,
    splits_csv: str | Path,
    output_dir: str | Path,
    minimum_cosine_similarity: float = DEFAULT_MINIMUM_COSINE_SIMILARITY,
) -> RepresentativeExtractionResult:
    """Extract and compare four locked BEETLE representative ROI grids.

    The selected set is exactly the three curated native-spacing exceptions plus the
    first ordinary manifest slide with an eligible annotation-sampled ROI. One
    lexicographically smallest ROI is retained per slide. The same derived ROI manifest is
    then extracted into independent fp16 and fp32 caches through Soma's production dense
    slide extractor.
    """
    threshold = float(minimum_cosine_similarity)
    if not math.isfinite(threshold) or not 0.0 < threshold <= 1.0:
        raise ValueError("minimum_cosine_similarity must be finite and in (0, 1]")

    dataset_csv = Path(dataset_csv).resolve()
    splits_csv = Path(splits_csv).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = SegmentationManifest(dataset_csv)
    preprocessing, masks, sampling = _recipe()
    coords_by_slide, coarse_ids = _representative_coordinates(
        dataset,
        masks=masks,
        sampling=sampling,
        preprocessing=preprocessing,
    )
    roi_manifest = build_roi_dataset(
        dataset, coords_by_slide, out_dir=output_dir / "representative_rois"
    )
    roi_dataset = SegmentationManifest(roi_manifest)

    stores = {
        precision: _run_precision(
            precision=precision,
            roi_dataset=roi_dataset,
            output_dir=output_dir,
            preprocessing=preprocessing,
            masks=masks,
            sampling=sampling,
        )
        for precision in ("fp16", "fp32")
    }
    before_resume = _snapshot_payloads(stores, roi_dataset)
    resumed_stores = {
        precision: _run_precision(
            precision=precision,
            roi_dataset=roi_dataset,
            output_dir=output_dir,
            preprocessing=preprocessing,
            masks=masks,
            sampling=sampling,
        )
        for precision in ("fp16", "fp32")
    }
    after_resume = _snapshot_payloads(resumed_stores, roi_dataset)
    resume = {}
    for precision in ("fp16", "fp32"):
        unchanged = before_resume[precision] == after_resume[precision]
        if not unchanged:
            raise ValueError(
                f"BEETLE representative {precision} cache changed during resume verification"
            )
        resume[precision] = {
            "payloads_verified": len(before_resume[precision]),
            "unchanged": True,
        }
    stores = resumed_stores
    label_remap, _ = build_label_remap(_PIXEL_MAPPING, num_classes=4)
    head = SegmentationHead(
        num_classes=4,
        geometry=compute_dense_geometry(target_size=512, patch_size=14),
        spacing_um=0.5,
        spacing_policy="native_if_coarser",
        backend=preprocessing.mask_backend,
        tolerance=preprocessing.tolerance,
        label_remap=label_remap,
    )

    representatives = []
    for record in roi_dataset.samples.values():
        paths = {}
        tensors = {}
        metadata_by_precision = {}
        for precision, store in stores.items():
            tensor_path, sidecar_path = _payload_paths(store, record)
            if not tensor_path.is_file() or not sidecar_path.is_file():
                raise ValueError(
                    f"BEETLE representative {record.sample_id} is missing its "
                    f"{precision} payload or sidecar"
                )
            tensor = torch.load(tensor_path, weights_only=True, map_location="cpu")
            expected_dtype = torch.float16 if precision == "fp16" else torch.float32
            if tensor.dtype != expected_dtype or tuple(tensor.shape) != (1280, 37, 37):
                raise ValueError(
                    f"BEETLE representative {record.sample_id} {precision} tensor has "
                    f"dtype/shape {tensor.dtype}/{tuple(tensor.shape)}"
                )
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))
            _validate_sidecar(
                metadata,
                record=record,
                precision=precision,
                preprocessing=preprocessing,
            )
            tensors[precision] = tensor
            paths[precision] = (tensor_path, sidecar_path)
            metadata_by_precision[precision] = metadata

        geometry_fields = (
            "feature_dim",
            "grid_shape",
            "target_size",
            "patch_size",
            "encoded_size",
            "pad",
            "declared_spacing_um",
            "source_spacing_um",
            "read_spacing_um",
            "effective_spacing_um",
            "spacing_at_level_0",
            "read_level",
            "read_size",
            "output_size",
            "is_within_tolerance",
            "window_size",
            "overlap",
            "feature_kind",
        )
        geometry = {field: metadata_by_precision["fp16"].get(field) for field in geometry_fields}
        fp32_geometry = {
            field: metadata_by_precision["fp32"].get(field) for field in geometry_fields
        }
        if geometry != fp32_geometry:
            raise ValueError(
                f"BEETLE representative {record.sample_id} fp16/fp32 sidecar geometry differs"
            )

        parity = _parity(tensors["fp16"], tensors["fp32"])
        if parity.cosine_similarity < threshold:
            raise ValueError(
                f"BEETLE representative {record.sample_id} fp16/fp32 cosine similarity "
                f"{parity.cosine_similarity} is below {threshold}"
            )
        mask = head.extract_targets(record)["mask"]
        if tuple(mask.shape) != (512, 512):
            raise ValueError(
                f"BEETLE representative {record.sample_id} mask has shape {tuple(mask.shape)}"
            )
        labels = sorted(int(value) for value in torch.unique(mask).tolist())
        if any(label not in {0, 1, 2, 3, 255} for label in labels):
            raise ValueError(
                f"BEETLE representative {record.sample_id} mask has invalid labels {labels}"
            )
        representatives.append(
            RepresentativeExtraction(
                sample_id=str(record.slide_id),
                kind=("native_spacing_exception" if record.slide_id in coarse_ids else "ordinary"),
                selection="annotation_eligible_roi",
                roi_id=record.sample_id,
                region=[int(record.region[0]), int(record.region[1])],
                mask_shape=list(mask.shape),
                mask_labels=labels,
                geometry=geometry,
                fp16_tensor=_file_evidence(paths["fp16"][0]),
                fp32_tensor=_file_evidence(paths["fp32"][0]),
                fp16_sidecar=_file_evidence(paths["fp16"][1]),
                fp32_sidecar=_file_evidence(paths["fp32"][1]),
                parity=parity,
            )
        )

    minimum = min(row.parity.cosine_similarity for row in representatives)
    return RepresentativeExtractionResult(
        status="passed",
        minimum_required_cosine_similarity=threshold,
        minimum_cosine_similarity=minimum,
        resume=resume,
        representatives=tuple(representatives),
    )


__all__ = [
    "DEFAULT_MINIMUM_COSINE_SIMILARITY",
    "FileEvidence",
    "ParityStatistics",
    "RepresentativeExtraction",
    "RepresentativeExtractionResult",
    "run_representative_extraction_parity",
]
