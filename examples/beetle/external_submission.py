"""Build, validate, and optionally score the BEETLE External submission.

The released External ROIs are flat PNGs, so their physical scale comes from a validated
ROI-to-WSI sidecar rather than image metadata. This Project-protocol command owns the
BEETLE cohort, filename, label, and archive contracts; Soma remains dataset-neutral.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image

from examples.beetle.external_contract import (
    CLASS_VOCABULARY,
    EXTERNAL_PATIENT_COUNT,
    EXTERNAL_ROI_COUNT,
    MODEL_INDEX_TO_SUBMISSION_LABEL,
    NUM_CLASSES,
    SUBMISSION_LABELS,
    ExternalCohort,
    ExternalRoi,
    exact_directory_paths,
    load_roi_sidecar,
    validate_roi_inputs,
    validate_submission_pngs,
    write_flat_submission_zip,
)
from examples.beetle.external_runtime import (
    PredictorLoader,
    encoder_runtime_environment,
    load_selected_fold_predictor,
    selected_arm_from_file,
    selected_checkpoints,
    sha256_file,
    validate_selected_run_recipe,
)


def _spacing_decision(result, predictor) -> str:
    if result.applied_scale is not None:
        if result.applied_scale >= 1.0:
            raise ValueError(
                "BEETLE External inference must never upsample coarse tissue"
            )
        return "downsample_finer_input"
    training_spacing = getattr(predictor, "spacing_um", None)
    tolerance = float(getattr(predictor, "tolerance", 0.05))
    if training_spacing is None:
        raise ValueError("BEETLE External predictor must declare its training spacing")
    ratio = float(result.native_spacing_um) / float(training_spacing)
    if ratio > 1.0 + tolerance:
        return "native_coarse_no_upsample"
    return "native_within_tolerance"


def evaluate_external_submission(
    *,
    roi_sidecar: str | Path,
    predictions_dir: str | Path,
    labels_dir: str | Path,
    output_path: str | Path,
    expected_rois: int = EXTERNAL_ROI_COUNT,
    expected_patients: int = EXTERNAL_PATIENT_COUNT,
) -> dict:
    """Score fixed predictions only when the paper lead supplies sequestered labels."""
    from examples.beetle.report_oof import (
        bootstrap_summary_payload,
        confusion_metrics_payload,
        group_sample_confusions,
        patient_bootstrap,
    )
    from soma.evaluation import (
        SegmentationConfusionRecord,
        aggregate_confusion_matrices,
    )

    records = load_roi_sidecar(roi_sidecar, expected_rois=expected_rois)
    validate_submission_pngs(predictions_dir, records)
    label_paths = exact_directory_paths(
        labels_dir,
        [record.roi_filename for record in records],
        artifact_label="BEETLE External label filenames",
    )
    labels_by_name = {path.name: path for path in label_paths}

    sample_records = []
    sample_to_patient = {}
    for record in records:
        prediction_path = Path(predictions_dir) / record.roi_filename
        label_path = labels_by_name[record.roi_filename]
        with Image.open(prediction_path) as prediction_image:
            prediction = np.asarray(prediction_image, dtype=np.uint8)
        with Image.open(label_path) as label_image:
            if label_image.format != "PNG" or label_image.mode != "L":
                raise ValueError(
                    f"BEETLE External label {record.roi_filename!r} must be grayscale PNG"
                )
            if label_image.size != (record.width, record.height):
                raise ValueError(
                    f"BEETLE External label {record.roi_filename!r} dimensions "
                    f"{label_image.size} disagree with the ROI sidecar"
                )
            truth = np.asarray(label_image, dtype=np.uint8)
        truth_values = set(int(value) for value in np.unique(truth))
        if not truth_values <= (SUBMISSION_LABELS | {0}):
            raise ValueError(
                f"BEETLE External label {record.roi_filename!r} contains invalid labels "
                f"{sorted(truth_values)}"
            )
        annotated = truth != 0
        encoded = (truth[annotated].astype(np.int64) - 1) * NUM_CLASSES + (
            prediction[annotated].astype(np.int64) - 1
        )
        matrix = np.bincount(encoded, minlength=NUM_CLASSES * NUM_CLASSES).reshape(
            NUM_CLASSES, NUM_CLASSES
        )
        sample_records.append(
            SegmentationConfusionRecord(
                sample_id=record.roi_filename,
                fold=0,
                class_vocabulary=CLASS_VOCABULARY,
                confusion_matrix=matrix,
            )
        )
        sample_to_patient[record.roi_filename] = record.patient_id

    patients = group_sample_confusions(sample_records, sample_to_patient)
    if len(patients) != expected_patients:
        raise ValueError(
            f"BEETLE External evaluation requires exactly {expected_patients} patients, "
            f"got {len(patients)}"
        )
    pooled = aggregate_confusion_matrices(
        [patient.confusion_matrix for patient in patients]
    )
    bootstrap = patient_bootstrap(patients)
    rois_per_patient = {
        patient.patient_id: sum(
            record.patient_id == patient.patient_id for record in records
        )
        for patient in patients
    }
    source_wsis_per_patient = {
        patient.patient_id: len(
            {
                record.source_wsi
                for record in records
                if record.patient_id == patient.patient_id
            }
        )
        for patient in patients
    }
    report = {
        "schema_version": 1,
        "hidden_external_labels_supplied": True,
        "class_vocabulary": list(CLASS_VOCABULARY),
        "coverage": {
            "expected_patient_count": expected_patients,
            "observed_patient_count": len(patients),
            "roi_count": len(records),
            "patient_ids": [patient.patient_id for patient in patients],
            "rois_per_patient": rois_per_patient,
            "source_wsis_per_patient": source_wsis_per_patient,
        },
        "pooled": confusion_metrics_payload(pooled, CLASS_VOCABULARY),
        "bootstrap": bootstrap_summary_payload(bootstrap, CLASS_VOCABULARY),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def run_external_inference(
    *,
    selection_path: str | Path,
    run_dir: str | Path,
    roi_dir: str | Path,
    roi_sidecar: str | Path,
    output_dir: str | Path,
    audit_path: str | Path,
    zip_path: str | Path,
    protocol_resolution: str | Path | None = None,
    expected_rois: int = EXTERNAL_ROI_COUNT,
    predictor_loader: PredictorLoader = load_selected_fold_predictor,
    require_encoder_binding: bool = True,
) -> dict:
    """Generate the submission without consulting or requiring External labels."""
    run_dir = Path(run_dir)
    roi_dir = Path(roi_dir)
    output_dir = Path(output_dir)
    audit_path = Path(audit_path)
    records = load_roi_sidecar(roi_sidecar, expected_rois=expected_rois)
    validate_roi_inputs(roi_dir, records)
    selected_arm = selected_arm_from_file(selection_path)
    checkpoints = selected_checkpoints(run_dir, selected_arm)
    if require_encoder_binding and protocol_resolution is None:
        raise ValueError(
            "BEETLE production inference requires --protocol-resolution to bind the "
            "validated immutable encoder"
        )
    selected_recipe = (
        validate_selected_run_recipe(protocol_resolution, selected_arm, run_dir)
        if protocol_resolution is not None
        else {"mode": "standin_loader", "validated_selected_recipe": False}
    )
    runtime_context = (
        encoder_runtime_environment(protocol_resolution)
        if protocol_resolution is not None
        else nullcontext({"mode": "standin_loader", "validated_encoder_loaded": False})
    )
    with runtime_context as encoder_runtime:
        predictor = predictor_loader(run_dir, checkpoints)
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"BEETLE submission output directory must be empty: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        decisions = []
        for record in records:
            result = predictor.predict_image(
                roi_dir / record.roi_filename,
                native_spacing_um=record.native_spacing_um,
                allow_upsample=False,
                return_probs=False,
            )
            expected_shape = (record.height, record.width)
            if result.labels.shape != expected_shape:
                raise ValueError(
                    f"BEETLE prediction {record.roi_filename!r} shape {result.labels.shape} "
                    f"does not match input {expected_shape}"
                )
            model_indices = np.asarray(result.labels)
            if np.any(model_indices < 0) or np.any(model_indices >= NUM_CLASSES):
                raise ValueError(
                    f"BEETLE prediction {record.roi_filename!r} has a class index "
                    "outside the four-class organizer vocabulary"
                )
            labels = MODEL_INDEX_TO_SUBMISSION_LABEL[model_indices]
            Image.fromarray(labels, mode="L").save(output_dir / record.roi_filename)
            decisions.append(
                {
                    "roi_filename": record.roi_filename,
                    "patient_id": record.patient_id,
                    "source_wsi": record.source_wsi,
                    "native_spacing_um": record.native_spacing_um,
                    "training_spacing_um": float(predictor.spacing_um),
                    "tolerance": float(predictor.tolerance),
                    "spacing_decision": _spacing_decision(result, predictor),
                    "applied_scale": result.applied_scale,
                    "input_width": record.width,
                    "input_height": record.height,
                    "output_width": int(result.labels.shape[1]),
                    "output_height": int(result.labels.shape[0]),
                    "output_matches_input_dimensions": result.labels.shape
                    == expected_shape,
                }
            )

    paths = validate_submission_pngs(output_dir, records)
    archive = write_flat_submission_zip(paths, zip_path)
    audit = {
        "schema_version": 1,
        "selected_arm": selected_arm,
        "probability_ensemble": "mean_of_five_fold_softmaxes",
        "hidden_external_labels_used": False,
        "expected_roi_count": expected_rois,
        "uses_publication_roi_count": expected_rois == EXTERNAL_ROI_COUNT,
        "encoder_runtime": encoder_runtime,
        "selected_recipe": selected_recipe,
        "roi_sidecar": str(Path(roi_sidecar)),
        "checkpoints": [
            {"fold": fold, "path": str(path), "sha256": sha256_file(path)}
            for fold, path in enumerate(checkpoints)
        ],
        "roi_decisions": decisions,
        "submission_zip": {"path": str(archive), "sha256": sha256_file(archive)},
    }
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main(
    argv: Sequence[str] | None = None,
    *,
    predictor_loader: PredictorLoader = load_selected_fold_predictor,
    cohort: ExternalCohort = ExternalCohort(),
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    infer = subparsers.add_parser(
        "infer", help="infer, validate, and ZIP the External ROIs"
    )
    infer.add_argument("--selection", type=Path, required=True)
    infer.add_argument("--run-dir", type=Path, required=True)
    infer.add_argument("--protocol-resolution", type=Path)
    infer.add_argument("--roi-dir", type=Path, required=True)
    infer.add_argument("--roi-sidecar", type=Path, required=True)
    infer.add_argument("--output-dir", type=Path, required=True)
    infer.add_argument("--audit", type=Path, required=True)
    infer.add_argument("--zip", type=Path, required=True)
    validate = subparsers.add_parser(
        "validate", help="validate existing submission PNGs and write the flat ZIP"
    )
    validate.add_argument("--roi-sidecar", type=Path, required=True)
    validate.add_argument("--output-dir", type=Path, required=True)
    validate.add_argument("--zip", type=Path, required=True)
    evaluate = subparsers.add_parser(
        "evaluate", help="score fixed predictions after sequestered labels are supplied"
    )
    evaluate.add_argument("--roi-sidecar", type=Path, required=True)
    evaluate.add_argument("--predictions-dir", type=Path, required=True)
    evaluate.add_argument("--labels-dir", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "infer":
        run_external_inference(
            selection_path=args.selection,
            run_dir=args.run_dir,
            roi_dir=args.roi_dir,
            roi_sidecar=args.roi_sidecar,
            output_dir=args.output_dir,
            audit_path=args.audit,
            zip_path=args.zip,
            protocol_resolution=args.protocol_resolution,
            expected_rois=cohort.roi_count,
            predictor_loader=predictor_loader,
            require_encoder_binding=predictor_loader is load_selected_fold_predictor,
        )
    elif args.command == "validate":
        records = load_roi_sidecar(args.roi_sidecar, expected_rois=cohort.roi_count)
        paths = validate_submission_pngs(args.output_dir, records)
        write_flat_submission_zip(paths, args.zip)
    elif args.command == "evaluate":
        evaluate_external_submission(
            roi_sidecar=args.roi_sidecar,
            predictions_dir=args.predictions_dir,
            labels_dir=args.labels_dir,
            output_path=args.output,
            expected_rois=cohort.roi_count,
            expected_patients=cohort.patient_count,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "EXTERNAL_PATIENT_COUNT",
    "EXTERNAL_ROI_COUNT",
    "ExternalCohort",
    "ExternalRoi",
    "encoder_runtime_environment",
    "evaluate_external_submission",
    "load_roi_sidecar",
    "load_selected_fold_predictor",
    "main",
    "run_external_inference",
    "validate_submission_pngs",
    "write_flat_submission_zip",
]
