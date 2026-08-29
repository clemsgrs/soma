"""Assemble, validate, and report the BEETLE publication handoff archive.

The paper lead receives one compact, self-describing archive: resolved configs,
provenance, all ten decoder checkpoints, histories, sampler audits, development
evidence, the External submission, and per-file checksums. This module builds that
archive from the campaign's real artifacts, proves its completeness and integrity
after unpacking, and emits the final acceptance report that maps every artifact
group to the reviewer request. The gated Virchow2 weights and the large dense
feature cache are never archived; their absence is asserted, not assumed.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
import json
from pathlib import Path
import re
import shutil
from typing import Mapping, Sequence
import zipfile

import yaml

from examples.beetle import launch
from examples.beetle.curate import FULL_COHORT_PATIENTS
from examples.beetle.external_contract import (
    EXTERNAL_ROI_COUNT,
    load_roi_sidecar,
    validate_submission_pngs,
)
from examples.beetle.external_runtime import selected_arm_from_file, sha256_file
from examples.beetle.protocol import ARM_NAMES, NUM_FOLDS
from examples.beetle.report_oof import NATIVE_SPACING_EXCEPTION_PATIENT_IDS


README_NAME = "README.md"
MANIFEST_NAME = "artifact_manifest.json"
CHECKPOINT_FILENAME = "best_model.pt"
HISTORY_FILENAME = "training_history.json"
SAMPLER_AUDIT_FILENAME = "roi_batch_sampling.json"
RUN_METADATA_FILENAME = "run.yaml"

CONFIGS_DIR = "configs"
CHECKPOINTS_DIR = "checkpoints"
HISTORIES_DIR = "histories"
SAMPLER_AUDITS_DIR = "sampler_audits"
PROVENANCE_DIR = "provenance"
DEVELOPMENT_DIR = "development"
EXTERNAL_DIR = "external"
EXTERNAL_MASKS_DIR = f"{EXTERNAL_DIR}/masks"

ENCODER_LOCK_ARCHIVE_PATH = f"{PROVENANCE_DIR}/encoder_lock.json"
PROTOCOL_RESOLUTION_ARCHIVE_PATH = f"{PROVENANCE_DIR}/protocol_resolution.json"
HARDWARE_PREFLIGHT_ARCHIVE_PATH = f"{PROVENANCE_DIR}/hardware_preflight.json"
OOF_REPORT_ARCHIVE_PATH = f"{DEVELOPMENT_DIR}/oof_report.json"
ARM_SELECTION_ARCHIVE_PATH = f"{DEVELOPMENT_DIR}/arm_selection.json"
ROI_SIDECAR_ARCHIVE_PATH = f"{EXTERNAL_DIR}/roi_to_wsi.json"
SUBMISSION_AUDIT_ARCHIVE_PATH = f"{EXTERNAL_DIR}/submission_audit.json"
SUBMISSION_ZIP_ARCHIVE_PATH = f"{EXTERNAL_DIR}/submission.zip"
EXTERNAL_REPORT_ARCHIVE_PATH = f"{EXTERNAL_DIR}/external_report.json"

EXPECTED_CHECKPOINT_COUNT = len(ARM_NAMES) * NUM_FOLDS
SELECTED_CHECKPOINT_COUNT = NUM_FOLDS
ALLOWED_SPACING_DECISIONS = frozenset(
    {"native_within_tolerance", "downsample_finer_input", "native_coarse_no_upsample"}
)

# Gated Virchow2 weight files and dense feature-cache payload directories must never
# reach the archive. `soma.dense.store` writes `<sample>.pt` grids under one of these
# directory names, and the resolved cache root is namespaced `virchow2_*_dense_*`.
FORBIDDEN_FILENAME_PATTERNS = ("*.safetensors", "pytorch_model*.bin")
FORBIDDEN_DIRECTORY_NAMES = frozenset(
    {"dense_embeddings", "dense_image_embeddings", "cached_grids", ".hf-home"}
)
FORBIDDEN_DIRECTORY_PATTERNS = ("virchow2_*_dense_*",)

_SHA1_PATTERN = re.compile(r"^[0-9a-f]{40}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MISSING = object()


def arm_config_archive_path(arm: str) -> str:
    return f"{CONFIGS_DIR}/{arm}.yaml"


def arm_run_metadata_archive_path(arm: str) -> str:
    return f"{PROVENANCE_DIR}/run_{arm}.yaml"


def checkpoint_archive_path(arm: str, fold: int) -> str:
    return f"{CHECKPOINTS_DIR}/{arm}/fold_{fold}/{CHECKPOINT_FILENAME}"


def history_archive_path(arm: str, fold: int) -> str:
    return f"{HISTORIES_DIR}/{arm}/fold_{fold}/{HISTORY_FILENAME}"


def sampler_audit_archive_path(arm: str, fold: int) -> str:
    return f"{SAMPLER_AUDITS_DIR}/{arm}/fold_{fold}/{SAMPLER_AUDIT_FILENAME}"


def required_archive_paths() -> tuple[str, ...]:
    """Every fixed relative path a complete archive must contain (masks excluded)."""
    paths = [
        README_NAME,
        ENCODER_LOCK_ARCHIVE_PATH,
        PROTOCOL_RESOLUTION_ARCHIVE_PATH,
        HARDWARE_PREFLIGHT_ARCHIVE_PATH,
        OOF_REPORT_ARCHIVE_PATH,
        ARM_SELECTION_ARCHIVE_PATH,
        ROI_SIDECAR_ARCHIVE_PATH,
        SUBMISSION_AUDIT_ARCHIVE_PATH,
        SUBMISSION_ZIP_ARCHIVE_PATH,
    ]
    for arm in ARM_NAMES:
        paths.append(arm_config_archive_path(arm))
        paths.append(arm_run_metadata_archive_path(arm))
        for fold in range(NUM_FOLDS):
            paths.append(checkpoint_archive_path(arm, fold))
            paths.append(history_archive_path(arm, fold))
            paths.append(sampler_audit_archive_path(arm, fold))
    return tuple(sorted(paths))


@dataclass(frozen=True)
class CheckOutcome:
    """One named acceptance check and its individually actionable failures."""

    name: str
    failures: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class ValidationOutcome:
    """The complete acceptance validation of one unpacked archive directory."""

    archive_dir: Path
    checks: tuple[CheckOutcome, ...]

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failures(self) -> tuple[str, ...]:
        return tuple(
            failure for check in self.checks for failure in check.failures
        )


def _archive_files(archive_dir: Path) -> list[str]:
    return sorted(
        path.relative_to(archive_dir).as_posix()
        for path in archive_dir.rglob("*")
        if path.is_file()
    )


def _load_json_object(path: Path, label: str, failures: list[str]) -> dict | None:
    if not path.is_file():
        failures.append(f"{label} is missing: expected file {path.name} was not found")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        failures.append(f"{label} is not readable JSON: {exc}")
        return None
    if not isinstance(payload, dict):
        failures.append(f"{label} must be a JSON object")
        return None
    return payload


def _leaf_items(payload: object, prefix: tuple[str, ...] = ()) -> dict[tuple[str, ...], object]:
    """Flatten nested mappings to dotted leaf paths; lists count as leaves."""
    if isinstance(payload, Mapping):
        items: dict[tuple[str, ...], object] = {}
        for key, value in payload.items():
            items.update(_leaf_items(value, prefix + (str(key),)))
        return items
    return {prefix: payload}


def _tracked_encoder_lock() -> dict:
    return json.loads(launch.ENCODER_LOCK_PATH.read_text(encoding="utf-8"))


def _sampling_overlay_leaf_paths() -> set[tuple[str, ...]]:
    """Leaf key paths the tracked arm overlays are allowed to differ on."""
    allowed: set[tuple[str, ...]] = set()
    for arm in ARM_NAMES:
        overlay = yaml.safe_load(
            (launch.CONFIG_DIR / f"{arm}.yaml").read_text(encoding="utf-8")
        )
        allowed.update(_leaf_items(overlay))
    return allowed


class _ArchiveValidator:
    """Run every acceptance check over one unpacked archive, collecting failures."""

    def __init__(self, archive_dir: Path) -> None:
        self.archive_dir = archive_dir
        self._sidecar_records = None
        self._sidecar_failures: list[str] = []
        try:
            self._sidecar_records = load_roi_sidecar(
                archive_dir / ROI_SIDECAR_ARCHIVE_PATH
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._sidecar_failures.append(
                f"{ROI_SIDECAR_ARCHIVE_PATH} failed mask-manifest validation: {exc}"
            )
        self._selected_arm: str | None = None
        self._selection_failures: list[str] = []
        try:
            self._selected_arm = selected_arm_from_file(
                archive_dir / ARM_SELECTION_ARCHIVE_PATH
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            self._selection_failures.append(
                f"{ARM_SELECTION_ARCHIVE_PATH} is not a valid development-only "
                f"arm-selection artifact: {exc}"
            )

    def run(self) -> ValidationOutcome:
        checks = []
        for name, method in (
            ("manifest_coverage", self.check_manifest_coverage),
            ("file_checksums", self.check_file_checksums),
            ("required_artifacts", self.check_required_artifacts),
            ("excluded_payloads", self.check_excluded_payloads),
            ("decoder_checkpoints", self.check_decoder_checkpoints),
            ("training_evidence", self.check_training_evidence),
            ("resolved_arm_configs", self.check_resolved_arm_configs),
            ("encoder_identity", self.check_encoder_identity),
            ("hardware_preflight", self.check_hardware_preflight),
            ("environment_provenance", self.check_environment_provenance),
            ("external_mask_manifest", self.check_external_mask_manifest),
            ("submission_zip", self.check_submission_zip),
            ("arm_selection", self.check_arm_selection),
            ("selected_external_models", self.check_selected_external_models),
            ("patient_matrix_coverage", self.check_patient_matrix_coverage),
            ("external_metrics", self.check_external_metrics),
        ):
            checks.append(CheckOutcome(name=name, failures=tuple(method())))
        return ValidationOutcome(archive_dir=self.archive_dir, checks=tuple(checks))

    # --- manifest and integrity -------------------------------------------------

    def _manifest_entries(self, failures: list[str]) -> list[dict] | None:
        payload = _load_json_object(
            self.archive_dir / MANIFEST_NAME, MANIFEST_NAME, failures
        )
        if payload is None:
            return None
        entries = payload.get("files")
        if not isinstance(entries, list) or not all(
            isinstance(entry, dict) for entry in entries
        ):
            failures.append(f"{MANIFEST_NAME} must record a list of file objects")
            return None
        return entries

    def check_manifest_coverage(self) -> list[str]:
        failures: list[str] = []
        entries = self._manifest_entries(failures)
        if entries is None:
            return failures
        listed = [str(entry.get("path", "")) for entry in entries]
        duplicates = sorted(
            {path for path in listed if listed.count(path) > 1}
        )
        for path in duplicates:
            failures.append(f"{MANIFEST_NAME} lists {path} more than once")
        listed_set = set(listed)
        present = set(_archive_files(self.archive_dir)) - {MANIFEST_NAME}
        for path in sorted(listed_set - present):
            failures.append(
                f"archive is missing {path}, which is listed in {MANIFEST_NAME}"
            )
        for path in sorted(present - listed_set):
            failures.append(
                f"{path} is present in the archive but not listed in {MANIFEST_NAME}"
            )
        return failures

    def check_file_checksums(self) -> list[str]:
        failures: list[str] = []
        entries = self._manifest_entries(failures)
        if entries is None:
            return failures
        for entry in entries:
            rel = str(entry.get("path", ""))
            path = self.archive_dir / rel
            if not path.is_file():
                continue  # reported by manifest_coverage
            actual_size = path.stat().st_size
            expected_size = entry.get("size_bytes")
            if actual_size != expected_size:
                failures.append(
                    f"{rel} is {actual_size} bytes but {MANIFEST_NAME} "
                    f"records {expected_size}"
                )
            actual_sha256 = sha256_file(path)
            expected_sha256 = entry.get("sha256")
            if actual_sha256 != expected_sha256:
                failures.append(
                    f"{rel} sha256 {actual_sha256} does not match {MANIFEST_NAME} "
                    f"entry {expected_sha256}"
                )
        return failures

    def check_required_artifacts(self) -> list[str]:
        return [
            f"required artifact {rel} is missing from the archive"
            for rel in required_archive_paths()
            if not (self.archive_dir / rel).is_file()
        ]

    def check_excluded_payloads(self) -> list[str]:
        failures: list[str] = []
        for rel in _archive_files(self.archive_dir):
            name = Path(rel).name
            for pattern in FORBIDDEN_FILENAME_PATTERNS:
                if fnmatch.fnmatch(name, pattern):
                    failures.append(
                        f"{rel} matches gated Virchow2 weight pattern {pattern!r}; "
                        "encoder weights must not be redistributed"
                    )
            for part in Path(rel).parent.parts:
                if part in FORBIDDEN_DIRECTORY_NAMES or any(
                    fnmatch.fnmatch(part, pattern)
                    for pattern in FORBIDDEN_DIRECTORY_PATTERNS
                ):
                    failures.append(
                        f"{rel} lies in feature-cache directory {part!r}; the dense "
                        "feature cache must not be redistributed"
                    )
        return failures

    # --- development artifacts --------------------------------------------------

    def check_decoder_checkpoints(self) -> list[str]:
        failures: list[str] = []
        expected = {
            checkpoint_archive_path(arm, fold)
            for arm in ARM_NAMES
            for fold in range(NUM_FOLDS)
        }
        found = {
            rel
            for rel in _archive_files(self.archive_dir)
            if rel.startswith(f"{CHECKPOINTS_DIR}/")
        }
        if len(found) != EXPECTED_CHECKPOINT_COUNT:
            failures.append(
                f"expected exactly {EXPECTED_CHECKPOINT_COUNT} decoder checkpoints "
                f"({len(ARM_NAMES)} arms x {NUM_FOLDS} folds), found {len(found)}"
            )
        for rel in sorted(expected - found):
            failures.append(f"decoder checkpoint {rel} is missing")
        for rel in sorted(found - expected):
            failures.append(f"unexpected file {rel} under {CHECKPOINTS_DIR}/")
        return failures

    def check_training_evidence(self) -> list[str]:
        failures: list[str] = []
        for arm in ARM_NAMES:
            for fold in range(NUM_FOLDS):
                for rel, label in (
                    (history_archive_path(arm, fold), "training history"),
                    (sampler_audit_archive_path(arm, fold), "sampler audit"),
                ):
                    if not (self.archive_dir / rel).is_file():
                        failures.append(
                            f"{label} {rel} is missing for arm {arm!r} fold {fold}"
                        )
        return failures

    def _load_arm_configs(
        self, failures: list[str]
    ) -> dict[str, dict] | None:
        configs: dict[str, dict] = {}
        for arm in ARM_NAMES:
            rel = arm_config_archive_path(arm)
            path = self.archive_dir / rel
            if not path.is_file():
                failures.append(f"resolved arm config {rel} is missing")
                continue
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                failures.append(f"{rel} is not readable YAML: {exc}")
                continue
            if not isinstance(payload, dict):
                failures.append(f"{rel} must be a YAML mapping")
                continue
            configs[arm] = payload
        return configs if len(configs) == len(ARM_NAMES) else None

    def check_resolved_arm_configs(self) -> list[str]:
        failures: list[str] = []
        configs = self._load_arm_configs(failures)
        if configs is None:
            return failures
        lock = _tracked_encoder_lock()
        for arm, payload in configs.items():
            rel = arm_config_archive_path(arm)
            sampling = payload.get("training", {}).get("roi_batch_sampling")
            if sampling != arm:
                failures.append(
                    f"{rel} records training.roi_batch_sampling {sampling!r}; "
                    f"the {arm} arm config must record {arm!r}"
                )
            cache_root = str(payload.get("cache", {}).get("root_dir", ""))
            if lock["revision"] not in cache_root or lock["weight_sha256"] not in cache_root:
                failures.append(
                    f"{rel} cache.root_dir {cache_root!r} is not namespaced by the "
                    "locked encoder revision and weight digest"
                )
        allowed = _sampling_overlay_leaf_paths()
        flattened = {arm: _leaf_items(configs[arm]) for arm in ARM_NAMES}
        first, second = ARM_NAMES
        for path_key in sorted(set(flattened[first]) | set(flattened[second])):
            left = flattened[first].get(path_key, _MISSING)
            right = flattened[second].get(path_key, _MISSING)
            if left != right and path_key not in allowed:
                failures.append(
                    f"{arm_config_archive_path(first)} and "
                    f"{arm_config_archive_path(second)} differ outside the "
                    f"sampling-arm overlay at {'.'.join(path_key)}: "
                    f"{left!r} != {right!r}"
                )
        batch_sizes = {
            arm: flattened[arm].get(("training", "batch_size"), _MISSING)
            for arm in ARM_NAMES
        }
        if len(set(batch_sizes.values())) != 1:
            failures.append(
                f"resolved arm configs disagree on training.batch_size: {batch_sizes}"
            )
        return failures

    def check_encoder_identity(self) -> list[str]:
        failures: list[str] = []
        lock = _tracked_encoder_lock()
        archived = _load_json_object(
            self.archive_dir / ENCODER_LOCK_ARCHIVE_PATH,
            ENCODER_LOCK_ARCHIVE_PATH,
            failures,
        )
        if archived is not None:
            for field in sorted(set(lock) | set(archived)):
                if archived.get(field) != lock.get(field):
                    failures.append(
                        f"{ENCODER_LOCK_ARCHIVE_PATH} field {field!r} is "
                        f"{archived.get(field)!r}; the tracked protocol lock records "
                        f"{lock.get(field)!r}"
                    )
        resolution = _load_json_object(
            self.archive_dir / PROTOCOL_RESOLUTION_ARCHIVE_PATH,
            PROTOCOL_RESOLUTION_ARCHIVE_PATH,
            failures,
        )
        if resolution is not None:
            if resolution.get("encoder_lock") != lock:
                failures.append(
                    f"{PROTOCOL_RESOLUTION_ARCHIVE_PATH} encoder_lock disagrees with "
                    "the tracked protocol lock"
                )
            observed = resolution.get("encoder_preflight")
            if not isinstance(observed, dict):
                failures.append(
                    f"{PROTOCOL_RESOLUTION_ARCHIVE_PATH} is missing encoder_preflight "
                    "provenance"
                )
            else:
                for field in lock:
                    if observed.get(field) != lock[field]:
                        failures.append(
                            f"{PROTOCOL_RESOLUTION_ARCHIVE_PATH} encoder_preflight "
                            f"{field!r} is {observed.get(field)!r}; the tracked "
                            f"protocol lock records {lock[field]!r}"
                        )
                if observed.get("weight_checksum_verified") is not True:
                    failures.append(
                        f"{PROTOCOL_RESOLUTION_ARCHIVE_PATH} does not record a "
                        "verified Virchow2 weight checksum"
                    )
        return failures

    def check_hardware_preflight(self) -> list[str]:
        failures: list[str] = []
        lock = _tracked_encoder_lock()
        preflight_path = self.archive_dir / HARDWARE_PREFLIGHT_ARCHIVE_PATH
        preflight = _load_json_object(
            preflight_path, HARDWARE_PREFLIGHT_ARCHIVE_PATH, failures
        )
        resolution = _load_json_object(
            self.archive_dir / PROTOCOL_RESOLUTION_ARCHIVE_PATH,
            PROTOCOL_RESOLUTION_ARCHIVE_PATH,
            failures,
        )
        if preflight is None or resolution is None:
            return failures
        recorded = resolution.get("hardware_preflight", {})
        recorded_sha256 = recorded.get("sha256")
        actual_sha256 = sha256_file(preflight_path)
        if actual_sha256 != recorded_sha256:
            failures.append(
                f"{HARDWARE_PREFLIGHT_ARCHIVE_PATH} sha256 {actual_sha256} does not "
                f"match {PROTOCOL_RESOLUTION_ARCHIVE_PATH} record {recorded_sha256}"
            )
        encoder = preflight.get("encoder")
        if not isinstance(encoder, dict):
            failures.append(
                f"{HARDWARE_PREFLIGHT_ARCHIVE_PATH} is missing encoder provenance"
            )
        else:
            for field in lock:
                if encoder.get(field) != lock[field]:
                    failures.append(
                        f"{HARDWARE_PREFLIGHT_ARCHIVE_PATH} encoder {field!r} is "
                        f"{encoder.get(field)!r}; the tracked protocol lock records "
                        f"{lock[field]!r}"
                    )
        selected = preflight.get("selected_batch_size")
        recorded_batch = recorded.get("selected_batch_size")
        if selected != recorded_batch:
            failures.append(
                f"{HARDWARE_PREFLIGHT_ARCHIVE_PATH} selected_batch_size {selected!r} "
                f"disagrees with {PROTOCOL_RESOLUTION_ARCHIVE_PATH} {recorded_batch!r}"
            )
        config_failures: list[str] = []
        configs = self._load_arm_configs(config_failures)
        if configs is not None:
            for arm, payload in configs.items():
                configured = payload.get("training", {}).get("batch_size")
                if configured != selected:
                    failures.append(
                        f"{arm_config_archive_path(arm)} training.batch_size "
                        f"{configured!r} disagrees with the preflight-frozen "
                        f"batch size {selected!r}"
                    )
        return failures

    def check_environment_provenance(self) -> list[str]:
        failures: list[str] = []
        metadata: dict[str, dict] = {}
        for arm in ARM_NAMES:
            rel = arm_run_metadata_archive_path(arm)
            path = self.archive_dir / rel
            if not path.is_file():
                failures.append(f"run metadata {rel} is missing for arm {arm!r}")
                continue
            try:
                payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            except (OSError, yaml.YAMLError) as exc:
                failures.append(f"{rel} is not readable YAML: {exc}")
                continue
            if not isinstance(payload, dict):
                failures.append(f"{rel} must be a YAML mapping")
                continue
            metadata[arm] = payload
            git_sha = str(payload.get("git_sha", ""))
            if not _SHA1_PATTERN.match(git_sha):
                failures.append(
                    f"{rel} does not record a full Soma commit SHA (git_sha "
                    f"{git_sha!r})"
                )
            if payload.get("git_dirty") != "false":
                failures.append(
                    f"{rel} records git_dirty {payload.get('git_dirty')!r}; the "
                    "handoff requires a clean public Soma commit"
                )
            environment = payload.get("environment")
            if not isinstance(environment, dict) or not environment:
                failures.append(f"{rel} does not record the dependency environment")
            for field in ("dataset_file_checksum", "splits_file_checksum"):
                if not _SHA256_PATTERN.match(str(payload.get(field, ""))):
                    failures.append(
                        f"{rel} does not record a cohort hash ({field} "
                        f"{payload.get(field)!r})"
                    )
        if len(metadata) == len(ARM_NAMES):
            for field in ("git_sha", "dataset_file_checksum", "splits_file_checksum"):
                values = {arm: metadata[arm].get(field) for arm in ARM_NAMES}
                if len(set(values.values())) != 1:
                    failures.append(
                        f"the two arm run metadata files disagree on {field}: {values}"
                    )
        return failures

    # --- External submission ----------------------------------------------------

    def check_external_mask_manifest(self) -> list[str]:
        failures = list(self._sidecar_failures)
        if self._sidecar_records is None:
            return failures
        masks_dir = self.archive_dir / EXTERNAL_MASKS_DIR
        if not masks_dir.is_dir():
            failures.append(
                f"external mask directory {EXTERNAL_MASKS_DIR}/ is missing"
            )
            return failures
        try:
            validate_submission_pngs(masks_dir, self._sidecar_records)
        except (OSError, ValueError) as exc:
            failures.append(f"{EXTERNAL_MASKS_DIR}/ failed mask validation: {exc}")
        return failures

    def check_submission_zip(self) -> list[str]:
        failures: list[str] = []
        zip_path = self.archive_dir / SUBMISSION_ZIP_ARCHIVE_PATH
        if not zip_path.is_file():
            failures.append(f"submission ZIP {SUBMISSION_ZIP_ARCHIVE_PATH} is missing")
            return failures
        if self._sidecar_records is None:
            failures.append(
                f"cannot check {SUBMISSION_ZIP_ARCHIVE_PATH} members because the ROI "
                "sidecar failed mask-manifest validation"
            )
            return failures
        try:
            with zipfile.ZipFile(zip_path) as archive:
                members = archive.namelist()
        except (OSError, zipfile.BadZipFile) as exc:
            failures.append(f"{SUBMISSION_ZIP_ARCHIVE_PATH} is not a readable ZIP: {exc}")
            return failures
        expected = {record.roi_filename for record in self._sidecar_records}
        observed = set(members)
        for name in sorted(expected - observed):
            failures.append(
                f"{SUBMISSION_ZIP_ARCHIVE_PATH} is missing mask member {name}"
            )
        for name in sorted(observed - expected):
            failures.append(
                f"{SUBMISSION_ZIP_ARCHIVE_PATH} contains unexpected member {name}"
            )
        audit = _load_json_object(
            self.archive_dir / SUBMISSION_AUDIT_ARCHIVE_PATH,
            SUBMISSION_AUDIT_ARCHIVE_PATH,
            failures,
        )
        if audit is not None:
            recorded = (audit.get("submission_zip") or {}).get("sha256")
            actual = sha256_file(zip_path)
            if actual != recorded:
                failures.append(
                    f"{SUBMISSION_ZIP_ARCHIVE_PATH} sha256 {actual} does not match "
                    f"the submission audit record {recorded}"
                )
        return failures

    def check_arm_selection(self) -> list[str]:
        failures = list(self._selection_failures)
        if self._selected_arm is None:
            return failures
        payload = _load_json_object(
            self.archive_dir / ARM_SELECTION_ARCHIVE_PATH,
            ARM_SELECTION_ARCHIVE_PATH,
            failures,
        )
        if payload is None:
            return failures
        arms = payload.get("arms")
        if not isinstance(arms, dict) or set(arms) != set(ARM_NAMES):
            failures.append(
                f"{ARM_SELECTION_ARCHIVE_PATH} must report both arms {list(ARM_NAMES)}"
            )
            return failures
        for arm, report in arms.items():
            scores = report.get("fold_scores") if isinstance(report, dict) else None
            if not isinstance(scores, list) or len(scores) != NUM_FOLDS:
                failures.append(
                    f"{ARM_SELECTION_ARCHIVE_PATH} arm {arm!r} must report exactly "
                    f"{NUM_FOLDS} fold scores"
                )
        return failures

    def check_selected_external_models(self) -> list[str]:
        failures: list[str] = []
        audit = _load_json_object(
            self.archive_dir / SUBMISSION_AUDIT_ARCHIVE_PATH,
            SUBMISSION_AUDIT_ARCHIVE_PATH,
            failures,
        )
        if audit is None:
            return failures
        if audit.get("hidden_external_labels_used") is not False:
            failures.append(
                f"{SUBMISSION_AUDIT_ARCHIVE_PATH} must record that hidden External "
                "labels were not used"
            )
        audit_arm = audit.get("selected_arm")
        if self._selected_arm is not None and audit_arm != self._selected_arm:
            failures.append(
                f"{SUBMISSION_AUDIT_ARCHIVE_PATH} records selected arm {audit_arm!r} "
                f"but {ARM_SELECTION_ARCHIVE_PATH} selected {self._selected_arm!r}"
            )
        checkpoints = audit.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != SELECTED_CHECKPOINT_COUNT:
            observed = len(checkpoints) if isinstance(checkpoints, list) else "none"
            failures.append(
                f"{SUBMISSION_AUDIT_ARCHIVE_PATH} must record exactly "
                f"{SELECTED_CHECKPOINT_COUNT} selected fold checkpoints, got {observed}"
            )
        elif self._selected_arm is not None:
            for entry in checkpoints:
                fold = entry.get("fold")
                if not isinstance(fold, int) or not 0 <= fold < NUM_FOLDS:
                    failures.append(
                        f"{SUBMISSION_AUDIT_ARCHIVE_PATH} checkpoint entry has an "
                        f"invalid fold {fold!r}"
                    )
                    continue
                rel = checkpoint_archive_path(self._selected_arm, fold)
                archived = self.archive_dir / rel
                if not archived.is_file():
                    failures.append(
                        f"selected external model for fold {fold} has no archived "
                        f"checkpoint {rel}"
                    )
                    continue
                actual = sha256_file(archived)
                recorded = entry.get("sha256")
                if actual != recorded:
                    failures.append(
                        f"selected external model for fold {fold}: archived "
                        f"checkpoint {rel} sha256 {actual} does not match the "
                        f"submission audit record {recorded}"
                    )
        decisions = audit.get("roi_decisions")
        if not isinstance(decisions, list):
            failures.append(
                f"{SUBMISSION_AUDIT_ARCHIVE_PATH} is missing per-ROI spacing decisions"
            )
        else:
            if self._sidecar_records is not None:
                expected = {record.roi_filename for record in self._sidecar_records}
                observed = {
                    str(decision.get("roi_filename", "")) for decision in decisions
                }
                if observed != expected:
                    failures.append(
                        f"{SUBMISSION_AUDIT_ARCHIVE_PATH} spacing decisions cover "
                        f"{len(observed)} ROIs; the mask manifest declares "
                        f"{len(expected)}"
                    )
            for decision in decisions:
                verdict = decision.get("spacing_decision")
                if verdict not in ALLOWED_SPACING_DECISIONS:
                    failures.append(
                        f"{SUBMISSION_AUDIT_ARCHIVE_PATH} ROI "
                        f"{decision.get('roi_filename')!r} records spacing decision "
                        f"{verdict!r}; External tissue must never be upsampled"
                    )
        return failures

    # --- patient coverage -------------------------------------------------------

    def check_patient_matrix_coverage(self) -> list[str]:
        failures: list[str] = []
        report = _load_json_object(
            self.archive_dir / OOF_REPORT_ARCHIVE_PATH,
            OOF_REPORT_ARCHIVE_PATH,
            failures,
        )
        if report is None:
            return failures
        arms = report.get("arms")
        if not isinstance(arms, dict) or not set(ARM_NAMES) <= set(arms):
            failures.append(
                f"{OOF_REPORT_ARCHIVE_PATH} must report both arms {list(ARM_NAMES)}"
            )
            return failures
        exceptions = set(NATIVE_SPACING_EXCEPTION_PATIENT_IDS)
        patient_sets: dict[str, set[str]] = {}
        for arm in ARM_NAMES:
            payload = arms[arm]
            patient_ids = [
                str(value)
                for value in (payload.get("coverage") or {}).get("patient_ids", [])
            ]
            duplicates = sorted(
                {value for value in patient_ids if patient_ids.count(value) > 1}
            )
            if duplicates:
                failures.append(
                    f"{OOF_REPORT_ARCHIVE_PATH} arm {arm!r} repeats patient(s) "
                    f"{duplicates}; every patient must appear exactly once"
                )
            patient_sets[arm] = set(patient_ids)
            if len(patient_ids) != FULL_COHORT_PATIENTS:
                failures.append(
                    f"{OOF_REPORT_ARCHIVE_PATH} arm {arm!r} pools "
                    f"{len(patient_ids)} patients; the full development cohort has "
                    f"{FULL_COHORT_PATIENTS}"
                )
            confusion_ids = sorted(
                str(entry.get("patient_id", ""))
                for entry in payload.get("patient_confusions", [])
            )
            if confusion_ids != sorted(patient_ids):
                failures.append(
                    f"{OOF_REPORT_ARCHIVE_PATH} arm {arm!r} patient confusion "
                    "matrices do not cover exactly the pooled coverage patients"
                )
            missing_exceptions = sorted(exceptions - patient_sets[arm])
            if missing_exceptions:
                failures.append(
                    f"{OOF_REPORT_ARCHIVE_PATH} arm {arm!r} is missing "
                    f"spacing-exception patient(s) {missing_exceptions}"
                )
            excluded = set(
                str(value)
                for value in (payload.get("spacing_sensitivity") or {}).get(
                    "excluded_patient_ids", []
                )
            )
            if excluded != exceptions:
                failures.append(
                    f"{OOF_REPORT_ARCHIVE_PATH} arm {arm!r} spacing-sensitivity view "
                    f"flags {sorted(excluded)}; the protocol declares "
                    f"{sorted(exceptions)}"
                )
        for arm in ARM_NAMES:
            others = set().union(
                *(patient_sets[other] for other in ARM_NAMES if other != arm)
            )
            missing = sorted(others - patient_sets[arm])
            if missing:
                failures.append(
                    f"patient(s) {missing} are missing from arm {arm!r} in the "
                    f"pooled OOF evidence of {OOF_REPORT_ARCHIVE_PATH}"
                )
        return failures

    def check_external_metrics(self) -> list[str]:
        failures: list[str] = []
        path = self.archive_dir / EXTERNAL_REPORT_ARCHIVE_PATH
        if not path.is_file():
            return failures  # absent by design until sequestered labels arrive
        report = _load_json_object(path, EXTERNAL_REPORT_ARCHIVE_PATH, failures)
        if report is not None and report.get("hidden_external_labels_supplied") is not True:
            failures.append(
                f"{EXTERNAL_REPORT_ARCHIVE_PATH} exists but does not record that "
                "sequestered labels were supplied"
            )
        return failures


def validate_archive(archive_dir: str | Path) -> ValidationOutcome:
    """Run every acceptance check over one unpacked archive directory."""
    archive_dir = Path(archive_dir)
    if not archive_dir.is_dir():
        raise FileNotFoundError(f"BEETLE archive directory does not exist: {archive_dir}")
    return _ArchiveValidator(archive_dir).run()


# --- assembly -------------------------------------------------------------------


def write_artifact_manifest(archive_dir: str | Path) -> Path:
    """Hash every archived file into a per-file sha256 + size manifest."""
    archive_dir = Path(archive_dir)
    files = [rel for rel in _archive_files(archive_dir) if rel != MANIFEST_NAME]
    entries = [
        {
            "path": rel,
            "sha256": sha256_file(archive_dir / rel),
            "size_bytes": (archive_dir / rel).stat().st_size,
        }
        for rel in files
    ]
    payload = {
        "schema_version": 1,
        "archive": "beetle-virchow2-handoff",
        "generated_by": "examples.beetle.acceptance",
        "file_count": len(entries),
        "total_bytes": sum(entry["size_bytes"] for entry in entries),
        "files": entries,
    }
    manifest_path = archive_dir / MANIFEST_NAME
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest_path


def _readme_text(*, selected_arm: str, soma_commit_sha: str) -> str:
    exceptions = ", ".join(NATIVE_SPACING_EXCEPTION_PATIENT_IDS)
    return f"""# BEETLE frozen-Virchow2 validation handoff

Direct artifact archive for the BEETLE rebuttal: a fresh Protocol-locked Virchow2
validation on the organizer's five patient folds. All {FULL_COHORT_PATIENTS} development
patients are pooled out-of-fold; the three TCGA native-spacing exceptions
({exceptions}) stayed at native resolution and are flagged in the development report.
Selected sampling arm: `{selected_arm}` (development evidence only). Public Soma
commit: `{soma_commit_sha}`.

## Layout

- `{CONFIGS_DIR}/` - both resolved arm configs; they differ only in training-batch
  sampling and arm-identifying output metadata.
- `{PROVENANCE_DIR}/` - encoder lock (Virchow2 revision and weight checksum),
  protocol resolution, hardware preflight, and per-arm run metadata with the code
  revision, environment, and cohort/fold hashes.
- `{CHECKPOINTS_DIR}/` - all {EXPECTED_CHECKPOINT_COUNT} fold-selected decoder
  checkpoints ({len(ARM_NAMES)} arms x {NUM_FOLDS} folds).
- `{HISTORIES_DIR}/`, `{SAMPLER_AUDITS_DIR}/` - per-fold training histories and
  ROI-sampling audits for both arms.
- `{DEVELOPMENT_DIR}/` - the two-arm OOF report (fold metrics, per-patient confusion
  matrices, fixed-seed bootstrap) and the development-only arm selection.
- `{EXTERNAL_DIR}/` - the validated ROI-to-WSI sidecar ({EXTERNAL_ROI_COUNT} entries),
  all {EXTERNAL_ROI_COUNT} submission masks, the official flat ZIP, and the submission
  audit with per-ROI spacing decisions. External metrics stay pending until the paper
  lead supplies sequestered labels.
- `{MANIFEST_NAME}` - sha256 and size for every archived file.

The gated Virchow2 weights and the dense feature cache are deliberately absent.

## Validation

From a Soma checkout at the commit above:

```bash
python -m examples.beetle.acceptance validate --archive-dir <this directory>
python -m examples.beetle.acceptance report --archive-dir <this directory> --output-dir <report directory>
```
"""


def assemble_archive(
    *,
    run_dirs: Mapping[str, str | Path],
    resolved_dir: str | Path,
    hardware_preflight: str | Path,
    oof_report: str | Path,
    arm_selection: str | Path,
    roi_sidecar: str | Path,
    masks_dir: str | Path,
    submission_zip: str | Path,
    submission_audit: str | Path,
    archive_dir: str | Path,
    external_report: str | Path | None = None,
) -> Path:
    """Collect the campaign artifacts into a validated self-describing archive."""
    if set(run_dirs) != set(ARM_NAMES):
        raise ValueError(f"BEETLE assembly requires run directories for {list(ARM_NAMES)}")
    archive_dir = Path(archive_dir)
    if archive_dir.exists() and any(archive_dir.iterdir()):
        raise ValueError(f"BEETLE archive directory must be empty: {archive_dir}")
    resolved_dir = Path(resolved_dir)

    copies: list[tuple[Path, str]] = [
        (Path(launch.ENCODER_LOCK_PATH), ENCODER_LOCK_ARCHIVE_PATH),
        (resolved_dir / "protocol_resolution.json", PROTOCOL_RESOLUTION_ARCHIVE_PATH),
        (Path(hardware_preflight), HARDWARE_PREFLIGHT_ARCHIVE_PATH),
        (Path(oof_report), OOF_REPORT_ARCHIVE_PATH),
        (Path(arm_selection), ARM_SELECTION_ARCHIVE_PATH),
        (Path(roi_sidecar), ROI_SIDECAR_ARCHIVE_PATH),
        (Path(submission_audit), SUBMISSION_AUDIT_ARCHIVE_PATH),
        (Path(submission_zip), SUBMISSION_ZIP_ARCHIVE_PATH),
    ]
    for arm in ARM_NAMES:
        run_dir = Path(run_dirs[arm])
        copies.append((resolved_dir / f"{arm}.yaml", arm_config_archive_path(arm)))
        copies.append(
            (run_dir / RUN_METADATA_FILENAME, arm_run_metadata_archive_path(arm))
        )
        for fold in range(NUM_FOLDS):
            fold_dir = run_dir / f"fold_{fold}"
            copies.append(
                (fold_dir / CHECKPOINT_FILENAME, checkpoint_archive_path(arm, fold))
            )
            copies.append(
                (fold_dir / HISTORY_FILENAME, history_archive_path(arm, fold))
            )
            copies.append(
                (
                    fold_dir / SAMPLER_AUDIT_FILENAME,
                    sampler_audit_archive_path(arm, fold),
                )
            )
    if external_report is not None:
        copies.append((Path(external_report), EXTERNAL_REPORT_ARCHIVE_PATH))

    problems = [
        f"required input for {rel} is missing: {source}"
        for source, rel in copies
        if not source.is_file()
    ]
    masks_dir = Path(masks_dir)
    try:
        records = load_roi_sidecar(roi_sidecar)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        records = None
        problems.append(f"ROI sidecar {roi_sidecar} failed validation: {exc}")
    if records is not None:
        for record in records:
            source = masks_dir / record.roi_filename
            if source.is_file():
                copies.append(
                    (source, f"{EXTERNAL_MASKS_DIR}/{record.roi_filename}")
                )
            else:
                problems.append(
                    f"required input for {EXTERNAL_MASKS_DIR}/"
                    f"{record.roi_filename} is missing: {source}"
                )
    if problems:
        raise ValueError(
            "BEETLE handoff assembly refused:\n" + "\n".join(f"- {p}" for p in problems)
        )

    for source, rel in copies:
        destination = archive_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    selection_failures: list[str] = []
    selection = _load_json_object(
        archive_dir / ARM_SELECTION_ARCHIVE_PATH, ARM_SELECTION_ARCHIVE_PATH, selection_failures
    )
    selected_arm = str((selection or {}).get("selected_arm", "unknown"))
    run_metadata = yaml.safe_load(
        (archive_dir / arm_run_metadata_archive_path(ARM_NAMES[0])).read_text(
            encoding="utf-8"
        )
    )
    soma_commit_sha = str((run_metadata or {}).get("git_sha", "")) or "unknown"
    (archive_dir / README_NAME).write_text(
        _readme_text(selected_arm=selected_arm, soma_commit_sha=soma_commit_sha),
        encoding="utf-8",
    )
    write_artifact_manifest(archive_dir)

    outcome = validate_archive(archive_dir)
    if not outcome.passed:
        raise ValueError(
            "BEETLE assembled archive failed acceptance validation:\n"
            + "\n".join(f"- {failure}" for failure in outcome.failures)
        )
    return archive_dir


# --- acceptance report ----------------------------------------------------------


ARTIFACT_GROUPS: tuple[tuple[str, str, tuple[str, ...], tuple[str, ...]], ...] = (
    (
        "training_code_and_configs",
        "Reviewer request: full training configurations for the alternative "
        "pretrained pathology encoder; the public Soma commit carries the code.",
        (
            arm_config_archive_path(ARM_NAMES[0]),
            arm_config_archive_path(ARM_NAMES[1]),
            PROTOCOL_RESOLUTION_ARCHIVE_PATH,
        ),
        ("resolved_arm_configs",),
    ),
    (
        "encoder_identity",
        "Reviewer request: unambiguous frozen Virchow2 revision and weight checksum "
        "without redistributing the gated weights.",
        (ENCODER_LOCK_ARCHIVE_PATH, HARDWARE_PREFLIGHT_ARCHIVE_PATH),
        ("encoder_identity", "hardware_preflight"),
    ),
    (
        "environment_and_code_revision",
        "Collaborator expectation: pinned environment, cohort/fold hashes, and the "
        "clean public Soma commit SHA.",
        tuple(arm_run_metadata_archive_path(arm) for arm in ARM_NAMES),
        ("environment_provenance",),
    ),
    (
        "decoder_checkpoints",
        "Reviewer request: model weights - all ten fold-selected decoder "
        "checkpoints for both development arms.",
        (f"{CHECKPOINTS_DIR}/",),
        ("decoder_checkpoints",),
    ),
    (
        "training_evidence",
        "Collaborator expectation: per-fold training histories and ROI-sampler "
        "audits for both arms.",
        (f"{HISTORIES_DIR}/", f"{SAMPLER_AUDITS_DIR}/"),
        ("training_evidence",),
    ),
    (
        "development_results",
        "Reviewer request: five-fold development results on the organizer folds, "
        "patient-level confusion matrices, and bootstrap uncertainty inputs.",
        (OOF_REPORT_ARCHIVE_PATH,),
        ("patient_matrix_coverage",),
    ),
    (
        "arm_selection",
        "Collaborator expectation: the sampling arm was selected from development "
        "evidence only; the External set stayed sequestered.",
        (ARM_SELECTION_ARCHIVE_PATH,),
        ("arm_selection",),
    ),
    (
        "external_submission",
        "Reviewer request: valid External-evaluation PNG masks, the official flat "
        "ZIP, and per-ROI spacing decisions proving no tissue was upsampled.",
        (
            ROI_SIDECAR_ARCHIVE_PATH,
            f"{EXTERNAL_MASKS_DIR}/",
            SUBMISSION_ZIP_ARCHIVE_PATH,
            SUBMISSION_AUDIT_ARCHIVE_PATH,
        ),
        ("external_mask_manifest", "submission_zip", "selected_external_models"),
    ),
    (
        "external_metrics",
        "External Dice and confidence intervals are pending unless the paper lead "
        "supplies sequestered labels or returns evaluation results.",
        (EXTERNAL_REPORT_ARCHIVE_PATH,),
        ("external_metrics",),
    ),
    (
        "integrity",
        "Collaborator expectation: every archived file has a checksum, and the "
        "gated Virchow2 weights and dense feature cache are absent.",
        (MANIFEST_NAME,),
        (
            "manifest_coverage",
            "file_checksums",
            "required_artifacts",
            "excluded_payloads",
        ),
    ),
)


def build_acceptance_report(archive_dir: str | Path) -> dict:
    """Summarize validation and map artifact groups to the reviewer request."""
    archive_dir = Path(archive_dir)
    outcome = validate_archive(archive_dir)
    checks = {check.name: check for check in outcome.checks}

    soma_commit_sha = None
    metadata_path = archive_dir / arm_run_metadata_archive_path(ARM_NAMES[0])
    if metadata_path.is_file():
        try:
            payload = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) or {}
            candidate = str(payload.get("git_sha", ""))
            if _SHA1_PATTERN.match(candidate):
                soma_commit_sha = candidate
        except (OSError, yaml.YAMLError):
            soma_commit_sha = None

    selected_arm = None
    try:
        selected_arm = selected_arm_from_file(archive_dir / ARM_SELECTION_ARCHIVE_PATH)
    except (OSError, ValueError, json.JSONDecodeError):
        selected_arm = None

    groups = [
        {
            "group": group,
            "expectation": expectation,
            "artifacts": list(artifacts),
            "checks": list(check_names),
            "complete": all(checks[name].passed for name in check_names),
        }
        for group, expectation, artifacts, check_names in ARTIFACT_GROUPS
    ]
    return {
        "schema_version": 1,
        "archive_dir": str(archive_dir),
        "passed": outcome.passed,
        "failure_count": len(outcome.failures),
        "soma_commit_sha": soma_commit_sha,
        "selected_arm": selected_arm,
        "external_metrics": (
            "present"
            if (archive_dir / EXTERNAL_REPORT_ARCHIVE_PATH).is_file()
            else "pending"
        ),
        "checks": {
            check.name: {"passed": check.passed, "failures": list(check.failures)}
            for check in outcome.checks
        },
        "artifact_groups": groups,
    }


def _acceptance_report_text(report: dict) -> str:
    lines = [
        "BEETLE handoff acceptance report",
        f"Archive: {report['archive_dir']}",
        f"Result: {'ACCEPTED' if report['passed'] else 'REJECTED'} "
        f"({report['failure_count']} failure(s))",
        f"Soma commit: {report['soma_commit_sha'] or 'not recorded'}",
        f"Selected arm: {report['selected_arm'] or 'not recorded'}",
        f"External metrics: {report['external_metrics']}",
        "",
        "Checks:",
    ]
    for name, check in report["checks"].items():
        lines.append(f"  [{'PASS' if check['passed'] else 'FAIL'}] {name}")
        for failure in check["failures"]:
            lines.append(f"    - {failure}")
    lines.append("")
    lines.append("Artifact groups:")
    for group in report["artifact_groups"]:
        status = "complete" if group["complete"] else "INCOMPLETE"
        lines.append(f"  [{status}] {group['group']}: {group['expectation']}")
        lines.append(f"    artifacts: {', '.join(group['artifacts'])}")
    return "\n".join(lines) + "\n"


def write_acceptance_report(
    archive_dir: str | Path, output_dir: str | Path
) -> dict:
    """Write the JSON and human-readable acceptance report, passing or not."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report = build_acceptance_report(archive_dir)
    (output_dir / "acceptance_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "acceptance_report.txt").write_text(
        _acceptance_report_text(report), encoding="utf-8"
    )
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    assemble = subparsers.add_parser(
        "assemble", help="collect the campaign artifacts into a validated archive"
    )
    assemble.add_argument("--uniform-run-dir", type=Path, required=True)
    assemble.add_argument("--class-conditioned-run-dir", type=Path, required=True)
    assemble.add_argument(
        "--resolved-dir",
        type=Path,
        required=True,
        help="launch-resolve output with both arm YAMLs and protocol_resolution.json",
    )
    assemble.add_argument("--hardware-preflight", type=Path, required=True)
    assemble.add_argument("--oof-report", type=Path, required=True)
    assemble.add_argument("--selection", type=Path, required=True)
    assemble.add_argument("--roi-sidecar", type=Path, required=True)
    assemble.add_argument("--masks-dir", type=Path, required=True)
    assemble.add_argument("--submission-zip", type=Path, required=True)
    assemble.add_argument("--submission-audit", type=Path, required=True)
    assemble.add_argument("--external-report", type=Path, default=None)
    assemble.add_argument("--archive-dir", type=Path, required=True)

    validate = subparsers.add_parser(
        "validate", help="prove completeness and integrity of an unpacked archive"
    )
    validate.add_argument("--archive-dir", type=Path, required=True)

    report = subparsers.add_parser(
        "report", help="write the acceptance report, whether or not the archive passes"
    )
    report.add_argument("--archive-dir", type=Path, required=True)
    report.add_argument("--output-dir", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "assemble":
        archive_dir = assemble_archive(
            run_dirs={
                "uniform": args.uniform_run_dir,
                "class_conditioned": args.class_conditioned_run_dir,
            },
            resolved_dir=args.resolved_dir,
            hardware_preflight=args.hardware_preflight,
            oof_report=args.oof_report,
            arm_selection=args.selection,
            roi_sidecar=args.roi_sidecar,
            masks_dir=args.masks_dir,
            submission_zip=args.submission_zip,
            submission_audit=args.submission_audit,
            archive_dir=args.archive_dir,
            external_report=args.external_report,
        )
        print(archive_dir)
        return 0
    if args.command == "validate":
        outcome = validate_archive(args.archive_dir)
        for failure in outcome.failures:
            print(f"FAIL: {failure}")
        if outcome.passed:
            print(
                f"BEETLE archive {args.archive_dir} passed all "
                f"{len(outcome.checks)} acceptance checks"
            )
            return 0
        print(
            f"BEETLE archive {args.archive_dir} failed acceptance validation with "
            f"{len(outcome.failures)} failure(s)"
        )
        return 1
    report_payload = write_acceptance_report(args.archive_dir, args.output_dir)
    print(Path(args.output_dir) / "acceptance_report.json")
    return 0 if report_payload else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ARTIFACT_GROUPS",
    "CheckOutcome",
    "EXPECTED_CHECKPOINT_COUNT",
    "MANIFEST_NAME",
    "SELECTED_CHECKPOINT_COUNT",
    "ValidationOutcome",
    "assemble_archive",
    "build_acceptance_report",
    "main",
    "required_archive_paths",
    "validate_archive",
    "write_acceptance_report",
    "write_artifact_manifest",
]
