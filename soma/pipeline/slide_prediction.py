"""``soma predict`` and :func:`predict_slides`: segmentation label maps and detection
points over whole slides.

A prediction config has three sections::

    predict:
      output_dir: predictions/beetle-test
      bundles: [runs/beetle/fold_0/bundle, runs/beetle/fold_1/bundle]   # or one run directory
      ensemble: false
      batch_size: 8
      num_workers: 0
      devices: [0, 1]
    cohort:
      records: dataset.csv
      splits: splits.csv          # optional
      subset: test                # which split's slides; requires splits
      fold: 0
    extraction:                   # the features.extraction schema
      tiling: {overlap: 0.5}      # tile size, spacing and slide reader come from the bundle
      segmentation: {method: hsv}
      filter: null
      masks: {min_coverage: {tissue: 0.01}}

It resolves to a plan, written as ``plan.yaml`` in the output directory before any slide
is predicted: every path absolute, a run directory expanded to its bundles, and the
extraction spelled in full, with the tile size, the read spacing, the read tolerance and
the slide reader (``tiling.backend``) taken from the bundles. ``soma predict
<output_dir>/plan.yaml`` replays or resumes it.

The output directory holds ``plan.yaml``, ``provenance.json`` (versions and bundle
identities), one directory per bundle (``ensemble`` with ``ensemble: true``) with one
prediction file and one record (``<sample_id>.json``) per slide, and ``index.parquet``,
assembled from the records at the end. A segmentation bundle writes a label TIFF per slide
(:mod:`soma.evaluation.whole_slide`), a detection bundle a points parquet
(:mod:`soma.evaluation.whole_slide_points`). A slide is done when each of its records
exists and parses; a relaunch predicts the others only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from soma._internal.atomic_io import atomic_write_bytes, atomic_write_text
from soma.config import hydrate, plain

logger = logging.getLogger(__name__)

__all__ = ["predict_slides", "resolve_slide_prediction", "slide_prediction_kwargs"]

PLAN_FILENAME = "plan.yaml"
PROVENANCE_FILENAME = "provenance.json"
INDEX_FILENAME = "index.parquet"
#: The directory of the one map ``ensemble: true`` writes per slide.
ENSEMBLE_NAME = "ensemble"
PREDICT_KEYS = ("output_dir", "bundles", "ensemble", "batch_size", "num_workers", "devices")
COHORT_KEYS = ("records", "splits", "subset", "fold")
#: Plan keys that only place the work: a relaunch may change them.
PLACEMENT_KEYS = ("devices", "num_workers")
#: The tile fields the bundles decide.
BUNDLE_TILING_KEYS = ("requested_tile_size_px", "requested_spacing_um")
#: The task families soma predict writes a prediction for, one file kind each.
FAMILIES = ("segmentation", "detection")
#: The ``extraction.segmentation`` paths, stored absolute in the plan.
SAM2_PATH_KEYS = ("sam2_checkpoint_path", "sam2_config_path")
_DEFAULT_PIXEL_MAPPING = {"background": [0], "tissue": [1]}


# --- the entry points ------------------------------------------------------------------ #


def predict_slides(
    *,
    output_dir: str | Path,
    bundles: str | Path | Sequence[str | Path],
    records: str | Path,
    splits: str | Path | None = None,
    subset: str | None = None,
    fold: int | None = None,
    ensemble: bool = False,
    batch_size: int = 8,
    num_workers: int = 0,
    devices: Sequence[str | int] | None = None,
    extraction: Mapping[str, Any] | Any = None,
) -> pd.DataFrame:
    """Predict over every slide of a cohort: one file per slide and bundle, a label TIFF
    for a segmentation bundle and a points parquet for a detection bundle.

    One keyword per config key (see the module docstring for the config): ``output_dir``,
    ``bundles``, ``ensemble``, ``batch_size``, ``num_workers`` and ``devices`` are the
    ``predict`` section, ``records``, ``splits``, ``subset`` and ``fold`` the ``cohort``
    section, and ``extraction`` the ``extraction`` section (a mapping, or a
    ``slide2vec.PreprocessingConfig``). Relative paths resolve against the working
    directory.

    ``bundles`` are segmentation or detection bundles (:func:`soma.save`), all of one
    family, that agree on tile size and read spacing, or one run directory, which stands
    for its bundles. Each slide is cut into hs2p's slide grid with
    ``extraction.tiling.overlap``; a tile is kept when its row's ``mask_path``, or else
    hs2p's tissue segmentation, covers at least ``extraction.masks.min_coverage`` of it.
    Kept tiles are read at the bundles' read spacing and run through each bundle's model,
    behind one encoder forward for the bundles that share an encoder recipe.

    Segmentation: each map pixel takes the class of one tile
    (:mod:`soma.evaluation.whole_slide`), stored as class + 1; 0 is "not predicted".
    Detection: each tile keeps the peaks in the region it contributes, lifted to level-0
    pixels, and one per-class NMS over the slide removes the duplicates across seams
    (:mod:`soma.evaluation.whole_slide_points`). The parquet holds every peak above the
    head's ``min_score``, with columns ``x``, ``y``, ``class`` and ``score``; the bundle's
    tune threshold is recorded, not applied. With an overlap, the margin
    (``tile * overlap / 2``) must be at least the head's NMS distance.

    The slides are read with the reader the bundles were trained with (the ``backend``
    their encoder recipe records), unless ``extraction.tiling.backend`` names one. Bundles
    that record different readers are refused unless it does, and a reader other than a
    bundle's is used with a warning. A bundle that records no reader leaves the choice to
    the others, and ``auto`` when none records one. A ``slide2vec.PreprocessingConfig``
    always names its reader. The masks are read with ``extraction.tiling.mask_backend``.

    Without ``splits`` every row of ``records`` is predicted; with it, the ``subset``
    split (default ``test``) of fold ``fold`` (default 0). ``ensemble`` writes one file per
    slide instead of one per bundle: the argmax of the bundles' mean softmax, or the peaks
    of their mean heatmap, with no threshold recorded.
    ``devices`` (``[0, 1]``, ``["cpu"]``) predicts in one new process per device, the
    slides split across them; by default the slides are predicted in this process.

    Returns the index of the files, also written as ``index.parquet``: one row per file
    with its ``bundle`` directory, ``sample_id``, ``path`` (relative to ``output_dir``),
    ``width``, ``height``, ``spacing_um``, ``read_spacing_um`` and ``num_tiles``, then
    ``not_predicted_fraction`` for label maps or ``num_points`` for points. ``width``,
    ``height`` and ``spacing_um`` are the frame of the file: the map's pixels, or the
    slide's level 0 for points.
    """
    plan = resolve_slide_prediction(
        output_dir=output_dir,
        bundles=bundles,
        records=records,
        splits=splits,
        subset=subset,
        fold=fold,
        ensemble=ensemble,
        batch_size=batch_size,
        num_workers=num_workers,
        devices=devices,
        extraction=extraction,
    )
    return _run(plan)


def slide_prediction_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    """The :func:`predict_slides` keywords of a parsed prediction config (or of a
    ``plan.yaml``): its ``predict`` and ``cohort`` keys, and ``extraction``. Raises
    ``ValueError`` for an unknown section or key, or a missing ``predict.output_dir``,
    ``predict.bundles`` or ``cohort.records``."""
    if not isinstance(config, Mapping):
        raise ValueError(f"a prediction config is a mapping of sections, got {type(config).__name__}")
    sections = {"predict": PREDICT_KEYS, "cohort": COHORT_KEYS}
    unknown = sorted(set(config) - {*sections, "extraction"})
    if unknown:
        raise ValueError(f"a prediction config has no section(s) {unknown}; its sections are predict, cohort, extraction")
    kwargs: dict[str, Any] = {}
    for name, keys in sections.items():
        section = config.get(name) or {}
        if not isinstance(section, Mapping):
            raise ValueError(f"{name} must be a mapping, got {type(section).__name__}")
        extra = sorted(set(section) - set(keys))
        if extra:
            raise ValueError(f"{name} has no key(s) {extra}; its keys are {list(keys)}")
        kwargs.update(section)
    kwargs["extraction"] = config.get("extraction")
    for required, section in (("output_dir", "predict"), ("bundles", "predict"), ("records", "cohort")):
        if kwargs.get(required) is None:
            raise ValueError(f"{section}.{required} is required")
    return kwargs


# --- the plan --------------------------------------------------------------------------- #


def resolve_slide_prediction(
    *,
    output_dir,
    bundles,
    records,
    splits=None,
    subset=None,
    fold=None,
    ensemble=False,
    batch_size=8,
    num_workers=0,
    devices=None,
    extraction=None,
) -> dict[str, Any]:
    """The plan of a prediction: the config with every value resolved (see the module)."""
    from soma.evaluation.whole_slide import MAX_CLASSES, stitch_margin_px

    if not isinstance(ensemble, bool):
        raise ValueError(f"predict.ensemble must be true or false, got {ensemble!r}")
    for name, value, minimum in (("batch_size", batch_size, 1), ("num_workers", num_workers, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"predict.{name} must be an integer >= {minimum}, got {value!r}")
    if devices is not None:
        from soma.pipeline.parallel import resolve_devices

        devices = list(resolve_devices(list(devices)))
    bundle_dirs = _bundle_dirs(bundles)
    infos = [_bundle_info(path) for path in bundle_dirs]
    family = _shared_family(infos)
    tile_px, spacing_um, tolerance = _shared_tile(infos)
    if ensemble:
        _check_ensemble(infos, family)
    for info in infos:
        if family == "segmentation" and info["num_classes"] > MAX_CLASSES:
            raise ValueError(
                f"bundle {info['path']} has {info['num_classes']} classes; a label map holds "
                f"at most {MAX_CLASSES} (class k is stored as k + 1 in uint8)"
            )
    preprocessing = _extraction(extraction, tile_px=tile_px, spacing_um=spacing_um, tolerance=tolerance, infos=infos)
    overlap = float(preprocessing.tiling.overlap)
    margin = stitch_margin_px(tile_px, overlap)
    if family == "detection" and overlap > 0:
        _check_margin(infos, tile_px=tile_px, overlap=overlap, margin_px=margin, spacing_um=spacing_um)
    return {
        "predict": {
            "output_dir": str(_absolute(output_dir)),
            "bundles": [str(path) for path in bundle_dirs],
            "ensemble": ensemble,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
        },
        "cohort": _cohort(records, splits, subset, fold),
        "extraction": plain(preprocessing),
    }


def _absolute(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _bundle_dirs(bundles) -> list[Path]:
    """The bundle directories ``bundles`` names: bundles, or one run directory's bundles."""
    from soma.learn.bundle import MODEL

    entries = [bundles] if isinstance(bundles, (str, Path)) else list(bundles or ())
    if not entries:
        raise ValueError("predict.bundles must name at least one bundle")
    found: list[Path] = []
    for entry in entries:
        path = _absolute(entry)
        if (path / MODEL).is_file():
            found.append(path)
            continue
        folds = sorted((candidate.parent for candidate in path.glob(f"fold_*/bundle/{MODEL}")), key=_fold_number)
        if folds:
            found.extend(folds)
        elif (path / "bundle" / MODEL).is_file():
            found.append(path / "bundle")
        else:
            raise ValueError(
                f"predict.bundles: {entry} is neither a bundle (it has no {MODEL}) nor a run "
                "directory with bundles (bundle/ or fold_<k>/bundle/)"
            )
    repeated = sorted({str(path) for path in found if found.count(path) > 1})
    if repeated:
        raise ValueError(f"predict.bundles names a bundle more than once: {repeated}")
    return found


def _fold_number(bundle: Path) -> tuple[int, str]:
    suffix = bundle.parent.name.split("_", 1)[1]
    return (int(suffix), "") if suffix.isdigit() else (-1, suffix)


def _bundle_info(path: Path) -> dict[str, Any]:
    """What the plan needs from a bundle, read from its ``preprocessing.json``."""
    from soma.learn.bundle import PREPROCESSING

    preprocessing = json.loads((path / PREPROCESSING).read_text(encoding="utf-8"))
    where = f"bundle {path}"
    if preprocessing.get("shape") != "grid":
        raise ValueError(f"{where} is not a grid bundle; soma predict needs a segmentation or detection bundle")
    family = preprocessing.get("task_family")
    if family not in FAMILIES:
        raise ValueError(f"{where} is a {family} bundle; soma predict needs a segmentation or detection bundle")
    recipe = preprocessing.get("encoder")
    if isinstance(recipe, Mapping) and recipe.get("composite"):
        raise ValueError(
            f"{where} was trained on a composite of several encoders' grids. Whole-slide "
            "prediction needs a single frozen encoder stage, and a composite has none."
        )
    tile = preprocessing.get("tile_size")
    tile = [tile, tile] if isinstance(tile, int) else list(tile or ())
    if len(tile) != 2 or tile[0] != tile[1]:
        raise ValueError(f"{where} records the tile size {preprocessing.get('tile_size')}; soma predict needs square tiles")
    spacing = preprocessing.get("spacing_um")
    if spacing is None and isinstance(recipe, Mapping):
        spacing = recipe.get("spacing_um")
    if spacing is None:
        raise ValueError(f"{where} records no read spacing (spacing_um)")
    tolerance = recipe.get("tolerance", 0.05) if isinstance(recipe, Mapping) else 0.05
    backend = recipe.get("backend") if isinstance(recipe, Mapping) else None
    info = {
        "path": str(path),
        "task_family": family,
        "tile_px": int(tile[0]),
        "spacing_um": float(spacing),
        "tolerance": float(tolerance),
        "backend": None if backend is None else str(backend),
        "num_classes": int(preprocessing["num_classes"]),
        "class_names": preprocessing.get("class_names"),
    }
    if family == "segmentation":
        info["ignore_index"] = int(preprocessing.get("ignore_index", 255))
        info["label_remap"] = preprocessing.get("label_remap")
    else:
        head = _detection_head(path)
        info["match_distance_um"] = float(head.match_distance)
        info["nms_distance_um"] = float(head.nms_distance)
        info["min_score"] = float(head.min_score)
    return info


def _detection_head(path: Path):
    """A detection bundle's head, built from its ``model.json`` alone: its distances are
    there, and building the whole model would load the encoder."""
    from soma.learn.bundle import MODEL, _decode
    from soma.models.heads.registry import task_registry

    recipe = json.loads((path / MODEL).read_text(encoding="utf-8"))
    spec = (recipe.get("params") or {}).get("head")
    if not isinstance(spec, Mapping) or "name" not in spec:
        raise ValueError(f"bundle {path} records no head in its {MODEL}, so soma predict cannot read its NMS distance")
    spec = _decode(dict(spec))
    return task_registry.resolve(spec["name"])(**spec.get("params", {}))


def _shared_family(infos: Sequence[Mapping[str, Any]]) -> str:
    """The task family every bundle shares: one call writes one kind of file."""
    families = {info["task_family"] for info in infos}
    if len(families) > 1:
        raise ValueError(
            "the bundles of one prediction must be of one task family; these are "
            f"{_listing(infos, ('task_family',))}"
        )
    return families.pop()


#: What the bundles of an ensemble must share, per family: one file records one of each.
_ENSEMBLE_KEYS = {
    "segmentation": ("num_classes", "class_names", "ignore_index", "label_remap"),
    "detection": ("num_classes", "class_names", "match_distance_um", "nms_distance_um", "min_score"),
}


def _check_ensemble(infos: Sequence[Mapping[str, Any]], family: str) -> None:
    keys = _ENSEMBLE_KEYS[family]
    if len({json.dumps([info[key] for key in keys]) for info in infos}) <= 1:
        return
    if family == "segmentation":
        raise ValueError(
            "predict.ensemble averages the bundles' softmax, so they must share one class "
            "scheme: the classes, the raw annotation values that form them (label_remap) "
            f"and the values ignored; these bundles have {_listing(infos, keys[:3])}"
        )
    raise ValueError(
        "predict.ensemble averages the bundles' heatmaps and takes one set of peaks from the "
        "mean, so they must share the classes, the matching and NMS distances and min_score; "
        f"these bundles have {_listing(infos, keys)}"
    )


def _check_margin(infos, *, tile_px: int, overlap: float, margin_px: int, spacing_um: float) -> None:
    """Refuse a margin narrower than a detection bundle's NMS distance: a tile keeps the
    peaks in its middle only, and each of them must be found with its whole NMS
    neighbourhood inside the tile."""
    import math

    margin_um = margin_px * spacing_um
    for info in infos:
        nms_um = info["nms_distance_um"]
        if margin_um >= nms_um * (1 - 1e-9):
            continue
        needed = 2 * math.ceil(nms_um / spacing_um - 1e-9) / tile_px
        hint = (
            f"Use an overlap of at least {needed:g}, or 0."
            if needed < 1
            else "No overlap below 1 leaves a margin that wide: use an overlap of 0."
        )
        raise ValueError(
            f"extraction.tiling.overlap {overlap:g} leaves a margin of {margin_px} px "
            f"({margin_um:g} um) on each side of a {tile_px} px tile, less than the NMS "
            f"distance of bundle {info['path']} ({nms_um:g} um). A tile keeps the peaks in "
            "its middle only, and finds each of them with its whole NMS neighbourhood inside "
            f"the tile only when the margin is at least that distance. {hint}"
        )


def _listing(infos: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> str:
    return "; ".join(f"{info['path']}: " + ", ".join(f"{key} {info[key]}" for key in keys) for info in infos)


def _shared_tile(infos: Sequence[Mapping[str, Any]]) -> tuple[int, float, float]:
    """The tile size, read spacing and tolerance every bundle shares."""
    for key in ("tile_px", "spacing_um", "tolerance"):
        if len({info[key] for info in infos}) > 1:
            raise ValueError(
                "the bundles of one prediction must agree on tile size, read spacing and "
                f"read tolerance; these record {_listing(infos, ('tile_px', 'spacing_um', 'tolerance'))}"
            )
    first = infos[0]
    return first["tile_px"], first["spacing_um"], first["tolerance"]


def _slide_reader(configured: str | None, infos: Sequence[Mapping[str, Any]]) -> str:
    """The reader of the slides: ``extraction.tiling.backend``, or else the one the bundles
    record. A bundle that records none (one saved before soma recorded it) leaves the
    choice to the others, and ``auto`` when no bundle records one."""
    recording = [info for info in infos if info["backend"] is not None]
    if configured is None:
        if len({info["backend"] for info in recording}) > 1:
            raise ValueError(
                "the bundles of one prediction were trained on slides read with different "
                f"readers: {_readers(recording)}. Name the reader to predict with in "
                "extraction.tiling.backend."
            )
        return recording[0]["backend"] if recording else "auto"
    others = [info for info in recording if info["backend"] != configured]
    if others:
        warnings.warn(
            f"extraction.tiling.backend is {configured!r}, but these bundles were trained on "
            f"slides read with another reader: {_readers(others)}. The slides are read with "
            f"{configured!r}, which may decode their pixels differently than in training.",
            stacklevel=2,
        )
    return str(configured)


def _readers(infos: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(f"{info['path']} with {info['backend']!r}" for info in infos)


def _extraction(given, *, tile_px: int, spacing_um: float, tolerance: float, infos: Sequence[Mapping[str, Any]]):
    """The ``extraction`` section as a ``slide2vec.PreprocessingConfig``: the tile size and
    read spacing from the bundles, the read tolerance and the slide reader from them unless
    set, the slide grid, absolute SAM2 paths, and the binary tissue vocabulary when
    ``masks`` names none."""
    import slide2vec

    data = copy.deepcopy(plain(given)) if given is not None else {}
    if not isinstance(data, dict):
        raise ValueError(f"extraction must be a mapping, got {type(given).__name__}")
    tiling = dict(data.get("tiling") or {})
    for key, value in zip(BUNDLE_TILING_KEYS, (tile_px, spacing_um)):
        if tiling.get(key) is not None and float(tiling[key]) != float(value):
            raise ValueError(
                f"extraction.tiling.{key} is {tiling[key]}, but the tile size and read spacing "
                f"come from the bundles ({tile_px} px at {spacing_um} um/px): remove it from "
                "the config"
            )
    if tiling.get("grid", "slide") != "slide":
        raise ValueError(f"extraction.tiling.grid must be 'slide' for whole-slide prediction, got {tiling['grid']!r}")
    # ``backend`` and ``mask_backend`` name the readers. hs2p tiles with its ``requested_*``
    # record of them, which a saved plan.yaml holds: drop it so hs2p fills it from the plan.
    for key in ("requested_backend", "requested_mask_backend"):
        tiling.pop(key, None)
    data["tiling"] = {
        "tolerance": tolerance,
        "overlap": 0.0,
        **tiling,
        "backend": _slide_reader(tiling.get("backend"), infos),
        "requested_tile_size_px": tile_px,
        "requested_spacing_um": spacing_um,
        "grid": "slide",
    }
    segmentation = data.get("segmentation")
    if isinstance(segmentation, Mapping):
        data["segmentation"] = {
            **segmentation,
            **{key: _absolute(segmentation[key]) for key in SAM2_PATH_KEYS if segmentation.get(key) is not None},
        }
    masks = data.get("masks")
    if isinstance(masks, Mapping) and masks.get("pixel_mapping") is None:
        data["masks"] = {**masks, "pixel_mapping": dict(_DEFAULT_PIXEL_MAPPING)}
    return hydrate(slide2vec.PreprocessingConfig, data, "extraction")


def _cohort(records, splits, subset, fold) -> dict[str, Any]:
    if splits is None:
        given = [name for name, value in (("subset", subset), ("fold", fold)) if value is not None]
        if given:
            raise ValueError(f"cohort.{' and cohort.'.join(given)} select from cohort.splits; set cohort.splits")
        return {"records": str(_absolute(records)), "splits": None, "subset": None, "fold": None}
    if fold is not None and (isinstance(fold, bool) or not isinstance(fold, int) or fold < 0):
        raise ValueError(f"cohort.fold must be an integer >= 0, got {fold!r}")
    return {
        "records": str(_absolute(records)),
        "splits": str(_absolute(splits)),
        "subset": "test" if subset is None else str(subset),
        "fold": 0 if fold is None else fold,
    }


def _comparable(plan: Mapping[str, Any]) -> dict[str, Any]:
    payload = copy.deepcopy(dict(plan))
    for key in PLACEMENT_KEYS:
        (payload.get("predict") or {}).pop(key, None)
    return payload


def _differences(saved: Any, current: Any, where: str = "") -> list[str]:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        found: list[str] = []
        for key in sorted({*saved, *current}, key=str):
            found += _differences(saved.get(key), current.get(key), f"{where}.{key}" if where else str(key))
        return found
    return [] if saved == current else [where]


def _guard_plan(output_dir: Path, plan: Mapping[str, Any]) -> None:
    """Refuse a launch whose plan differs from the one ``output_dir`` was written with."""
    path = output_dir / PLAN_FILENAME
    if not path.is_file():
        return
    saved = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    differing = _differences(_comparable(saved), _comparable(plan))
    if differing:
        raise ValueError(
            f"Refusing to predict into {output_dir}: its {PLAN_FILENAME} differs from the "
            f"current plan (differing keys: {differing}).{_reader_hint(saved, plan, differing)} "
            "Use another predict.output_dir, or the settings it was written with."
        )


def _reader_hint(saved: Mapping[str, Any], plan: Mapping[str, Any], differing: Sequence[str]) -> str:
    """How to keep the slide reader a directory was predicted with, when it differs."""
    if "extraction.tiling.backend" not in differing:
        return ""
    before = ((saved.get("extraction") or {}).get("tiling") or {}).get("backend")
    now = plan["extraction"]["tiling"]["backend"]
    return (
        f" Its slides were read with {before!r}; the current plan reads them with {now!r}, "
        "the reader the bundles record unless the config names one: set "
        f"extraction.tiling.backend: {before} to keep it."
    )


# --- the run ---------------------------------------------------------------------------- #


def _output_names(plan: Mapping[str, Any]) -> list[str]:
    """The directory of each map a slide gets: one per bundle, or ``ensemble``."""
    if plan["predict"]["ensemble"]:
        return [ENSEMBLE_NAME]
    return bundle_names([Path(path) for path in plan["predict"]["bundles"]])


def bundle_names(paths: Sequence[Path]) -> list[str]:
    """One directory name per bundle: the shortest trailing part of its path that tells
    the bundles apart, without a last ``bundle`` (``runs/a/fold_0/bundle`` is ``fold_0``)."""
    parts = []
    for path in paths:
        relative = path.relative_to(path.anchor).parts
        parts.append(relative[:-1] if len(relative) > 1 and relative[-1] == "bundle" else relative)
    for depth in range(1, max(len(p) for p in parts) + 1):
        names = ["-".join(p[-depth:]) for p in parts]
        if len(set(names)) == len(names) and ENSEMBLE_NAME not in names:
            return names
    return [f"bundle_{k}" for k in range(len(paths))]


def _record_path(output_dir: Path, name: str, sample_id: str) -> Path:
    return output_dir / name / f"{sample_id}.json"


def _read_record(path: Path) -> dict[str, Any] | None:
    """A slide's record, or ``None`` when it is missing or does not parse."""
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(record, dict) or not (path.parent / str(record.get("file", ""))).is_file():
        return None
    return record


def _slides(cohort: Mapping[str, Any]) -> list:
    """The cohort's slides to predict, in ``records`` order."""
    from soma.data import Cohort, ImageManifest

    manifest = ImageManifest.from_csv(cohort["records"])
    if cohort["splits"] is None:
        return [manifest[sid] for sid in manifest.sample_ids]
    folds = Cohort.from_csv(cohort["records"], cohort["splits"], targets=[], allow_missing_test=True).folds
    if cohort["fold"] >= len(folds):
        raise ValueError(f"cohort.fold is {cohort['fold']}, but {cohort['splits']} has {len(folds)} fold(s)")
    split = folds[cohort["fold"]]
    subsets = {"train": split.train, "tune": split.tune, **split.tests}
    if cohort["subset"] not in subsets:
        raise ValueError(f"cohort.subset {cohort['subset']!r} is not a split of fold {cohort['fold']}; its splits are {sorted(subsets)}")
    wanted = set(subsets[cohort["subset"]])
    return [manifest[sid] for sid in manifest.sample_ids if sid in wanted]


def _bundle_digest(path: Path) -> str:
    """SHA-256 over a bundle's model recipe, preprocessing and weights."""
    from soma.learn.bundle import MODEL, PREPROCESSING, WEIGHTS

    digest = hashlib.sha256()
    for name in (MODEL, PREPROCESSING, WEIGHTS):
        file = path / name
        if file.is_file():
            digest.update(name.encode())
            with file.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
    return digest.hexdigest()


def _identities(plan: Mapping[str, Any]) -> list[dict[str, str]]:
    paths = [Path(path) for path in plan["predict"]["bundles"]]
    return [
        {"name": name, "path": str(path), "sha256": _bundle_digest(path)}
        for name, path in zip(bundle_names(paths), paths)
    ]


def _guard_bundles(output_dir: Path, identities: Sequence[Mapping[str, str]]) -> None:
    """Refuse a launch whose bundles changed since ``output_dir``'s maps were written: the
    plan names a bundle by its path, ``provenance.json`` by its digest."""
    try:
        saved = json.loads((output_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))["bundles"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    digests = {entry.get("path"): entry.get("sha256") for entry in saved if isinstance(entry, Mapping)}
    for identity in identities:
        before = digests.get(identity["path"])
        if before is not None and before != identity["sha256"]:
            raise ValueError(
                f"Refusing to predict into {output_dir}: bundle {identity['path']} changed since "
                f"its maps were predicted into it (sha256 {before[:12]} then, "
                f"{identity['sha256'][:12]} now). Use another predict.output_dir."
            )


def _write_provenance(output_dir: Path, plan: Mapping[str, Any], identities) -> None:
    import hs2p
    import slide2vec
    import torch

    import soma

    provenance = {
        "soma_version": str(soma.__version__),
        "hs2p_version": str(getattr(hs2p, "__version__", None)),
        "slide2vec_version": str(getattr(slide2vec, "__version__", None)),
        "torch_version": str(torch.__version__),
        "ensemble": plan["predict"]["ensemble"],
        "bundles": identities,
    }
    atomic_write_text(output_dir / PROVENANCE_FILENAME, json.dumps(provenance, indent=2) + "\n")


def _run(plan: dict[str, Any]) -> pd.DataFrame:
    output_dir = Path(plan["predict"]["output_dir"])
    slides = _slides(plan["cohort"])
    _guard_plan(output_dir, plan)
    identities = _identities(plan)
    _guard_bundles(output_dir, identities)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / PLAN_FILENAME, yaml.safe_dump(plan, sort_keys=False))
    _write_provenance(output_dir, plan, identities)
    names = _output_names(plan)
    pending = [
        entry
        for entry in slides
        if any(_read_record(_record_path(output_dir, name, entry.sample_id)) is None for name in names)
    ]
    _check_spacings(pending, plan["extraction"]["tiling"])
    ids = [entry.sample_id for entry in pending]
    logger.info("Predicting %d of %d slide(s) into %s", len(ids), len(slides), output_dir)
    devices = plan["predict"]["devices"]
    if ids and devices is None:
        predict_shard(plan, ids)
    elif ids:
        _predict_on_devices(plan, ids, devices)
    index = _index(output_dir, names, slides, _RECORD_FIELDS[_plan_family(plan)])
    atomic_write_bytes(output_dir / INDEX_FILENAME, _parquet_bytes(index))
    return index


def _plan_family(plan: Mapping[str, Any]) -> str:
    """The task family of the plan's bundles (the plan admits only one)."""
    from soma.learn.bundle import PREPROCESSING

    path = Path(plan["predict"]["bundles"][0]) / PREPROCESSING
    return str(json.loads(path.read_text(encoding="utf-8"))["task_family"])


def _parquet_bytes(frame: pd.DataFrame) -> bytes:
    import io

    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def _check_spacings(slides, tiling: Mapping[str, Any]) -> None:
    """Refuse a slide coarser than the read spacing: soma predict never upsamples one."""
    from hs2p.wsi.wsi import WSI

    spacing, tolerance = float(tiling["requested_spacing_um"]), float(tiling["tolerance"])
    for entry in slides:
        wsi = WSI(path=Path(entry.image_path), backend=tiling["backend"], spacing_at_level_0=entry.spacing_at_level_0)
        level0 = float(wsi.get_level_spacing(0))
        if level0 > spacing * (1.0 + tolerance):
            raise ValueError(
                f"slide '{entry.sample_id}' is at {level0:g} um/px, coarser than the bundles' "
                f"read spacing of {spacing:g} um/px (tolerance {tolerance:g}): soma predict "
                "does not upsample a slide. Remove it from the cohort."
            )


def _index(output_dir: Path, names: Sequence[str], slides, fields: Sequence[str]) -> pd.DataFrame:
    rows = []
    for name in names:
        for entry in slides:
            record = _read_record(_record_path(output_dir, name, entry.sample_id))
            if record is None:
                raise RuntimeError(f"slide '{entry.sample_id}' has no complete record in {output_dir / name}")
            rows.append(
                {
                    "bundle": name,
                    "sample_id": record["sample_id"],
                    "path": f"{name}/{record['file']}",
                    **{key: record[key] for key in fields},
                }
            )
    columns = ["bundle", "sample_id", "path", *fields]
    return pd.DataFrame(rows, columns=columns)


#: The index columns each family's records give, after ``bundle``, ``sample_id`` and ``path``.
_RECORD_FIELDS = {
    "segmentation": ("width", "height", "spacing_um", "read_spacing_um", "num_tiles", "not_predicted_fraction"),
    "detection": ("width", "height", "spacing_um", "read_spacing_um", "num_tiles", "num_points"),
}


def _predict_on_devices(plan: dict[str, Any], ids: list[str], devices: Sequence[str]) -> None:
    from soma.pipeline.parallel import DeviceJob, free_device_memory, run_on_devices

    output_dir = Path(plan["predict"]["output_dir"])
    shards = [ids[k :: len(devices)] for k in range(min(len(devices), len(ids)))]
    jobs = [
        DeviceJob(
            label=f"slides {k}",
            log_path=output_dir / "logs" / f"worker_{k}.log",
            target=predict_shard,
            args=(plan, shard),
        )
        for k, shard in enumerate(shards)
    ]
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    free_device_memory()
    run_on_devices(jobs, devices)


def predict_shard(plan: Mapping[str, Any], sample_ids: Sequence[str]) -> list[str]:
    """Predict the slides ``sample_ids`` of ``plan`` in this process (a device job's
    target); returns the ids predicted."""
    import torch

    output_dir = Path(plan["predict"]["output_dir"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = _models(plan["predict"]["bundles"], device)
    names = _output_names(plan)
    identities = _identities(plan)
    by_id = {entry.sample_id: entry for entry in _slides(plan["cohort"])}
    for sample_id in sample_ids:
        _predict_slide(plan, models, names, identities, by_id[sample_id], output_dir)
    return list(sample_ids)


def _models(bundle_paths: Sequence[str], device) -> list:
    from soma.learn import load
    from soma.models.model import DenseModel
    from soma.pipeline.build import bundle_encoder_stage

    models = []
    for path in bundle_paths:
        bundle = load(path)
        stage = bundle_encoder_stage(bundle)
        model = bundle.model
        if model.encoder is None:
            model = DenseModel(model.decoder, model.head, encoder=stage, adaptor=model.adaptor)
        models.append(model.to(device).eval())
    return models


def _predict_slide(plan, models, names, identities, entry, output_dir: Path) -> None:
    import slide2vec

    extraction = hydrate(slide2vec.PreprocessingConfig, plan["extraction"], "extraction")
    origins, grid = _kept_tiles(entry, extraction)
    for name in names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    detection = models[0].head.task_family == "detection"
    if not origins:
        empty = "its points file has no rows" if detection else "its label map is all zeros, not predicted"
        warnings.warn(
            f"slide '{entry.sample_id}' has no kept tile (no tissue under the mask or the "
            f"tissue segmentation): {empty}",
            stacklevel=2,
        )
    write = _write_slide_points if detection else _write_label_maps
    write(plan, models, names, identities, entry, output_dir, extraction=extraction, origins=origins, grid=grid)
    logger.info("Predicted slide %s: %d tile(s)", entry.sample_id, len(origins))


def _write_label_maps(plan, models, names, identities, entry, output_dir: Path, *, extraction, origins, grid) -> None:
    from soma.evaluation.whole_slide import SlideFrame, predict_slide_label_maps

    tiling = extraction.tiling
    frame = SlideFrame.from_grid(
        tile_px=tiling.requested_tile_size_px,
        overlap=tiling.overlap,
        step_lv0=grid.step_px_lv0,
        slide_size=tuple(grid.tiles.slide_dimensions),
        level0_spacing_um=grid.tiles.base_spacing_um,
    )
    ensemble = plan["predict"]["ensemble"]
    shared = {
        "sample_id": entry.sample_id,
        "ensemble": ensemble,
        "read_spacing_um": float(tiling.requested_spacing_um),
        "tile_size_px": int(tiling.requested_tile_size_px),
        "overlap": float(tiling.overlap),
    }
    if ensemble:
        metadata = [{**shared, **_class_scheme(models[0].head), "bundles": identities}]
    else:
        metadata = [
            {**shared, **_class_scheme(model.head), "bundles": [identity]}
            for model, identity in zip(models, identities)
        ]
    file = f"{entry.sample_id}.tif"
    fractions = predict_slide_label_maps(
        models,
        image=entry,
        origins=origins,
        frame=frame,
        paths=[output_dir / name / file for name in names],
        metadata=metadata,
        spacing_um=float(tiling.requested_spacing_um),
        ensemble=ensemble,
        backend=tiling.backend,
        tolerance=float(tiling.tolerance),
        batch_size=plan["predict"]["batch_size"],
        num_workers=plan["predict"]["num_workers"],
    )
    for name, fraction in zip(names, fractions):
        record = {
            "sample_id": entry.sample_id,
            "file": file,
            "width": frame.width,
            "height": frame.height,
            "spacing_um": frame.spacing_um,
            "read_spacing_um": float(tiling.requested_spacing_um),
            "num_tiles": len(origins),
            "not_predicted_fraction": fraction,
        }
        atomic_write_text(_record_path(output_dir, name, entry.sample_id), json.dumps(record, indent=2) + "\n")


def _write_slide_points(plan, models, names, identities, entry, output_dir: Path, *, extraction, origins, grid) -> None:
    from soma.evaluation.whole_slide_points import predict_slide_points

    tiling = extraction.tiling
    ensemble = plan["predict"]["ensemble"]
    tiles = grid.tiles
    level0_spacing = float(tiles.base_spacing_um)
    width, height = (int(v) for v in tiles.slide_dimensions)
    # hs2p rounds the step in the pixels it reads, so in tile pixels it is not always
    # tile - 2 * margin (on a slide resampled to the read spacing).
    step_px = (
        grid.step_px_lv0
        * level0_spacing
        / float(tiles.read_spacing_um)
        * int(tiling.requested_tile_size_px)
        / int(tiles.read_tile_size_px)
    )
    shared = {
        "sample_id": entry.sample_id,
        "ensemble": ensemble,
        "read_spacing_um": float(tiling.requested_spacing_um),
        "tile_size_px": int(tiling.requested_tile_size_px),
        "overlap": float(tiling.overlap),
    }
    if ensemble:
        # An ensemble's scores are a mean no bundle was tuned on: it has no threshold.
        metadata = [{**shared, "bundles": identities, "threshold": None}]
    else:
        metadata = [
            {**shared, "bundles": [identity], "threshold": _tune_threshold(identity["path"])}
            for identity in identities
        ]
    file = f"{entry.sample_id}.parquet"
    counts = predict_slide_points(
        models,
        image=entry,
        origins=origins,
        tile_px=int(tiling.requested_tile_size_px),
        step_px=step_px,
        overlap=float(tiling.overlap),
        slide_size=(width, height),
        level0_spacing_um=level0_spacing,
        paths=[output_dir / name / file for name in names],
        metadata=metadata,
        spacing_um=float(tiling.requested_spacing_um),
        ensemble=ensemble,
        backend=tiling.backend,
        tolerance=float(tiling.tolerance),
        batch_size=plan["predict"]["batch_size"],
        num_workers=plan["predict"]["num_workers"],
    )
    for name, count in zip(names, counts):
        record = {
            "sample_id": entry.sample_id,
            "file": file,
            "width": width,
            "height": height,
            "spacing_um": level0_spacing,
            "read_spacing_um": float(tiling.requested_spacing_um),
            "num_tiles": len(origins),
            "num_points": count,
        }
        atomic_write_text(_record_path(output_dir, name, entry.sample_id), json.dumps(record, indent=2) + "\n")


def _tune_threshold(bundle_path: str):
    """The operating threshold a detection bundle recorded: a float, or one per class
    name; ``None`` when it records none."""
    from soma.learn.bundle import PREPROCESSING

    preprocessing = json.loads((Path(bundle_path) / PREPROCESSING).read_text(encoding="utf-8"))
    return preprocessing.get("score_threshold")


def _value_names(head) -> dict[str, str]:
    """The map's value-to-name mapping: 0 is not predicted, class ``k`` is ``k + 1``."""
    return {"0": "not predicted", **{str(k + 1): str(name) for k, name in enumerate(head.classes)}}


def _class_scheme(head) -> dict[str, Any]:
    """What a map records to be scored as the head was trained.

    ``values`` names each stored value. ``classes`` lists, in class order, the raw
    annotation values each class is made of, and ``ignore`` the raw values left out of loss
    and metrics. ``pixel_mapping`` is the vocabulary hs2p reads an annotation raster
    against. Without a class scheme, the head reads class ``k`` from raw value ``k``.
    """
    import numpy as np

    remap, ignore_index = head.label_remap, int(head.ignore_index)
    if remap is None:
        classes = {str(name): [k] for k, name in enumerate(head.classes)}
        ignore = [ignore_index] if 0 <= ignore_index <= 255 else []
    else:
        remap = np.asarray(remap)
        classes = {str(name): np.flatnonzero(remap == k).tolist() for k, name in enumerate(head.classes)}
        ignore = np.flatnonzero(remap == ignore_index).tolist()
    pixel_mapping = {
        str(name): [int(v) for v in value] if isinstance(value, (list, tuple)) else int(value)
        for name, value in head.mask_vocabulary.items()
    }
    return {"values": _value_names(head), "classes": classes, "ignore": ignore, "pixel_mapping": pixel_mapping}


def _kept_tiles(entry, extraction):
    """The level-0 origins of the slide's kept tiles on the slide grid, and one hs2p
    tiling result (for the grid's stride and the slide's size)."""
    from hs2p import SlideSpec, tile_slide

    from soma.extract.preprocessing import is_annotation_sampling
    from soma.models.encoders.validation import resolve_filter_config

    tiling = extraction.tiling
    result = tile_slide(
        SlideSpec(
            sample_id=entry.sample_id,
            image_path=entry.image_path,
            mask_path=entry.mask_path,
            spacing_at_level_0=entry.spacing_at_level_0,
        ),
        tiling=tiling,
        segmentation=extraction.segmentation,
        filtering=resolve_filter_config(
            extraction.filter,
            tile_size_px=tiling.requested_tile_size_px,
            annotation_sampling=is_annotation_sampling(extraction.masks),
        ),
        masks=extraction.masks,
    )
    results = list(result.values()) if isinstance(result, dict) else [result]
    if not results:
        raise RuntimeError(f"hs2p returned no tiling for slide '{entry.sample_id}'")
    origins = {
        (int(x), int(y)) for grid in results for x, y in zip(grid.tiles.x.tolist(), grid.tiles.y.tolist())
    }
    return sorted(origins, key=lambda origin: (origin[1], origin[0])), results[0]
