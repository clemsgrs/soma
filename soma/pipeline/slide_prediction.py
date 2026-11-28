"""``soma predict`` and :func:`predict_slides`: segmentation label maps and detection
points over whole slides.

A prediction config has three sections::

    predict:
      output_dir: predictions/beetle-test
      bundles: [runs/beetle/fold_0/bundle, runs/beetle/fold_1/bundle]   # or one run directory
      ensemble: false
      stitch: pick_one            # or average (segmentation only)
      restrict_to_mask: false     # true: 0 outside the mask (segmentation only)
      batch_size: 8
      num_workers: 0
      devices: [0, 1]
      split: slides               # or tiles: each slide's tiles across the devices
    cohort:
      records: dataset.csv
      splits: splits.csv          # optional
      subset: test                # which split's slides; requires splits
      fold: 0                     # or bundle: each bundle's own fold
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
prediction file and one record (``<sample_id>.json``) per slide it predicts, and
``index.parquet``, assembled from the records at the end. A segmentation bundle writes a
label TIFF per slide (:mod:`soma.evaluation.whole_slide`), a detection bundle a points parquet
(:mod:`soma.evaluation.whole_slide_points`). A slide is done when each of its records
exists and parses; a relaunch predicts the others only. A relaunch may change
``devices``, ``num_workers``, ``split`` and ``batch_size``; each record holds the
``batch_size`` its slide was predicted with.

With ``cohort.fold: bundle`` each bundle predicts the ``subset`` split of its own fold, the
fold its ``provenance.json`` records, after a check that the fold's assignments in
``cohort.splits`` are the ones it was trained on (``fold_assignments``). The work list is
the union of the bundles' slides, in ``records`` order; each slide runs through its own
bundles only.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import warnings
from collections.abc import Mapping, Sequence
from contextlib import contextmanager, nullcontext
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
PREDICT_KEYS = (
    "output_dir", "bundles", "ensemble", "stitch", "restrict_to_mask", "batch_size", "num_workers", "devices", "split"
)
COHORT_KEYS = ("records", "splits", "subset", "fold")
#: The ``cohort.fold`` that gives each bundle the slides of its own fold.
BUNDLE_FOLD = "bundle"
#: Plan keys that only place the work: a relaunch may change them.
PLACEMENT_KEYS = ("devices", "num_workers", "split")
#: Plan keys a relaunch may change although a map can depend on them slightly: each
#: slide's record holds the value it was predicted with.
RECORDED_KEYS = ("batch_size",)
#: The tile fields the bundles decide.
BUNDLE_TILING_KEYS = ("requested_tile_size_px", "requested_spacing_um")
#: What ``predict.split`` deals to the devices: whole slides (the default) or one slide's tiles.
SPLITS = ("slides", "tiles")
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
    fold: int | str | None = None,
    ensemble: bool = False,
    stitch: str = "pick_one",
    restrict_to_mask: bool = False,
    batch_size: int = 8,
    num_workers: int = 0,
    devices: Sequence[str | int] | None = None,
    split: str = "slides",
    extraction: Mapping[str, Any] | Any = None,
) -> pd.DataFrame:
    """Predict over every slide of a cohort: one file per slide and bundle, a label TIFF
    for a segmentation bundle and a points parquet for a detection bundle.

    One keyword per config key (see the module docstring for the config): ``output_dir``,
    ``bundles``, ``ensemble``, ``stitch``, ``restrict_to_mask``, ``batch_size``,
    ``num_workers``, ``devices`` and ``split`` are the ``predict`` section, ``records``,
    ``splits``, ``subset`` and ``fold`` the ``cohort`` section, and ``extraction`` the
    ``extraction`` section (a mapping, or a ``slide2vec.PreprocessingConfig``). Relative
    paths resolve against the working directory.

    ``bundles`` are segmentation or detection bundles (:func:`soma.save`), all of one
    family, that agree on tile size and read spacing, or one run directory, which stands
    for its bundles. Each slide is cut into hs2p's slide grid with
    ``extraction.tiling.overlap``; a tile is kept when its row's ``mask_path``, or else
    hs2p's tissue segmentation, covers at least ``extraction.masks.min_coverage`` of it.
    Kept tiles are read at the bundles' read spacing and run through each bundle's model,
    behind one encoder forward for the bundles that share an encoder recipe.

    Segmentation: ``stitch`` names how overlapping tiles are combined
    (:mod:`soma.evaluation.whole_slide`). With ``pick_one`` (the default), each map pixel
    takes the class of one tile. With ``average``, it takes the argmax of the Hann-weighted
    mean softmax of every kept tile that covers it, so next to a tile that was not kept it
    labels the margin that pick one leaves at 0. Averaging also holds, per map, float32 sums
    over one tile's height of rows and the map's width, one plane per class. Class ``k`` is
    stored as ``k + 1``; 0 is "not predicted". ``average`` is refused for detection bundles.
    ``restrict_to_mask=True`` sets each map to 0 wherever the slide's mask is out of scope,
    after stitching and after an ensemble's averaging: a mask pixel is in scope when its
    value maps, through ``extraction.masks.pixel_mapping``, to a class that
    ``extraction.masks.min_coverage`` names. It does not change which tiles are kept. The
    mask is read on the map's grid with ``extraction.tiling.mask_backend``, and a map pixel
    it does not cover is out of scope. ``not_predicted_fraction`` is then the share of the
    in-scope pixels that are 0 (0.0 for a slide with none), not of all the map's pixels.
    It is refused for detection bundles and, before any slide is predicted, for a cohort
    with a slide without a ``mask_path``.
    Detection: each tile keeps the peaks in the region it contributes, lifted to level-0
    pixels, and one per-class NMS over the slide removes the duplicates across seams
    (:mod:`soma.evaluation.whole_slide_points`). The parquet holds every peak above the
    head's ``min_score``, with columns ``x``, ``y``, ``class`` and ``score``; the bundle's
    tune threshold is recorded, not applied, and so is its class scheme (``class_scheme``
    in its ``preprocessing.json``; a bundle that records none is refused). With an
    overlap, the margin (``tile * overlap / 2``) must be at least the head's NMS distance.

    The slides are read with the reader the bundles were trained with (the ``backend``
    their encoder recipe records), unless ``extraction.tiling.backend`` names one. Bundles
    that record different readers are refused unless it does, and a reader other than a
    bundle's is used with a warning. A bundle that records no reader leaves the choice to
    the others, and ``auto`` when none records one. A ``slide2vec.PreprocessingConfig``
    always names its reader. The masks are read with ``extraction.tiling.mask_backend``.

    Without ``splits`` every row of ``records`` is predicted; with it, the ``subset``
    split (default ``test``) of fold ``fold`` (default 0). ``fold="bundle"`` gives each
    bundle the ``subset`` split of its own fold, the ``fold`` its ``provenance.json``
    records. The bundle must also record that fold's ``fold_assignments``
    (:meth:`soma.data.FoldSplit.assignments_digest`), and ``splits`` must assign the fold
    the same way; ``ensemble`` and ``subset="train"`` are refused with it, and so is a
    bundle whose fold has no slides in ``subset``. ``ensemble`` writes one file per
    slide instead of one per bundle: the argmax of the bundles' mean softmax (to average,
    each tile's softmax is that mean), or the peaks of their mean heatmap, with no
    threshold recorded.
    ``devices`` (``[0, 1]``, ``["cpu"]``) predicts in one new process per device, the
    slides split across them; by default the slides are predicted in this process.
    ``split`` names what is split across the devices: ``"slides"`` (the default) or
    ``"tiles"``. With ``"tiles"`` the slides are predicted one after another: this process
    tiles each slide, gives chunk ``i`` of ``batch_size`` of its kept tiles, in order, to
    device ``i mod N``, takes the results back in order, and stitches and writes them on
    the CPU, holding no device. Each device process loads every bundle once and reads its
    tiles with ``num_workers`` processes; each logs to ``<output_dir>/logs/worker_<k>.log``.
    On CPU devices the files are byte for byte those of one device. A device process that
    raises or dies stops the call, the slide being written leaves no file, and the error
    names the device and its log. With one device, or without ``devices``, ``"tiles"`` is
    ``"slides"``.

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
        stitch=stitch,
        restrict_to_mask=restrict_to_mask,
        batch_size=batch_size,
        num_workers=num_workers,
        devices=devices,
        split=split,
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
    stitch="pick_one",
    restrict_to_mask=False,
    batch_size=8,
    num_workers=0,
    devices=None,
    split="slides",
    extraction=None,
) -> dict[str, Any]:
    """The plan of a prediction: the config with every value resolved (see the module)."""
    from soma.evaluation.whole_slide import MAX_CLASSES, STITCHES, stitch_margin_px

    if not isinstance(ensemble, bool):
        raise ValueError(f"predict.ensemble must be true or false, got {ensemble!r}")
    if not isinstance(restrict_to_mask, bool):
        raise ValueError(f"predict.restrict_to_mask must be true or false, got {restrict_to_mask!r}")
    if stitch not in STITCHES:
        raise ValueError(f"predict.stitch must be {' or '.join(map(repr, STITCHES))}, got {stitch!r}")
    if split not in SPLITS:
        raise ValueError(f"predict.split must be {' or '.join(map(repr, SPLITS))}, got {split!r}")
    if ensemble and fold == BUNDLE_FOLD:
        raise ValueError(
            "predict.ensemble averages the bundles over each slide, but cohort.fold: bundle "
            "gives each bundle the slides of its own fold, so the bundles do not predict the "
            "same slides. Predict a split they all held out with an explicit cohort.fold, or "
            "set predict.ensemble: false."
        )
    for name, value, minimum in (("batch_size", batch_size, 1), ("num_workers", num_workers, 0)):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"predict.{name} must be an integer >= {minimum}, got {value!r}")
    if devices is not None:
        from soma.pipeline.parallel import resolve_devices

        devices = list(resolve_devices(list(devices)))
    bundle_dirs = _bundle_dirs(bundles)
    infos = [_bundle_info(path) for path in bundle_dirs]
    family = _shared_family(infos)
    if stitch == "average" and family != "segmentation":
        raise ValueError(
            "predict.stitch: average applies to segmentation only: it averages the tiles' "
            f"softmax, and these are {family} bundles. Use predict.stitch: pick_one."
        )
    if restrict_to_mask and family != "segmentation":
        raise ValueError(
            "predict.restrict_to_mask applies to segmentation only: it sets the label map to 0 "
            f"outside the mask, and these are {family} bundles. Use predict.restrict_to_mask: false."
        )
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
            "stitch": stitch,
            "restrict_to_mask": restrict_to_mask,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "devices": devices,
            "split": split,
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
    info = {**_read_settings(preprocessing, path), "task_family": family}
    if info["tile_px"] is None:
        raise ValueError(f"{where} records the tile size {preprocessing.get('tile_size')}; soma predict needs square tiles")
    if info["spacing_um"] is None:
        raise ValueError(f"{where} records no read spacing (spacing_um)")
    info["num_classes"] = int(preprocessing["num_classes"])
    info["class_names"] = preprocessing.get("class_names")
    if family == "segmentation":
        info["ignore_index"] = int(preprocessing.get("ignore_index", 255))
        info["label_remap"] = preprocessing.get("label_remap")
    else:
        head = _bundle_head(path)
        info["class_names"] = [str(name) for name in head.classes]
        info["class_scheme"] = _detection_scheme(preprocessing, head.classes, where=where)
        info["match_distance_um"] = float(head.match_distance)
        info["nms_distance_um"] = float(head.nms_distance)
        info["min_score"] = float(head.min_score)
    return info


def _read_settings(preprocessing: Mapping[str, Any], path) -> dict[str, Any]:
    """How a bundle's slides were read in training, from its ``preprocessing.json`` (a
    loaded bundle's ``preprocessing``): the tile size (``tile_px``, ``None`` unless it
    records a square tile), the read spacing (the bundle's, else its encoder recipe's;
    ``None`` when neither records one), the read tolerance (0.05 when the recipe records
    none) and the slide reader (``None`` when the recipe records none)."""
    recipe = preprocessing.get("encoder")
    recipe = recipe if isinstance(recipe, Mapping) else {}
    tile = preprocessing.get("tile_size")
    tile = [tile, tile] if isinstance(tile, int) else list(tile or ())
    spacing = preprocessing.get("spacing_um")
    if spacing is None:
        spacing = recipe.get("spacing_um")
    backend = recipe.get("backend")
    return {
        "path": str(path),
        "tile_px": int(tile[0]) if len(tile) == 2 and tile[0] == tile[1] else None,
        "spacing_um": None if spacing is None else float(spacing),
        "tolerance": float(recipe.get("tolerance", 0.05)),
        "backend": None if backend is None else str(backend),
    }


def _bundle_head(path: Path, *, weights: bool = False):
    """A bundle's head, built from its ``model.json`` alone: building the whole model would
    load the encoder. A detection head's distances are there. With ``weights``, the head's
    own weights (none for the segmentation and detection heads) are loaded too, and it is
    put in eval mode: the head ``split: tiles`` stitches with, in a process without a
    device."""
    from soma.learn.bundle import MODEL, WEIGHTS, _decode
    from soma.models.heads.registry import task_registry

    recipe = json.loads((path / MODEL).read_text(encoding="utf-8"))
    spec = (recipe.get("params") or {}).get("head")
    if not isinstance(spec, Mapping) or "name" not in spec:
        raise ValueError(f"bundle {path} records no head in its {MODEL}, so soma predict cannot rebuild its head")
    spec = _decode(dict(spec))
    head = task_registry.resolve(spec["name"])(**spec.get("params", {}))
    if weights:
        from safetensors import safe_open

        # The grid model holds its head as ``head``: its weights are those under ``head.``.
        with safe_open(str(path / WEIGHTS), framework="pt", device="cpu") as stored:
            state = {key.removeprefix("head."): stored.get_tensor(key) for key in stored.keys() if key.startswith("head.")}
        head.load_state_dict(state)
        head.eval()
    return head


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
    "detection": ("num_classes", "class_names", "class_scheme", "match_distance_um", "nms_distance_um", "min_score"),
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
        "mean, so they must share the classes, the matching and NMS distances and min_score, "
        "and one class scheme (class_scheme): the annotated ids that form each class and the "
        f"ids dropped; these bundles have {_listing(infos, keys)}"
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
        # A whole margin, printed in full so the value proposed passes stitch_margin_px.
        needed = 2 * math.ceil(nms_um / spacing_um - 1e-9) / tile_px
        hint = (
            f"Use an overlap of at least {needed!r}, or 0."
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


def _slide_reader(configured: str | None, infos: Sequence[Mapping[str, Any]], *, setting: str) -> str:
    """The reader of the slides: ``configured``, the value of ``setting``
    (``extraction.tiling.backend`` for soma predict), or else the one the bundles record. A
    bundle that records none (one saved before soma recorded it) leaves the choice to the
    others, and ``auto`` when no bundle records one."""
    recording = [info for info in infos if info["backend"] is not None]
    if configured is None:
        if len({info["backend"] for info in recording}) > 1:
            raise ValueError(
                "the bundles of one prediction were trained on slides read with different "
                f"readers: {_readers(recording)}. Name the reader to predict with in "
                f"{setting}."
            )
        return recording[0]["backend"] if recording else "auto"
    others = [info for info in recording if info["backend"] != configured]
    if others:
        warnings.warn(
            f"{setting} is {configured!r}, but these bundles were trained on "
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
        "backend": _slide_reader(tiling.get("backend"), infos, setting="extraction.tiling.backend"),
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
    if fold is not None and fold != BUNDLE_FOLD and (isinstance(fold, bool) or not isinstance(fold, int) or fold < 0):
        raise ValueError(f"cohort.fold must be an integer >= 0 or {BUNDLE_FOLD!r}, got {fold!r}")
    subset = "test" if subset is None else str(subset)
    if fold == BUNDLE_FOLD and subset == "train":
        raise ValueError(
            "cohort.fold: bundle predicts only the slides each bundle held out, but "
            "cohort.subset: train is the split each bundle was trained on. To predict "
            "training slides, set an explicit cohort.fold."
        )
    return {
        "records": str(_absolute(records)),
        "splits": str(_absolute(splits)),
        "subset": subset,
        "fold": 0 if fold is None else fold,
    }


def _comparable(plan: Mapping[str, Any]) -> dict[str, Any]:
    from soma.evaluation.whole_slide import STITCHES

    payload = copy.deepcopy(dict(plan))
    predict = payload.get("predict") or {}
    for key in (*PLACEMENT_KEYS, *RECORDED_KEYS):
        predict.pop(key, None)
    # A plan written before these keys predicted with their defaults.
    predict.setdefault("stitch", STITCHES[0])
    predict.setdefault("restrict_to_mask", False)
    return payload


def _differences(saved: Any, current: Any, where: str = "") -> list[str]:
    if isinstance(saved, Mapping) and isinstance(current, Mapping):
        found: list[str] = []
        for key in sorted({*saved, *current}, key=str):
            found += _differences(saved.get(key), current.get(key), f"{where}.{key}" if where else str(key))
        return found
    return [] if saved == current else [where]


def _guard_plan(output_dir: Path, plan: Mapping[str, Any]) -> None:
    """Refuse a launch whose plan differs from the one ``output_dir`` was written with,
    but for the keys a relaunch may change; log a changed ``batch_size``."""
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
    before, now = (saved.get("predict") or {}).get("batch_size"), plan["predict"]["batch_size"]
    if before is not None and before != now:
        logger.info(
            "predict.batch_size changed from %s to %s: the slides already done in %s keep "
            "theirs, recorded in their <sample_id>.json, and the others are predicted with %s",
            before,
            now,
            output_dir,
            now,
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


def _work(plan: Mapping[str, Any]) -> tuple[list[tuple[Any, tuple[int, ...]]], list[dict[str, Any]] | None]:
    """The slides to predict, in ``records`` order, each with the indices of the bundles
    that predict it; and, with ``cohort.fold: bundle``, each bundle's checked ``fold`` and
    ``fold_assignments`` (``None`` otherwise)."""
    from soma.data import Cohort, ImageManifest

    cohort, bundles = plan["cohort"], plan["predict"]["bundles"]
    manifest = ImageManifest.from_csv(cohort["records"])
    every = tuple(range(len(bundles)))
    if cohort["splits"] is None:
        return [(manifest[sid], every) for sid in manifest.sample_ids], None
    folds = Cohort.from_csv(cohort["records"], cohort["splits"], targets=[], allow_missing_test=True).folds
    if cohort["fold"] != BUNDLE_FOLD:
        if cohort["fold"] >= len(folds):
            raise ValueError(f"cohort.fold is {cohort['fold']}, but {cohort['splits']} has {len(folds)} fold(s)")
        wanted = set(_subset(folds[cohort["fold"]], cohort["subset"], where=f"fold {cohort['fold']}"))
        return [(manifest[sid], every) for sid in manifest.sample_ids if sid in wanted], None
    recorded = [_bundle_fold(Path(path), folds, cohort["splits"]) for path in bundles]
    carriers: dict[str, list[int]] = {}
    for index, (path, entry) in enumerate(zip(bundles, recorded)):
        where = f"fold {entry['fold']} of bundle {path}"
        wanted = _subset(folds[entry["fold"]], cohort["subset"], where=where)
        if not wanted:
            # A fold with no tune rows still has a tune split, an empty one.
            raise ValueError(
                f"cohort.subset {cohort['subset']!r} of {where} has no slides in it, so the "
                "bundle would predict nothing. Predict another subset, or leave the bundle "
                "out of predict.bundles."
            )
        for sid in wanted:
            carriers.setdefault(sid, []).append(index)
    return [(manifest[sid], tuple(carriers[sid])) for sid in manifest.sample_ids if sid in carriers], recorded


def _subset(fold, subset: str, *, where: str) -> tuple[str, ...]:
    """The sample ids of split ``subset`` of ``fold`` (``where`` names it in the error)."""
    subsets = {"train": fold.train, "tune": fold.tune, **fold.tests}
    if subset not in subsets:
        raise ValueError(f"cohort.subset {subset!r} is not a split of {where}; its splits are {sorted(subsets)}")
    return tuple(subsets[subset])


def _bundle_fold(path: Path, folds, splits: str) -> dict[str, Any]:
    """The fold a bundle records, once its ``fold_assignments`` match that fold of
    ``splits``: what it held out is what it predicts."""
    from soma.learn.bundle import PROVENANCE

    provenance = json.loads((path / PROVENANCE).read_text(encoding="utf-8"))
    fold, digest = provenance.get("fold"), provenance.get("fold_assignments")
    if isinstance(fold, bool) or not isinstance(fold, int) or not isinstance(digest, str):
        raise ValueError(
            f"bundle {path} records no fold_assignments in its {PROVENANCE}: it was saved "
            "before soma recorded them, or by soma.save with no fold, so cohort.fold: bundle "
            "cannot check that it predicts the slides it held out. Predict it with one config "
            "per fold, each with an explicit cohort.fold."
        )
    current = folds[fold].assignments_digest() if 0 <= fold < len(folds) else None
    if current != digest:
        found = f"{splits} has {len(folds)} fold(s)" if current is None else (
            f"fold {fold} of {splits} has fold_assignments {current[:12]}"
        )
        raise ValueError(
            f"bundle {path} records fold {fold} with fold_assignments {digest[:12]}, but "
            f"{found}: these splits differ from the ones the bundle was trained on. Predict "
            "with the splits it was trained on."
        )
    return {"fold": fold, "fold_assignments": digest}


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


def _guard_bundles(output_dir: Path, identities: Sequence[Mapping[str, Any]]) -> None:
    """Refuse a launch whose bundles changed since ``output_dir``'s maps were written: the
    plan names a bundle by its path, ``provenance.json`` by its digest and, with
    ``cohort.fold: bundle``, by the fold and ``fold_assignments`` it records."""
    try:
        saved = json.loads((output_dir / PROVENANCE_FILENAME).read_text(encoding="utf-8"))["bundles"]
    except (OSError, ValueError, KeyError, TypeError):
        return
    entries = {entry.get("path"): entry for entry in saved if isinstance(entry, Mapping)}
    for identity in identities:
        before = entries.get(identity["path"])
        if before is None:
            continue
        if before.get("sha256") is not None and before["sha256"] != identity["sha256"]:
            raise ValueError(
                f"Refusing to predict into {output_dir}: bundle {identity['path']} changed since "
                f"its maps were predicted into it (sha256 {before['sha256'][:12]} then, "
                f"{identity['sha256'][:12]} now). Use another predict.output_dir."
            )
        fold_keys = ("fold", "fold_assignments")
        if any(key in before for key in fold_keys) and any(before.get(k) != identity.get(k) for k in fold_keys):
            raise ValueError(
                f"Refusing to predict into {output_dir}: bundle {identity['path']} records "
                "another fold than when its predictions were written into it (fold "
                f"{before.get('fold')}, fold_assignments {str(before.get('fold_assignments'))[:12]} "
                f"then; fold {identity.get('fold')}, fold_assignments "
                f"{str(identity.get('fold_assignments'))[:12]} now). Use another predict.output_dir."
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
    work, folds = _work(plan)
    if plan["predict"]["restrict_to_mask"]:
        bare = [entry.sample_id for entry, _ in work if entry.mask_path is None]
        if bare:
            raise ValueError(
                f"predict.restrict_to_mask needs a mask_path for every slide, and {len(bare)} "
                f"slide(s) have none: {bare}. Add their mask_path to {plan['cohort']['records']}, "
                "or set predict.restrict_to_mask: false."
            )
    _guard_plan(output_dir, plan)
    identities = _identities(plan)
    if folds is not None:
        identities = [{**identity, **fold} for identity, fold in zip(identities, folds)]
    _guard_bundles(output_dir, identities)
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_text(output_dir / PLAN_FILENAME, yaml.safe_dump(plan, sort_keys=False))
    _write_provenance(output_dir, plan, identities)
    names = _output_names(plan)
    pending = [
        entry
        for entry, carriers in work
        if any(
            _read_record(_record_path(output_dir, name, entry.sample_id)) is None
            for name in _names_of(plan, names, carriers)
        )
    ]
    _check_spacings(pending, plan["extraction"]["tiling"])
    ids = [entry.sample_id for entry in pending]
    logger.info("Predicting %d of %d slide(s) into %s", len(ids), len(work), output_dir)
    devices = plan["predict"]["devices"]
    if ids and devices is None:
        predict_shard(plan, ids)
    elif ids and plan["predict"]["split"] == "tiles" and len(devices) > 1:
        _predict_tiles_on_devices(plan, ids, devices)
    elif ids:
        _predict_on_devices(plan, ids, devices)
    index = _index(output_dir, plan, names, work, _RECORD_FIELDS[_plan_family(plan)])
    atomic_write_bytes(output_dir / INDEX_FILENAME, _parquet_bytes(index))
    return index


def _names_of(plan: Mapping[str, Any], names: Sequence[str], carriers: Sequence[int]) -> list[str]:
    """The output directories of a slide the bundles ``carriers`` predict."""
    return list(names) if plan["predict"]["ensemble"] else [names[index] for index in carriers]


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


def _index(output_dir: Path, plan, names: Sequence[str], work, fields: Sequence[str]) -> pd.DataFrame:
    """One row per output directory and slide it holds: each bundle's slides in
    ``records`` order, bundle after bundle."""
    rows = []
    for name in names:
        for entry, carriers in work:
            if name not in _names_of(plan, names, carriers):
                continue
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


def _predict_tiles_on_devices(plan: dict[str, Any], ids: list[str], devices: Sequence[str]) -> None:
    """``split: tiles``: predict the slides one after another in this process, each slide's
    tiles dealt across one process per device (:mod:`soma.pipeline.tile_split`)."""
    import torch

    from soma.pipeline.parallel import free_device_memory
    from soma.pipeline.tile_split import device_tile_outputs

    output_dir = Path(plan["predict"]["output_dir"])
    bundles = plan["predict"]["bundles"]
    # This process holds no device: it stitches with heads built from the bundles' files.
    heads = [_bundle_head(Path(path), weights=True) for path in bundles]
    free_device_memory()
    with device_tile_outputs(
        bundles, devices, heads=heads, num_workers=plan["predict"]["num_workers"], log_dir=output_dir / "logs"
    ) as outputs:
        # The heads run here, under inference mode as on a device (DeviceTileOutputs.batches).
        with torch.inference_mode():
            _predict_slides(plan, ids, outputs)


def predict_shard(plan: Mapping[str, Any], sample_ids: Sequence[str]) -> list[str]:
    """Predict the slides ``sample_ids`` of ``plan`` in this process (a device job's
    target); returns the ids predicted."""
    from soma.evaluation.whole_slide import DeviceTileOutputs

    outputs = DeviceTileOutputs.load(plan["predict"]["bundles"], num_workers=plan["predict"]["num_workers"])
    _predict_slides(plan, sample_ids, outputs)
    return list(sample_ids)


def _predict_slides(plan: Mapping[str, Any], sample_ids: Sequence[str], outputs) -> None:
    """Predict the slides ``sample_ids`` in turn with every bundle of ``outputs`` (a
    :class:`~soma.evaluation.whole_slide.TileOutputs`)."""
    output_dir = Path(plan["predict"]["output_dir"])
    names = _output_names(plan)
    identities = _identities(plan)
    by_id = {entry.sample_id: (entry, carriers) for entry, carriers in _work(plan)[0]}
    for sample_id in sample_ids:
        # Only the slide's own bundles run; those sharing an encoder recipe share its forward.
        entry, carriers = by_id[sample_id]
        _predict_slide(
            plan,
            outputs.subset(carriers),
            _names_of(plan, names, carriers),
            [identities[index] for index in carriers],
            entry,
            output_dir,
        )


def _predict_slide(plan, outputs, names, identities, entry, output_dir: Path) -> None:
    """Predict one slide with the bundles of ``outputs`` (a
    :class:`~soma.evaluation.whole_slide.TileOutputs`), the one source of every tile's
    results."""
    import slide2vec

    from soma.evaluation.whole_slide import SlideTiles

    extraction = hydrate(slide2vec.PreprocessingConfig, plan["extraction"], "extraction")
    origins, grid = _kept_tiles(entry, extraction)
    tiling = extraction.tiling
    tiles = SlideTiles(
        image=entry,
        origins=origins,
        spacing_um=float(tiling.requested_spacing_um),
        backend=tiling.backend,
        tolerance=float(tiling.tolerance),
    )
    for name in names:
        (output_dir / name).mkdir(parents=True, exist_ok=True)
    detection = outputs.heads[0].task_family == "detection"
    if not origins:
        empty = "its points file has no rows" if detection else "its label map is all zeros, not predicted"
        warnings.warn(
            f"slide '{entry.sample_id}' has no kept tile (no tissue under the mask or the "
            f"tissue segmentation): {empty}",
            stacklevel=2,
        )
    write = _write_slide_points if detection else _write_label_maps
    write(plan, outputs, names, identities, entry, output_dir, extraction=extraction, tiles=tiles, grid=grid)
    logger.info("Predicted slide %s: %d tile(s)", entry.sample_id, len(origins))


def _write_label_maps(plan, outputs, names, identities, entry, output_dir: Path, *, extraction, tiles, grid) -> None:
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
    restricted = plan["predict"]["restrict_to_mask"]
    shared = {
        "sample_id": entry.sample_id,
        "ensemble": ensemble,
        "read_spacing_um": float(tiling.requested_spacing_um),
        "tile_size_px": int(tiling.requested_tile_size_px),
        "overlap": float(tiling.overlap),
        "stitch": plan["predict"]["stitch"],
        "restrict_to_mask": restricted,
    }
    heads = outputs.heads
    if ensemble:
        metadata = [{**shared, **_class_scheme(heads[0]), "bundles": identities}]
    else:
        metadata = [
            {**shared, **_class_scheme(head), "bundles": [identity]} for head, identity in zip(heads, identities)
        ]
    file = f"{entry.sample_id}.tif"
    with _mask_scope(entry, extraction, grid, frame) if restricted else nullcontext() as scope:
        fractions = predict_slide_label_maps(
            outputs,
            tiles=tiles,
            frame=frame,
            paths=[output_dir / name / file for name in names],
            metadata=metadata,
            ensemble=ensemble,
            stitch=plan["predict"]["stitch"],
            batch_size=plan["predict"]["batch_size"],
            scope=scope,
        )
    for name, fraction in zip(names, fractions):
        record = {
            "sample_id": entry.sample_id,
            "file": file,
            "width": frame.width,
            "height": frame.height,
            "spacing_um": frame.spacing_um,
            "read_spacing_um": float(tiling.requested_spacing_um),
            "num_tiles": len(tiles.origins),
            "not_predicted_fraction": fraction,
            "restrict_to_mask": restricted,
            "batch_size": plan["predict"]["batch_size"],
        }
        atomic_write_text(_record_path(output_dir, name, entry.sample_id), json.dumps(record, indent=2) + "\n")


@contextmanager
def _mask_scope(entry, extraction, grid, frame):
    """The slide's ``mask_path`` as a :class:`~soma.evaluation.whole_slide.MaskScope` on
    the map's grid: read with ``extraction.tiling.mask_backend`` against
    ``extraction.masks.pixel_mapping``, aligned to the slide's level 0 as hs2p tiled it,
    with the values of the classes that select tiles (``extraction.masks.min_coverage``)
    in scope."""
    from hs2p import AnnotationLabels, Mask

    from soma.evaluation.whole_slide import MaskScope

    masks = extraction.masks
    values = {int(value) for name in masks.min_coverage for value in masks.pixel_mapping[name]}
    labels = AnnotationLabels(pixel_mapping={name: list(ids) for name, ids in masks.pixel_mapping.items()})
    with Mask(path=Path(entry.mask_path), labels=labels, backend=extraction.tiling.mask_backend) as mask:
        aligned = mask.align_to(
            reference_spacing_um=float(grid.tiles.base_spacing_um),
            reference_dimensions=tuple(int(v) for v in grid.tiles.slide_dimensions),
        )
        yield MaskScope(aligned, values, frame)


def _write_slide_points(plan, outputs, names, identities, entry, output_dir: Path, *, extraction, tiles, grid) -> None:
    from soma.evaluation.whole_slide_points import predict_slide_points

    tiling = extraction.tiling
    ensemble = plan["predict"]["ensemble"]
    geometry = grid.tiles
    level0_spacing = float(geometry.base_spacing_um)
    width, height = (int(v) for v in geometry.slide_dimensions)
    # hs2p rounds the step in the pixels it reads, so in tile pixels it is not always
    # tile - 2 * margin (on a slide resampled to the read spacing).
    step_px = (
        grid.step_px_lv0
        * level0_spacing
        / float(geometry.read_spacing_um)
        * int(tiling.requested_tile_size_px)
        / int(geometry.read_tile_size_px)
    )
    step_px = round(step_px, 6)  # float noise must not move a whole-pixel boundary
    shared = {
        "sample_id": entry.sample_id,
        "ensemble": ensemble,
        "read_spacing_um": float(tiling.requested_spacing_um),
        "tile_size_px": int(tiling.requested_tile_size_px),
        "overlap": float(tiling.overlap),
    }
    heads = outputs.heads
    if ensemble:
        # An ensemble's scores are a mean no bundle was tuned on: it has no threshold. Its
        # bundles share one class scheme (the plan checks it).
        recorded = _recorded_points(identities[0]["path"], heads[0])
        metadata = [{**shared, **recorded, "bundles": identities, "threshold": None}]
    else:
        metadata = [
            {**shared, **_recorded_points(identity["path"], head), "bundles": [identity]}
            for head, identity in zip(heads, identities)
        ]
    file = f"{entry.sample_id}.parquet"
    counts = predict_slide_points(
        outputs,
        tiles=tiles,
        tile_px=int(tiling.requested_tile_size_px),
        step_px=step_px,
        overlap=float(tiling.overlap),
        slide_size=(width, height),
        level0_spacing_um=level0_spacing,
        paths=[output_dir / name / file for name in names],
        metadata=metadata,
        ensemble=ensemble,
        batch_size=plan["predict"]["batch_size"],
    )
    for name, count in zip(names, counts):
        record = {
            "sample_id": entry.sample_id,
            "file": file,
            "width": width,
            "height": height,
            "spacing_um": level0_spacing,
            "read_spacing_um": float(tiling.requested_spacing_um),
            "num_tiles": len(tiles.origins),
            "num_points": count,
            "batch_size": plan["predict"]["batch_size"],
        }
        atomic_write_text(_record_path(output_dir, name, entry.sample_id), json.dumps(record, indent=2) + "\n")


def _recorded_points(bundle_path: str, head) -> dict[str, Any]:
    """What a detection bundle's points files copy from its ``preprocessing.json``: the
    class scheme, in the head's class order, and the operating threshold (a float, or one
    per class name; ``None`` when the bundle records none)."""
    from soma.learn.bundle import PREPROCESSING

    preprocessing = json.loads((Path(bundle_path) / PREPROCESSING).read_text(encoding="utf-8"))
    scheme = _detection_scheme(preprocessing, head.classes, where=f"bundle {bundle_path}")
    return {**scheme, "threshold": preprocessing.get("score_threshold")}


def _detection_scheme(preprocessing: Mapping[str, Any], class_names, *, where: str) -> dict[str, Any]:
    """The class scheme a detection bundle records: ``classes`` (class name → annotated
    ids) in the order of ``class_names``, and ``drop`` (the annotated ids dropped). The
    bundle's JSON is written with sorted keys, so the order of its mapping is not the class
    order. A scheme added by hand to an older bundle is checked as a run's config is."""
    from soma._internal.class_scheme import assign_raw_values, recorded_scheme

    scheme = preprocessing.get("class_scheme")
    if scheme is None:
        raise ValueError(
            f"{where} records no class scheme (class_scheme in its preprocessing.json): it "
            "was trained before soma recorded one, so its points could not be scored. Train "
            "it again, or add class_scheme to its preprocessing.json from the run's config: "
            "classes (each class name and the annotated ids that form it) and drop (the "
            "annotated ids dropped, [] for none)."
        )
    names = [str(name) for name in class_names]
    invalid = f"{where} records an invalid class_scheme {scheme!r}"
    classes = scheme.get("classes") if isinstance(scheme, Mapping) else None
    if not isinstance(classes, Mapping) or sorted(str(name) for name in classes) != sorted(names):
        raise ValueError(
            f"{invalid}: its classes must have one entry per class of the bundle ({names}), "
            "each listing the annotated ids that form the class"
        )
    if "drop" not in scheme:
        raise ValueError(f"{invalid}: it needs a drop list (the annotated ids dropped, [] for none)")
    try:
        class_of, dropped = assign_raw_values(
            {name: classes[name] for name in names}, excluded=scheme["drop"], excluded_name="drop"
        )
    except ValueError as error:
        raise ValueError(f"{invalid}: {error}") from None
    return recorded_scheme(names, class_of, dropped, excluded_key="drop")


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
