"""Recover MIDOG 2022 per-image domain metadata and emit an enriched COCO JSON.

WHY THIS EXISTS
---------------
The public MIDOG 2022 training bundle (Zenodo record ``6547151``, PNG version) ships a
COCO JSON (``MIDOG2022_training_png.json``) and a SlideRunner ``.sqlite`` that are **both
stripped of per-image domain metadata** -- only ``file_name`` / ``width`` / ``height`` /
``id`` survive. MIDOG's entire role in this benchmark is the *domain-shift axis*
(scanner / tumor type), so we must recover tumor type + scanner per image from an external
source before :mod:`soma.curation.midog` (which reads those fields from the JSON) can emit
a domain-aware manifest.

RECOVERY SOURCE
---------------
The **MIDOG++** dataset (``DeepMicroscopy/MIDOGpp``) is the labeled superset that
re-publishes the 2022 ROIs *with* a per-image ``tumor_type`` field, plus a ``Scanner``
column in ``datasets_xvalidation.csv``. MIDOG++ **renumbered** the cases (its ids ``1..553``
diverge from the 2022 ids ``1..405`` past the breast block), so we cannot map by id.
Instead we **fingerprint-match** each 2022 image to its MIDOG++ twin on
``(width, height, exact annotation-centre set)`` -- a coordinate-exact match for all 354
labeled images. From the matched MIDOG++ image we read ``tumor_type`` (its per-image field)
and ``Scanner`` (via the MIDOG++ slide id -> ``datasets_xvalidation.csv``).

Pinned sources (fetch with ``curl`` if not passed on the CLI)::

    MIDOG++.json              @ bd26c8197fc785b3e414ff2d41a63c1809dae1bc  (2023-06-13)
      raw.githubusercontent.com/DeepMicroscopy/MIDOGpp/bd26c81.../databases/MIDOG%2B%2B.json
    datasets_xvalidation.csv  @ c2726b1f40a1...                           (2023-12-01)
      raw.githubusercontent.com/DeepMicroscopy/MIDOGpp/c2726b1.../datasets_xvalidation.csv

RECOVERED STRUCTURE (405 = 354 labeled + 51 unlabeled)
------------------------------------------------------
======= ================================ ================================== ========
id blk  tumor type                       scanner(s)                         mitoses
======= ================================ ================================== ========
001-150 human breast cancer              Hamamatsu XR / S360 / Aperio CS2      1721
151-194 canine lung cancer               3D Histech                             855
195-249 canine lymphosarcoma             3D Histech                            3959
250-299 canine cutaneous mast cell tumor Aperio CS2                            2327
300-354 human neuroendocrine tumor       Hamamatsu XR                           639
355-405 human melanoma  (UNLABELED)      Hamamatsu XR                             0
======= ================================ ================================== ========

WHY HUMAN MELANOMA (355-405) IS DROPPED BY DEFAULT
--------------------------------------------------
MIDOG = the MItosis DOmain Generalization challenge. The organizers ship human melanoma as
a 6th training domain **without mitosis annotations on purpose**: it is an *unlabeled target
domain* for **domain-generalization / unsupervised-domain-adaptation** methods. The
challenge reference algorithm is a **domain-adversarial RetinaNet** (a domain-classifier
head + a gradient-reversal layer) that learns domain-**invariant** features; an extra
unlabeled domain enriches that invariance signal *without needing mitosis labels*. The
hidden MIDOG 2022 test set then scores on ten independent domains, several unseen in
training -- the point of the challenge is generalization, not in-domain accuracy.

Our benchmark is a **different protocol**: a *frozen* FM encoder + a lightweight decoder
trained by supervised heatmap regression **from point labels**. It has no domain head and no
adaptation loss, so it can extract nothing from unlabeled melanoma; those 51 images would
only ever act as pure-negative false-positive bait in the aggregate F1. We therefore
**exclude** them, yielding a 354-image, 5-tumor-type x 4-scanner detection benchmark. Pass
``--keep-melanoma`` to retain them as a qualitative "does the probe fire on an unseen
domain" specificity probe (they curate to empty point CSVs).
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

# Expected labeled composition (from the MIDOG 2022 data descriptor); asserted after
# reconciliation so a silently-wrong mapping fails loudly.
EXPECTED_TUMOR_COUNTS = {
    "human breast cancer": 150,
    "canine lung cancer": 44,
    "canine lymphosarcoma": 55,
    "canine cutaneous mast cell tumor": 50,
    "human neuroendocrine tumor": 55,
}
EXPECTED_TOTAL_MITOSES = 9501
MELANOMA_TUMOR = "human melanoma"
MELANOMA_SCANNER = "Hamamatsu XR"

# The MIDOG++ xval CSV spells the Hamamatsu NanoZoomer XR two ways ("Hamammatsu XR" for the
# breast block, "Hamamatsu XR" for neuroendocrine); they are the same physical scanner.
_SCANNER_NORMALIZE = {"Hamammatsu XR": "Hamamatsu XR"}


def _centres(coco: dict, *, mitotic_only: bool):
    """image_id -> sorted tuple of integer annotation centres (all, or mitotic cat==1)."""
    per = collections.defaultdict(list)
    for ann in coco["annotations"]:
        if mitotic_only and int(ann["category_id"]) != 1:
            continue
        x, y, w, h = ann["bbox"][:4]
        per[ann["image_id"]].append((round(x + w / 2), round(y + h / 2)))
    return {k: tuple(sorted(v)) for k, v in per.items()}


def _index_by_fp(coco: dict, centres: dict):
    """(w, h, centres) fingerprint -> list of MIDOG++ image ids (for collision detection)."""
    idx = collections.defaultdict(list)
    for im in coco["images"]:
        fp = (im["width"], im["height"], centres.get(im["id"], ()))
        idx[fp].append(im["id"])
    return idx


def reconcile(midog22: dict, midogpp: dict, scanner_by_ppid: dict):
    """Return {midog22_image_id: (tumor_type, scanner)} for the labeled images.

    Fingerprint-matches on the full annotation-centre set first (coordinate-exact), then
    falls back to the mitotic-only centre set for the handful of images whose hard-negative
    annotations were refined between the 2022 export and MIDOG++. Unmatched images (the
    unlabeled melanoma, which carry zero annotations) are simply absent from the result.
    """
    pp_tumor = {im["id"]: im.get("tumor_type") for im in midogpp["images"]}
    pp_all = _index_by_fp(midogpp, _centres(midogpp, mitotic_only=False))
    pp_mit = _index_by_fp(midogpp, _centres(midogpp, mitotic_only=True))
    c22_all = _centres(midog22, mitotic_only=False)
    c22_mit = _centres(midog22, mitotic_only=True)

    out: dict[int, tuple[str, str]] = {}
    for im in midog22["images"]:
        iid = im["id"]
        fp_all = (im["width"], im["height"], c22_all.get(iid, ()))
        cands = pp_all.get(fp_all, [])
        if len(cands) != 1:
            fp_mit = (im["width"], im["height"], c22_mit.get(iid, ()))
            cands = pp_mit.get(fp_mit, []) if c22_mit.get(iid) else []
        if len(cands) == 1:
            ppid = cands[0]
            scanner = _SCANNER_NORMALIZE.get(scanner_by_ppid.get(ppid), scanner_by_ppid.get(ppid))
            out[iid] = (pp_tumor[ppid], scanner)
    return out


def load_scanner_csv(path: Path) -> dict[int, str]:
    """MIDOG++ slide id -> Scanner, from datasets_xvalidation.csv (';'-delimited)."""
    scanner_by_ppid: dict[int, str] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            scanner_by_ppid[int(row["Slide"])] = row["Scanner"].strip()
    return scanner_by_ppid


def build_enriched(midog22: dict, tumor_scanner: dict, *, keep_melanoma: bool, spacing):
    """Return an enriched COCO dict: labeled images gain tumortype/scanner/patient_id
    (+ optional spacing); melanoma is dropped unless keep_melanoma."""
    mit = _centres(midog22, mitotic_only=True)
    kept_images, kept_ids = [], set()
    for im in sorted(midog22["images"], key=lambda i: i["file_name"]):
        iid = im["id"]
        stem = Path(im["file_name"]).stem
        if iid in tumor_scanner:
            tumor, scanner = tumor_scanner[iid]
        else:
            # Unmatched == unlabeled melanoma (asserted by caller).
            if not keep_melanoma:
                continue
            tumor, scanner = MELANOMA_TUMOR, MELANOMA_SCANNER
        enriched = {
            **im,
            "tumortype": tumor,
            "scanner": scanner,
            "patient_id": stem,  # MIDOG 2022 is one ROI per case, so patient == image
        }
        if spacing is not None:
            enriched["spacing"] = float(spacing)
        kept_images.append(enriched)
        kept_ids.add(iid)

    kept_anns = [a for a in midog22["annotations"] if a["image_id"] in kept_ids]
    out = {k: midog22[k] for k in midog22 if k not in ("images", "annotations")}
    out["images"] = kept_images
    out["annotations"] = kept_anns
    return out, mit


def _report_and_assert(enriched: dict, mit_centres: dict, *, keep_melanoma: bool):
    imgs = enriched["images"]
    by_tumor = collections.Counter(i["tumortype"] for i in imgs)
    by_scanner = collections.Counter(i["scanner"] for i in imgs)
    labeled = {t: n for t, n in by_tumor.items() if t != MELANOMA_TUMOR}
    total_mit = sum(len(mit_centres.get(i["id"], ())) for i in imgs)

    print(f"enriched images: {len(imgs)}  (annotations: {len(enriched['annotations'])})")
    print("tumor type:")
    for t, n in sorted(by_tumor.items()):
        print(f"    {t:34s} {n:4d}")
    print("scanner:")
    for s, n in sorted(by_scanner.items()):
        print(f"    {str(s):20s} {n:4d}")
    print(f"total mitotic figures: {total_mit}")

    assert labeled == EXPECTED_TUMOR_COUNTS, f"labeled tumor counts drifted: {labeled}"
    assert total_mit == EXPECTED_TOTAL_MITOSES, f"mitoses={total_mit} != {EXPECTED_TOTAL_MITOSES}"
    if not keep_melanoma:
        assert MELANOMA_TUMOR not in by_tumor, "melanoma leaked into the labeled manifest"
        assert len(imgs) == sum(EXPECTED_TUMOR_COUNTS.values()) == 354
    print("OK: reconciliation asserts passed.")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prepare_midog_metadata", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-json", type=Path, required=True,
                    help="MIDOG2022_training_png.json from Zenodo 6547151")
    ap.add_argument("--midogpp-json", type=Path, required=True,
                    help="MIDOG++.json (DeepMicroscopy/MIDOGpp databases/, pinned bd26c81)")
    ap.add_argument("--midogpp-xval-csv", type=Path, required=True,
                    help="datasets_xvalidation.csv (DeepMicroscopy/MIDOGpp, pinned c2726b1)")
    ap.add_argument("--out", type=Path, required=True, help="enriched COCO JSON to write")
    ap.add_argument("--keep-melanoma", action="store_true",
                    help="retain the 51 unlabeled melanoma images (default: drop)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="optional per-image spacing (µm/px) to stamp; default leaves it to "
                         "the curator's --level0-spacing-um")
    args = ap.parse_args(argv)

    midog22 = json.loads(args.raw_json.read_text())
    midogpp = json.loads(args.midogpp_json.read_text())
    scanner_by_ppid = load_scanner_csv(args.midogpp_xval_csv)

    tumor_scanner = reconcile(midog22, midogpp, scanner_by_ppid)
    n_labeled, n_total = len(tumor_scanner), len(midog22["images"])
    print(f"fingerprint-matched {n_labeled}/{n_total} images to MIDOG++ "
          f"({n_total - n_labeled} unmatched -> unlabeled melanoma)")

    enriched, mit = build_enriched(midog22, tumor_scanner,
                                   keep_melanoma=args.keep_melanoma, spacing=args.spacing)
    _report_and_assert(enriched, mit, keep_melanoma=args.keep_melanoma)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(enriched))
    print(f"wrote enriched COCO -> {args.out}")


if __name__ == "__main__":
    main()
