"""``soma.evaluate`` on the points parquets ``soma.predict_slides`` writes, against the
slides' annotated points.

The slides are synthetic TIFFs at 0.5 um/px, read through hs2p's ``tifffile`` reader. The
bundles carry the weight-free ``fake`` encoder's recipe and the marker decoder, which puts
each peak at a known level-0 pixel with a known score, so every count is hand-checked. The
class scheme is a dataset's own: annotated id 1 is ``mitosis``, 2 is ``mimic``, and 3 is
dropped. The matching distance is 3 um, 6 level-0 pixels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

import soma
from soma.data import AnnotationManifest, PointSource
from soma.evaluation import SlidePoints
from tests.whole_slides import (
    TIFFFILE,
    MarkerDecoder,
    MarkerParams,
    marker_slide,
    save_detection_bundle,
    score,
    write_mask,
    write_records,
    write_slide,
)

CLASSES = {"mitosis": [1], "mimic": [2]}
DROP = [3]
#: The predicted peaks ``(x, y, blue, class)``: a peak scores ``sigmoid(blue / 20)``. The
#: bundles' threshold of 0.9 keeps every peak but the second (0.73).
MARKERS = [
    (10, 12, 120, 0),  # mitosis, by a mitosis: a true positive
    (45, 20, 20, 0),  # mitosis, by a mitosis, below the threshold: that mitosis is missed
    (70, 50, 150, 1),  # mimic, by a mimic: a true positive
    (100, 75, 180, 0),  # mitosis, by a dropped point only: a false positive
    (20, 80, 200, 1),  # mimic, by nothing: a false positive
    (110, 20, 160, 1),  # mimic, by a mimic: a true positive
]
#: The annotated points ``(x, y, id)``.
ANNOTATED = [
    (12, 14, 1),
    (46, 22, 1),
    (72, 49, 2),
    (101, 76, 3),  # dropped: neither a target nor an ignored region
    (60, 10, 2),  # no peak in reach: missed
    (112, 18, 2),
]


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)


@pytest.fixture
def marker(monkeypatch):
    """Register the marker decoder for one test, then restore the registry."""
    from soma.models import decoder_registry

    monkeypatch.setattr(decoder_registry, "_entries", dict(decoder_registry._entries))
    soma.register("decoders", "marker", MarkerDecoder, params=MarkerParams)
    return {"name": "marker", "params": {}}


def write_points(path, points) -> str:
    pd.DataFrame(points, columns=["x", "y", "class"]).to_csv(path, index=False)
    return str(path)


def predict(tmp_path, marker, *, bundles=("run",), ensemble=False, markers=MARKERS):
    """Predict the 128 x 96 px slide ``s1`` with one bundle per name; return the output."""
    slide = write_slide(tmp_path / "s1.tif", marker_slide(96, 128, markers))
    mask = write_mask(tmp_path / "s1_tissue.tif", np.ones((96, 128)))
    records = write_records(
        tmp_path / "slides.csv", [{"sample_id": "s1", "image_path": str(slide), "mask_path": str(mask)}]
    )
    paths = [
        save_detection_bundle(
            tmp_path / name / "bundle", decoder=marker, classes=CLASSES, drop=DROP, score_threshold=0.9, seed=k
        )
        for k, name in enumerate(bundles)
    ]
    output = tmp_path / "predictions"
    soma.predict_slides(
        output_dir=output,
        bundles=paths if len(paths) > 1 else paths[0],
        records=records,
        extraction={"tiling": {"overlap": 0.0, **TIFFFILE}},
        ensemble=ensemble,
    )
    return output


def targets(tmp_path, *, points=ANNOTATED, ignore_mask=None) -> PointSource:
    row = {"sample_id": "s1", "points_path": write_points(tmp_path / "s1_points.csv", points)}
    if ignore_mask is not None:
        row["ignore_mask_path"] = str(ignore_mask)
    return PointSource.from_manifest(AnnotationManifest.from_frame(pd.DataFrame([row])))


def test_a_bundle_s_points_are_scored_at_its_threshold_with_its_class_scheme(tmp_path, marker):
    """mitosis: 1 true positive, 1 false positive (the peak by a dropped point), 1 missed
    (its peak is below the threshold), F1 = 2 / 4. mimic: 2 true positives, 1 false
    positive, 1 missed, F1 = 4 / 6."""
    points = SlidePoints(predict(tmp_path, marker) / "run")
    assert points.classes == ("mitosis", "mimic")
    assert points.threshold == 0.9

    evaluation = soma.evaluate(points, targets(tmp_path), metrics=["mean_f1", "f1_per_class"])

    assert evaluation.num_samples == 1
    assert evaluation.metrics["f1_class_0"] == pytest.approx(2 / 4)
    assert evaluation.metrics["f1_class_1"] == pytest.approx(4 / 6)
    assert evaluation.metrics["mean_f1"] == pytest.approx((2 / 4 + 4 / 6) / 2)
    (row,) = evaluation.per_sample
    assert row["sample_id"] == "s1"
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [1, 1, 1]
    assert [row[f"{kind}_class_1"] for kind in ("tp", "fp", "fn")] == [2, 1, 1]
    # No ignore mask: the whole 128 x 96 px slide at 0.5 um/px is annotated.
    assert row["annotated_area_mm2"] == pytest.approx(128 * 96 * 0.25 / 1e6)


def test_an_ensemble_s_points_are_refused_without_a_threshold(tmp_path, marker):
    """An ensemble's mean scores were tuned by no bundle, so its points record no
    threshold: scoring them needs one. The marker decoder has no weights, so the mean
    scores are each bundle's."""
    points = SlidePoints(predict(tmp_path, marker, bundles=("a", "b"), ensemble=True))
    assert points.threshold is None

    with pytest.raises(ValueError, match="record no score threshold.*with_threshold.*sweep_threshold"):
        soma.evaluate(points, targets(tmp_path))

    at_tune = soma.evaluate(points.with_threshold(0.9), targets(tmp_path), metrics=["f1_per_class"])
    assert at_tune.metrics["f1_class_0"] == pytest.approx(2 / 4)
    # At 0.7 the second mitosis peak (0.73) is kept: 2 true positives, 1 false positive.
    lower = points.with_threshold({"mitosis": 0.7, "mimic": 0.9})
    assert lower.threshold == {"mitosis": 0.7, "mimic": 0.9}
    evaluation = soma.evaluate(lower, targets(tmp_path), metrics=["f1_per_class"])
    assert evaluation.metrics["f1_class_0"] == pytest.approx(4 / 5)
    assert evaluation.metrics["f1_class_1"] == pytest.approx(4 / 6)


def test_points_on_the_not_annotated_region_of_a_low_resolution_ignore_mask_count_for_nothing(tmp_path, marker):
    """The ignore mask is 32 x 24 px at 2 um/px, a quarter of the slide's resolution, and
    marks x >= 64 level-0 px not annotated (255). Every peak and annotation there drops
    out: mitosis keeps 1 true positive and its missed point, F1 = 2 / 3; mimic keeps its
    false positive and its missed point, F1 = 0. The annotated area is the 16 x 24 mask
    pixels of 4 um² each."""
    output = predict(tmp_path, marker)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[:, 16:] = 255
    ignore = write_mask(tmp_path / "s1_ignore.tif", mask, spacing_um=2.0)

    evaluation = soma.evaluate(
        SlidePoints(output), targets(tmp_path, ignore_mask=ignore), metrics=["mean_f1", "f1_per_class"]
    )

    assert evaluation.metrics["f1_class_0"] == pytest.approx(2 / 3)
    assert evaluation.metrics["f1_class_1"] == 0.0
    (row,) = evaluation.per_sample
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [1, 0, 1]
    assert [row[f"{kind}_class_1"] for kind in ("tp", "fp", "fn")] == [0, 1, 1]
    assert row["annotated_area_mm2"] == pytest.approx(16 * 24 * 4.0 / 1e6)


def test_an_ignore_mask_is_read_to_its_last_row_and_column_on_each_axis_s_own_scale(tmp_path, marker):
    """The ignore mask is 32 x 25 px: 4 level-0 px per mask column, 3.84 per mask row, so
    hs2p aligns it by rounding the height. Its last row (level-0 y >= 93) and last column
    (x >= 124) are not annotated (255). The annotations added there drop out, so the
    counts are those of the slide without a mask, and an undeclared id there is not
    refused. A point within half a pixel of the far edge (x = 127.8) is on the slide."""
    output = predict(tmp_path, marker)
    mask = np.zeros((25, 32), dtype=np.uint8)
    mask[24, :] = 255
    mask[:, 31] = 255
    ignore = write_mask(tmp_path / "s1_ignore.tif", mask, spacing_um=2.0)
    on_ignored = [(50, 94, 1), (90, 95, 2), (127.8, 40, 2), (126, 10, 5)]

    evaluation = soma.evaluate(
        SlidePoints(output),
        targets(tmp_path, points=[*ANNOTATED, *on_ignored], ignore_mask=ignore),
        metrics=["f1_per_class"],
    )

    (row,) = evaluation.per_sample
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [1, 1, 1]
    assert [row[f"{kind}_class_1"] for kind in ("tp", "fp", "fn")] == [2, 1, 1]
    # 32 x 25 - 56 = 744 annotated mask pixels, each 128 x 96 / (32 x 25) level-0 px.
    assert row["annotated_area_mm2"] == pytest.approx(744 * (128 * 96 * 0.25 / (32 * 25)) / 1e6)


def test_an_ignore_mask_with_another_value_is_refused_naming_the_file(tmp_path, marker):
    output = predict(tmp_path, marker)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[3, 3] = 7
    ignore = write_mask(tmp_path / "s1_ignore.tif", mask, spacing_um=2.0)

    with pytest.raises(ValueError, match="slide 's1': its ignore mask .*s1_ignore.tif holds values other than 0") as refused:
        soma.evaluate(SlidePoints(output), targets(tmp_path, ignore_mask=ignore))
    assert "[7]" in str(refused.value)


def test_an_annotated_id_the_class_scheme_does_not_know_is_refused(tmp_path, marker):
    output = predict(tmp_path, marker)

    with pytest.raises(ValueError, match=r"slide 's1': its points file .*s1_points.csv has annotated id\(s\) \[5\]"):
        soma.evaluate(SlidePoints(output), targets(tmp_path, points=[*ANNOTATED, (5, 5, 5)]))


def test_points_written_without_a_class_scheme_are_refused(tmp_path, marker):
    """A points file of an older soma predict records no class scheme: its points cannot
    be scored against the annotated ids."""
    import json

    import pyarrow.parquet as pq

    output = predict(tmp_path, marker)
    path = output / "run" / "s1.parquet"
    table = pq.read_table(path)
    recorded = json.loads(table.schema.metadata[b"soma"])
    del recorded["classes"], recorded["drop"]
    pq.write_table(table.replace_schema_metadata({b"soma": json.dumps(recorded).encode()}), path)

    with pytest.raises(ValueError, match="records no class scheme .*older soma predict. Predict the slides again"):
        SlidePoints(output)


def test_the_per_slide_counts_use_the_matching_the_metrics_name(tmp_path, marker):
    """Two mitosis peaks, at x = 40 (the higher score) and 48, and two mitoses, at x = 43
    and 35. Hungarian matching pairs both peaks. Greedy matching pairs the first peak
    with its nearest mitosis, at 43, and the second peak reaches no other: 1 true
    positive, 1 false positive, 1 missed, F1 = 2 / 4. A mimic peak by a mimic is a true
    positive either way. The per-slide counts are those of the metrics' matching, and
    the default when the metrics name different ones."""
    markers = [(40, 40, 200, 0), (48, 40, 120, 0), (100, 70, 150, 1)]
    output = predict(tmp_path, marker, markers=markers)
    annotated = [(43, 40, 1), (35, 40, 1), (101, 71, 2)]
    greedy = {"name": "mean_f1", "params": {"matching": "greedy"}}

    evaluation = soma.evaluate(SlidePoints(output), targets(tmp_path, points=annotated), metrics=[greedy])

    assert evaluation.metrics["mean_f1"] == pytest.approx((2 / 4 + 1) / 2)
    (row,) = evaluation.per_sample
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [1, 1, 1]
    assert [row[f"{kind}_class_1"] for kind in ("tp", "fp", "fn")] == [1, 0, 0]

    mixed = soma.evaluate(
        SlidePoints(output), targets(tmp_path, points=annotated), metrics=[greedy, "f1_per_class"]
    )
    assert mixed.metrics["mean_f1"] == pytest.approx((2 / 4 + 1) / 2)
    assert mixed.metrics["f1_class_0"] == 1.0
    (row,) = mixed.per_sample
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [2, 0, 0]

    # An omitted distance is the recorded one, so these two metrics name one matching.
    recorded = SlidePoints(output).match_distance_um
    explicit = {"name": "f1_per_class", "params": {"matching": "greedy", "match_distance": recorded}}
    same = soma.evaluate(
        SlidePoints(output), targets(tmp_path, points=annotated), metrics=[greedy, explicit]
    )
    assert same.metrics["f1_class_0"] == pytest.approx(2 / 4)
    (row,) = same.per_sample
    assert [row[f"{kind}_class_0"] for kind in ("tp", "fp", "fn")] == [1, 1, 1]


def test_sweep_threshold_picks_the_threshold_of_an_ensemble_s_points(tmp_path, marker):
    """Per class: mitosis peaks the second mitosis only when its 0.73 peak is kept, F1
    = 4 / 5; mimic peaks with every peak kept, F1 = 4 / 6. The picked thresholds score
    the same through ``with_threshold``."""
    from soma.evaluation import sweep_threshold

    points = SlidePoints(predict(tmp_path, marker, bundles=("a", "b"), ensemble=True))

    best, evaluation = sweep_threshold(points, targets(tmp_path), per_class=True)

    assert evaluation.metrics["mean_f1"] == pytest.approx((4 / 5 + 4 / 6) / 2)
    assert best[0] <= score(20) and best[1] <= score(150)
    again = soma.evaluate(points.with_threshold(best), targets(tmp_path))
    assert again.metrics["mean_f1"] == pytest.approx((4 / 5 + 4 / 6) / 2)
    one, _ = sweep_threshold(points, targets(tmp_path))
    assert one <= score(20)
