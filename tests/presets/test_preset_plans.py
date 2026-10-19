"""The segmentation and detection presets, pinned as the plans they resolve to.

Their head's class count and matching distance are preset arguments, so ``soma run
--preset segmentation|detection`` builds a head; a value that moves (and so moves the run
identity) fails here. Regenerate the fixture only for a deliberate change:
``python -m tests.presets.test_preset_plans``.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from soma import presets
from soma.plan import load_cohort, resolve

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "preset_plans.yaml"

#: ``(case id, preset, arguments)``: each preset with one foreground class, at its defaults.
CASES: list[tuple[str, str, dict[str, Any]]] = [
    ("segmentation@uni2+heavy_conv", "segmentation", {"encoder": "uni2", "decoder": "heavy_conv", "num_classes": 2}),
    (
        "detection@uni2+lightweight_conv",
        "detection",
        {"encoder": "uni2", "decoder": "lightweight_conv", "num_classes": 1},
    ),
]


def preset_plan(preset: str, arguments: dict[str, Any], root: Path) -> dict[str, Any]:
    """The comparable plan ``preset`` resolves to, with the cohort files' paths left out."""
    ids = [f"s{index}" for index in range(8)]
    root.mkdir(parents=True, exist_ok=True)
    records, splits = root / "dataset.csv", root / "splits.csv"
    pd.DataFrame({"sample_id": ids, "image_path": [f"/images/{s}.png" for s in ids]}).to_csv(records, index=False)
    pd.DataFrame({"sample_id": ids, "split": ["train"] * 6 + ["tune", "test"]}).to_csv(splits, index=False)
    config = getattr(presets, preset)(**arguments)
    config.cohort.records, config.cohort.splits = records, splits
    plan = resolve(config, load_cohort(config)).comparable()
    plan["cohort"].update(records="<records>", splits="<splits>")
    return plan


@pytest.mark.parametrize(("case", "preset", "arguments"), CASES, ids=[case[0] for case in CASES])
def test_a_dense_preset_resolves_to_its_pinned_plan(tmp_path: Path, case, preset, arguments):
    pinned = yaml.safe_load(FIXTURE.read_text(encoding="utf-8"))
    assert preset_plan(preset, arguments, tmp_path) == pinned[case]


def _write_fixture() -> None:
    with tempfile.TemporaryDirectory() as directory:
        plans = {
            case: preset_plan(preset, arguments, Path(directory) / str(index))
            for index, (case, preset, arguments) in enumerate(CASES)
        }
    FIXTURE.write_text(yaml.safe_dump(plans, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    _write_fixture()
