"""Point-annotation reader for detection (design §3).

Loads a per-sample point file into ``(xy (N, 2), classes (N,))`` in the **level-0**
coordinate frame it is persisted in. v1 wire format is CSV with ``x, y, class``
columns (OCELOT ships this directly); a headerless ``x,y,class`` (or ``x,y``,
single-class) CSV is also accepted. GeoJSON can slot in here later behind the same
return contract without touching the head.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["read_points"]

_CLASS_ALIASES = ("class", "label", "category", "cls")


def read_points(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a point-annotation file → ``(xy (N, 2) float64, classes (N,) int64)``.

    Coordinates are returned unchanged in their stored (level-0) frame. An empty
    file yields ``((0, 2), (0,))``. Class ids are taken from the first of
    ``class``/``label``/``category``/``cls`` present; a 2-column ``x,y`` file is
    treated as single-class (all 0).
    """
    path = Path(path)
    # Sniff a header: if the first row's first two fields parse as floats, it is
    # headerless (OCELOT-style x,y,class with no header line).
    first = ""
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                first = line.strip()
                break
    if not first:
        return np.zeros((0, 2), dtype=np.float64), np.zeros((0,), dtype=np.int64)

    fields = first.split(",")
    headerless = False
    try:
        float(fields[0])
        float(fields[1])
        headerless = True
    except (ValueError, IndexError):
        headerless = False

    if headerless:
        df = pd.read_csv(path, header=None)
        ncol = df.shape[1]
        if ncol < 2:
            raise ValueError(f"point file '{path}' needs at least x,y columns, got {ncol}.")
        xy = df.iloc[:, :2].to_numpy(dtype=np.float64)
        classes = (
            df.iloc[:, 2].to_numpy()
            if ncol >= 3
            else np.zeros(len(df), dtype=np.int64)
        )
    else:
        df = pd.read_csv(path)
        cols = {c.lower(): c for c in df.columns}
        if "x" not in cols or "y" not in cols:
            raise ValueError(
                f"point file '{path}' must have 'x' and 'y' columns; got {list(df.columns)}."
            )
        xy = df[[cols["x"], cols["y"]]].to_numpy(dtype=np.float64)
        class_col = next((cols[a] for a in _CLASS_ALIASES if a in cols), None)
        classes = (
            df[class_col].to_numpy() if class_col is not None else np.zeros(len(df), dtype=np.int64)
        )

    classes = np.asarray(classes).reshape(-1).astype(np.int64)
    if xy.shape[0] != classes.shape[0]:
        raise ValueError(f"point file '{path}' has mismatched coordinate/class counts.")
    return xy.reshape(-1, 2), classes
