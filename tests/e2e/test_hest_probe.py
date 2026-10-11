"""HEST's closed-form probe as a benchmark-side predictor over the verbs.

``HestBenchmark.run`` extracts a rank-1 tile set source through soma's feature cache, fits
``RidgePCAProbe`` on each fold's train spots, predicts the test spots and scores them with
``soma.evaluate`` (``mean_pearson`` + one Pearson per gene). Synthetic spots carry gene
expression that is a linear function of their colour, so only a probe that receives each
spot's own features and expression can correlate.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from PIL import Image

from soma.benchmarks import get_benchmark
from soma.config import ExecutionConfig
from tests.e2e.harness import CPU_EXECUTION

GENES = ["ERBB2", "ESR1", "MKI67"]


def _spots(root, n: int = 40, size: int = 224, seed: int = 0):
    rng = np.random.default_rng(seed)
    tiles = root / "tiles"
    tiles.mkdir(parents=True)
    colours = rng.uniform(40, 215, size=(n, 3))
    weight = np.array([[1.0, 0.2, -0.5], [-0.3, 1.0, 0.4], [0.6, -0.4, 1.0]])
    expression = np.log1p(np.abs(colours / 64.0 @ weight + 0.05 * rng.standard_normal((n, 3))))
    ids = [f"spot{i:02d}" for i in range(n)]
    for sample_id, colour in zip(ids, colours):
        image = np.broadcast_to(colour.astype(np.uint8), (size, size, 3)).copy()
        Image.fromarray(image).save(tiles / f"{sample_id}.png")
    pd.DataFrame(
        {
            "sample_id": ids,
            "image_path": [str(tiles / f"{s}.png") for s in ids],
            "target_index": list(range(n)),
            "spacing_at_level_0": 0.5,
        }
    ).to_csv(root / "dataset.csv", index=False)
    np.save(root / "targets.npy", expression)
    (root / "genes.json").write_text(json.dumps(GENES))
    split = ["train"] * 30 + ["test"] * (n - 30)
    pd.DataFrame({"sample_id": ids, "split": split}).to_csv(root / "splits.csv", index=False)
    return root / "dataset.csv", root / "splits.csv"


def test_hest_run_composes_extract_fit_predict_evaluate(tmp_path, encoder_name, new_artifact):
    artifact = new_artifact("hest_probe_verbs")
    dataset_csv, splits_csv = _spots(tmp_path / "data")
    benchmark = get_benchmark("hest/IDC")
    config = benchmark.build_config(
        encoder=encoder_name,
        dataset_csv=dataset_csv,
        splits_csv=splits_csv,
        output_root=tmp_path / "run",
        overrides={"cache": {"enabled": True, "root_dir": str(tmp_path / "cache")}},
        pca_components=2,
        execution=ExecutionConfig(**CPU_EXECUTION),
    )

    summary = benchmark.run(config)

    run = tmp_path / "run"
    metrics = json.loads((run / "metrics.json").read_text())["test"]
    artifact.check_equal("summary/keys", sorted(summary)[:1], ["test/mean_pearson"])
    artifact.check_equal(
        "per_gene_keys", sorted(k for k in metrics if k.startswith("pearson_gene")),
        [f"pearson_gene{k:02d}_{gene}" for k, gene in enumerate(GENES)],
    )
    artifact.check_equal(
        "mean_is_mean_over_genes",
        round(metrics["mean_pearson"], 9),
        round(float(np.mean([metrics[f"pearson_gene{k:02d}_{g}"] for k, g in enumerate(GENES)])), 9),
    )
    artifact.check_at_least("test/mean_pearson", summary["test/mean_pearson"], 0.8)
    artifact.check_equal("score/reads_summary", benchmark.score(run)["test/mean_pearson"], summary["test/mean_pearson"])
    artifact.check_equal("predictions/written", (run / "predictions_test.parquet").is_file(), True)
    artifact.assert_passed()
