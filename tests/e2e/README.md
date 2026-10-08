# End-to-end scenarios

Each scenario writes synthetic images to disk, then runs them through the real
`soma <config.yaml>` entry point: tiling, feature extraction, caching, training and
evaluation. Nothing is mocked except the encoder, which is weight-free: it returns
the mean RGB of each tile or 8x8 patch (`tests/dense_literal_encoder.py`). The
synthetic labels are a function of tissue colour, so the model can only score well
if the correct pixels, features and labels reach it.

| Scenario | Covers |
| --- | --- |
| `slide_mil_binary_classification` | WSI tiling → extraction → cache → ABMIL → evaluation; a second run is a full cache hit (zero images encoded) with byte-identical predictions |
| `slide_mil_{multiclass,ordinal,regression,survival_nll,survival_cox}` | every slide task head on one cohort; the feature cache is shared across tasks |
| `slide_mil_flat_png` | flat PNG slides with `spacing_at_level_0` in the manifest: tiling, both default previews, on-the-fly extraction and training |
| `slide_mil_supplied_coordinates` | a `coordinates_path` manifest column (one hand-written hs2p tiling artifact per slide) replaces tiling: every bag holds exactly the supplied tiles, a second run is a full cache hit, and editing one slide's artifact re-encodes only that slide |
| `slide_mil_cv_leaderboard` | 2-fold cross-validation for two aggregators, then `soma leaderboard` ranks them |
| `mil_aggregator_<name>` | every registered aggregator on a rigged task (a slide is positive iff it has marker tiles): test accuracy and AUROC must be exactly 1, and re-scoring the trained model must give the same logits alone, in one padded batch, with garbage in the padding, and (for set-function aggregators) with tiles shuffled |
| `mil_aggregator_control` | the same cohort with labels independent of the markers must stay near chance, so the perfect scores above cannot come from a label leak |
| `tile_binary_classification` | `dataset_type: tile` (no bag, no aggregator) |
| `dense_segmentation` | ROI → dense token grid → decoder → Dice / IoU |
| `dense_detection` | ROI → dense token grid → heatmap decoder → F1 at a matching distance |
| `reproduce_eva_consep` | `soma reproduce eva/consep --raw-root …` on a synthetic HoVer-Net layout: curation (16 tiles per image, 224 px) → dense run with the benchmark-private `eva_conv_with_image` decoder (fed the tile's pixels) and `eva_segmentation` head → `test/foreground_mean_dice` scored |
| `extension_custom_head_decoder` | a user-registered `SegmentationHead` subclass (pure Dice loss) and a user-registered `Decoder` train through the CLI; both overrides are proven to run |

Every scenario checks that training loss fell (last epoch ≤ 0.9 × first epoch), as
well as checking its metric thresholds.

## Running

```bash
python -m pytest tests/e2e          # ~3.5 min on 4 CPU cores
python -m pytest -m "not e2e"       # everything else
```

Scenarios always run on CPU, even on a GPU node.

## Artifacts

Each scenario writes `e2e-artifacts/<scenario>.json` (override the directory with
`SOMA_E2E_ARTIFACT_DIR`). An artifact contains:

- every check, with its value, its expectation and whether it passed;
- the resolved config of each run;
- the metrics each run reported;
- SHA-256 digests of the prediction files.

Artifacts contain no timestamps, run ids or absolute paths. Two sessions on the same
code produce byte-identical artifacts, so a `diff` between two artifact directories
shows any behaviour change:

```bash
SOMA_E2E_ARTIFACT_DIR=/tmp/before python -m pytest tests/e2e
# ... change code ...
SOMA_E2E_ARTIFACT_DIR=/tmp/after python -m pytest tests/e2e
diff -r /tmp/before /tmp/after
```
