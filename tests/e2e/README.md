# End-to-end scenarios

Each scenario writes synthetic images to disk, then runs them through the real
`soma run <config.yaml>` entry point: tiling, feature extraction, caching, training and
evaluation. Nothing is mocked except the encoder, which is weight-free: it returns
the mean normalised RGB of each tile or 8x8 patch (`soma.testing`'s `fake` encoder, or
`soma.testing.FakeEncoder`, the same pooling, as the frozen encoder stage of the live grid
path). The whole-slide cohort is `soma.testing.synthetic_cohort`. The
synthetic labels are a function of tissue colour, so the model can only score well
if the correct pixels, features and labels reach it.

| Scenario | Covers |
| --- | --- |
| `slide_mil_binary_classification` | WSI tiling → extraction → cache → ABMIL → evaluation; a second run is a full cache hit (zero images encoded) with byte-identical predictions |
| `slide_mil_{multiclass_classification,ordinal_classification,regression,survival_nll,survival_cox}` | every slide task head on one cohort; the feature cache is shared across tasks |
| `slide_mil_survival_nll_train_quantiles` | `discrete_survival` with `binning: train_quantiles` on a cohort without a `bin` column: the bundle's `model.json` and the head `soma.load` rebuilds carry the quartiles of the training split's uncensored times, and `plan.yaml` holds `binning` and `num_bins` but no edges |
| `slide_mil_flat_png` | flat PNG slides with `spacing_at_level_0` in the manifest: tiling, both default previews, on-the-fly extraction and training |
| `slide_mil_supplied_coordinates` | a `coordinates_path` manifest column (one hand-written hs2p tiling artifact per slide) replaces tiling: every bag holds exactly the supplied tiles, a second run is a full cache hit, and editing one slide's artifact re-encodes only that slide |
| `slide_mil_cv_leaderboard` | 2-fold cross-validation for two aggregators, then `soma leaderboard` ranks them |
| `mil_aggregator_<name>` | every registered aggregator on a rigged task (a slide is positive iff it has marker tiles): test accuracy and AUROC must be exactly 1, and re-scoring the trained model must give the same logits alone, in one padded batch, with garbage in the padding, and (for set-function aggregators) with tiles shuffled |
| `mil_aggregator_control` | the same cohort with labels independent of the markers must stay near chance, so the perfect scores above cannot come from a label leak |
| `tile_binary_classification` | a tile encoder with `learner: {name: mil, params: {head: …}}` and no `aggregator` leaf: each image is one sample, with no bag |
| `dense_segmentation` | ROI → dense token grid → decoder → Dice / IoU; the label maps of tune and test land in `predictions_tune/` / `predictions_test/` |
| `dense_detection` | ROI → dense token grid → heatmap decoder → F1 at a matching distance; the points land in `predictions_test.parquet` and no `detection_thresholds.json` is written |
| `dense_preset_{segmentation,detection}` | `soma run --preset segmentation\|detection` with only the preset's own flags (`--num-classes`, `--match-distance`) on ROIs at the encoder's recommended tile size: the plan's head carries the flags, the run trains (loss falls) and is scored on test |
| `dense_attention_map_{cell_head,noise_heads}` | `decoder: attention_map` over hand-written attention-like grids (one channel marks the cells, three are noise) through `soma run --feature-dir`: nothing is trained (empty history, no weights in the bundle), the threshold is frozen on tune, and `heads: [0]` finds the cells (F1 ≥ 0.8) while the noise heads do not (F1 ≤ 0.5). No loss check: there is no training |
| `dense_live_segmentation` | `features.kind: live` through the CLI, with the run's encoder stage replaced by `FakeEncoder`: one stage built, zero ROIs extracted to the cache, the train split augmented through `loaders.augment: {name: segmentation, params}` → Dice |
| `dense_live_detection` | `features.kind: live` detection through the CLI with `FakeEncoder` (no augmentation) → F1 at a matching distance |
| `dense_live_detection_augmented` | live detection with augmentation: the same run with `loaders.augment` (crop, flips, rotation); the train points move with the pixels and points cropped out are dropped → F1 at a matching distance |
| `grid_live_segmentation` | the verbs over pixels: `TileImageSource` + `FakeEncoder` + `LabelMapSource` → `soma.learn.loaders` (with `SegmentationAugment`) → `soma.fit` → `soma.predict` (`LabelMapPredictions`) → `soma.evaluate`; a `save` / `load` round trip scores the same |
| `grid_live_detection` | the verbs over pixels with a `PointSource`: `loaders` → `fit` → `predict` (`PointPredictions`, every score above `min_score`) → per-class `sweep_threshold` on tune → `evaluate` on test; a parquet round trip scores the same |
| `reproduce_eva_consep` | `soma reproduce eva/consep --raw-root …` on a synthetic HoVer-Net layout: curation (16 tiles per image, 224 px) → dense run with the benchmark-private `eva_conv_with_image` decoder (fed the tile's pixels) and `eva_segmentation` head → `test/foreground_mean_dice` scored |
| `reproduce_eva_{camelyon16_small,panda_small}` | `soma reproduce eva/<dataset> --raw-root …` on tiny pyramidal-TIFF raw roots: the curator samples EVA's tiles into one hs2p artifact per slide → soma skips tiling and embeds exactly those tiles → `eva_abmil` + the EVA head → `test/balanced_accuracy` scored |
| `reproduce_hest` | `soma reproduce hest/IDC --curated-dir …` and `run_benchmark` on two folds of synthetic spots: HEST's own runner (`HestBenchmark.run`, no `Pipeline`) fits the closed-form probe, and both entry points score the same `test/mean_pearson_mean` |
| `benchmark_spec_hest` | `run_benchmark_spec` on a `BenchmarkSpec` that wraps the registered HEST recipe and passes its runner as `run`: no `Pipeline` is built, and the result's dataset, splits and test checksums are the ones `experiment.json` and `run.yaml` record |
| `reproduce_detection_midog_{1,3}_fold` | `soma reproduce detection/midog --from-run-dir` on a run over synthetic ROIs tiled by `soma.benchmark.tiles.tile_detection_manifest` (64 px tiles, 32 px overlap), with one fold and with three: per fold, thresholds frozen on the stitched tune ROIs, test stitched back to its ROIs in `detection_score/[fold_k/]predictions.json`; the reported F1 equals `score_dataset_points` over those files (mean over folds), and a second call re-scores off them with the parquet files deleted |
| `extension_custom_head_decoder` | a user-registered `SegmentationHead` subclass (pure Dice loss) and a user-registered `Decoder` train through the CLI; both overrides are proven to run |
| `extension_custom_aggregator` | a user-registered `Aggregator` (a projected mean over the real tiles) trains the marker cohort through the CLI under its registered name, on batches of padded bags: test accuracy 1, the override is proven to run on padded bags, `plan.yaml` holds its validated params, and it passes `check_aggregator` |
| `extension_custom_grid_learner` | a `DenseLearner` subclass adding a smoothness penalty (one `training_step` override) trains a cached segmentation run through the CLI, named by dotted import path in `learner.name` |
| `extension_custom_learner` | a `MILLearner` subclass adding an attention-entropy term (one `training_step` override) trains a user-registered head (label-smoothed loss) through the CLI, named by dotted import path in `learner.name`; the history carries `loss` columns without a `tune_` prefix; its `predict_step` override adds an `attention_entropy` column to `predictions_test.parquet`, the bundle's `provenance.json` records the learner's class and the digest of its file as its fingerprint, and the custom head passes `check_task_head` |
| `extension_component_code_guards` | a registered aggregator in a module file trains seed 0, and `run.yaml` and the bundle record its class and the file's digest as its fingerprint. A resume passes while the file is unchanged and is refused after an edit to `forward`, naming the aggregator and both fingerprints. Seed 1 on the edited code shares the experiment id, and `soma leaderboard` refuses to pool the two seeds; a rerun of seed 0 replaces the stale run, and the board pools seeds 0 and 1 |
| `extension_component_without_source` | an aggregator defined in a module without a file (as in a notebook cell) warns once at run start and is recorded with a null fingerprint |
| `evaluate_external_grid_predictions` | a hand-written label-map directory and points parquet are scored by `soma evaluate` against `dataset.csv` (`label_mask_path`, `points_path`) and `splits.csv`, with no run directory: hand-worked Dice, IoU and F1; then, through `soma.evaluate`, greedy matching as a metric-leaf param and a re-score after `with_threshold` |
| `evaluate_external_predictions` | predictions soma did not produce (a hand-written parquet) are scored against `dataset.csv` / `splits.csv` through `soma.evaluate` and `soma evaluate`, with no run directory: hand-worked AUROC, balanced accuracy, subgroup sizes and an unsplit regression MAE |
| `evaluate_cli_multi_output_regression` | a `value_0`, `value_1` predictions parquet (a `RegressionHead(num_outputs=2)` run's columns) is scored by `soma evaluate` against the same-named cohort columns: `mean_pearson` by default, hand-worked to 0 |
| `parity_set_mil_classification` | `soma run config.yaml --feature-dir` over a pooled cache with one tile-less test sample, then the same fold composed by hand from the public verbs in the canonical order (seed → adaptor fit on train → model → loaders → `Baseline.fit` → `soma.fit` → predict + baseline rows → evaluate → `soma.save`), without importing `soma.pipeline`: byte-identical `predictions_test.parquet` and `bundle/weights.safetensors`, equal metrics, no `best_model.pt` |
| `parity_grid_segmentation` | the same pipeline-versus-hand comparison over cached segmentation grids: identical label-map PNGs and indexes for tune and test, equal Dice / IoU, identical weights |
| `parity_grid_detection` | the same comparison over cached detection grids, with the per-class threshold swept on tune (step 7) and applied to every split; the bundle records the chosen threshold |
| `verbs_journey` | the library journey over user arrays: `Cohort` + `from_arrays` → `soma.learn.loaders` → `soma.fit` → `soma.predict` → `soma.evaluate`, with string labels mapped through the head's classes |

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
