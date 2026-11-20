# Whole-slide inference: `soma predict` writes one file per slide, `soma evaluate` scores it

`soma predict CONFIG` (`soma.predict_slides`) applies trained segmentation or detection bundles to the tissue tiles of a cohort's slides. It writes one file per slide and bundle: a label TIFF or a points parquet. `soma evaluate` scores those files against the slides' annotations, as it scores any other predictions. Prediction reads the pixels and runs each bundle's frozen encoder. It tiles each slide on one fixed grid and takes each map pixel from exactly one tile. It needs only the bundles and the slides.

## Why

A run trains and scores on the ROIs or tiles it samples. Nothing applied a bundle to a whole slide: there was no way to predict slides without annotations, or to score a model on every annotated pixel of a test slide. `SlidingWindowSegmentationPredictor` predicts one image held in memory. That does not scale to a slide of 100,000 × 100,000 pixels, and it handles segmentation only.

## Decisions

- **Two verbs.** `soma predict` writes predictions and `soma evaluate` scores them. Prediction needs no annotations, so one command serves annotated and unannotated slides. Because scoring is separate, one prediction can be scored again with other metrics, readers or thresholds, and an ensemble's threshold can be picked on tune predictions afterwards. The predict config follows ADR 0020: one section per group of `predict_slides` keywords, resolved to a `plan.yaml` that a relaunch replays or resumes.
- **Live features only.** Prediction reads the pixels and runs the encoder stage the bundle records (`bundle_encoder_stage`). It never reads a feature cache: a run caches only the ROIs or tiles it sampled, not the tiles of a slide grid.
- **Pick-one stitching on a fixed grid.** hs2p lays one grid over the whole slide from its top-left corner (`tiling.grid: slide`), so neighbouring tiles line up at a whole-pixel stride. With an overlap, each tile contributes the middle of its output, after a margin of `tile * overlap / 2` pixels that must be whole; the first row and column keep their outer margin. Nothing is averaged. Each pixel's class is final when its tile is decoded, so the map streams to disk band by band, as uint8, and its full-resolution level is never held whole. Detection uses the same rule, then one NMS per class over the slide; the margin must be at least the NMS distance.
- **Shifted class encoding.** A label TIFF stores class `k` as `k + 1`, and `0` means "not predicted": a pixel that no kept tile contributed. `0` is also what hs2p's writer gives a storage tile with no data. A run's PNG label maps store class `k` as `k`, because a run predicts each ROI in full and needs no "not predicted" value. The TIFF records the name of each value and the class scheme in its own metadata, so it can be read without the bundle. A map holds at most 255 classes.
- **One prediction per bundle by default.** Several bundles, such as a run's folds, each get their own directory, so each fold is scored on its own and their spread can be reported. `ensemble: true` is the option: the mean softmax, or the peaks of the mean heatmap, of bundles that share a class scheme. In both modes, bundles that share an encoder recipe share one encoder pass per tile.
- **Not predicted counts as a miss.** In scoring, an annotated pixel that no kept tile contributed is a false negative for its class and a false positive for no class. Each slide's row reports the share of its annotated pixels not predicted.
- **Self-sufficient bundles.** A bundle records all that prediction and scoring need: the encoder recipe with its slide reader, the tile size, the read spacing and tolerance, and the class scheme; a detection bundle also records `class_scheme` and its tune threshold. The config cannot override the tile size or the read spacing. Each prediction file copies what scoring needs, so `soma evaluate` reads no bundle.

## Considered and rejected

- **One verb that predicts and scores.** It would need annotations to run, and scoring again with another threshold or metric would mean predicting again.
- **Blending overlapping tiles**, as `SlidingWindowSegmentationPredictor` does with a Hann-weighted mean of softmaxes. It needs a float accumulator per class over every row of tiles in flight, and no single tile explains a pixel.
- **Leaving pixels that were not predicted out of the score.** A tissue mask that misses annotated tissue would then raise the score.

## Consequences

- `soma predict` is a new command, and `soma evaluate` accepts its output directories, with `--backend`, `--mask-backend` and `--threshold`.
- Code that reads a label TIFF must subtract one from each value; its metadata says so (`values`).
- A detection bundle saved before soma recorded `class_scheme` is refused until the scheme is added to its `preprocessing.json`.
- Slide scores are not comparable to a run's test metrics: a run scores only the ROIs or tiles it samples, and averages per sample over ROIs, not slides.
- Peak memory grows with a slide's size in pixels, because hs2p's writer holds the map's coarser levels. Scoring reads every window of the map and of the annotation raster, including the empty ones.
