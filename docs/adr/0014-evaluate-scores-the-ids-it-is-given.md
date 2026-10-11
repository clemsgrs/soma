# Evaluate scores exactly the ids it is given; baselines are predictors; no HTML report

Inference and scoring are two public verbs with one noun between them. `soma.predict(learner_or_model, loader, callbacks=())` returns `Predictions`, a frame keyed by `sample_id` whose other columns the task head declares (`TaskHead.prediction_columns` / `to_columns`). `soma.evaluate(predictions, targets, metrics="default", subgroups=())` joins them with the cohort's targets (`Cohort.targets(split, fold)`) and returns a frozen `Evaluation`. The pipeline's set-shape folds compose these verbs; the private evaluation path (`_evaluate`, `SamplePrediction`, `EvaluationReport`, the CSV writer) is gone, and `soma/reporting/` with it.

## Why

The old path fused inference, target bookkeeping, placeholder filling, metric computation and file writing in one private function per shape, and copied the target into every prediction row (`true_label`). Scoring predictions made elsewhere, or re-scoring a run with another metric or subgroup, meant reimplementing it. The HTML report rendered from the same private records and had to follow every change to them.

## Decisions

- **Targets never travel inside predictions.** A predictions file holds what the model said; the truth comes from the cohort at scoring time. `predictions_<split>.parquet` replaces the CSV for the set shape; its schema metadata records the task family, the soma version, each column's role and the class label of each probability column.
- **`evaluate` scores exactly the ids it is given.** The join is a strict set equality: a prediction without a target, a target without a prediction, or a duplicate id is refused with the count. It never drops or invents rows, so a number always describes a known set of samples.
- **Baselines are predictors.** Samples the model cannot score (no tiles, no embedding) are filled by `Baseline.fit(train_targets).predict(ids)` (class prior, mean value, constant risk), flagged `is_placeholder`. `evaluate` then adds `coverage` and the real / placeholder counts. The fill is an explicit step before scoring, not a side effect of it.
- **Metrics are frame functions in a registry.** `(predictions, targets) -> float` over the aligned frames, registered with their families and direction in `metric_registry`. The array functions stay as numerical oracles behind one-line adapters. The registry answers `higher_is_better`, which the leaderboard reads. A subgroup breakdown is the same metrics over a `groupby` of the joined frame.
- **Side artefacts are callbacks.** `predict` takes no artefact keywords; attention maps are `soma.heatmaps.SaveAttention(dir)`, hooked through `on_predict_batch(batch, output)`. The grid shape's writers use the same hook from one private `_evaluate_grid` until slice 4 gives the grid its own `Predictions`.
- **No HTML report.** `report.html`, `generate_report`, `compare_runs` and the comparison page are deleted. The leaderboard absorbs the config diff and the stylesheet it used and keeps its behaviour; `soma.evaluation.compare` keeps the paired permutation test over `Predictions`; `soma evaluate` scores a predictions file against a cohort without a run directory.

## Consequences

- `FoldResult` carries `tune_evaluation` / `test_evaluations` (`Evaluation`s). `metrics.json` keeps `{split: metrics}` and `subgroup_metrics_<split>.json` keeps its name and shape. Grid outputs (segmentation rasters and CSV, the detection level-0 points CSV) are unchanged.
- Labels map to class indices through the predictions' `classes`; without them, target labels must already be class indices. A patient-level target takes its metadata from the patient's first slide, as `Cohort.collapse` does.
- `soma.testing.check_task_head` is the conformance suite for set-shape heads and runs over every built-in one.
- Old run directories keep their `predictions_<split>.csv` and `report.html`; nothing converts them.
