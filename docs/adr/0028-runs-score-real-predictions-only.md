# Runs score real predictions only; one policy decides what a target without a prediction does

A set-shape run no longer invents a prediction for a sample it cannot predict: a predictions file holds one row per sample the model predicted. A target without a prediction follows one policy, with one name in three places: `evaluation.unpredicted` in a config, `--unpredicted` on `soma evaluate` and `unpredicted=` on `soma.evaluate`. `error`, the default, refuses to score and names the samples; `exclude` scores the predicted samples only. Every set-shape evaluation reports `num_samples` (the targets), `num_predicted` and `coverage`. This supersedes the "Baselines are predictors" decision of [ADR 0014](0014-evaluate-scores-the-ids-it-is-given.md) and step 5 of the canonical fold order of [ADR 0017](0017-pipeline-composes-the-verbs-in-a-canonical-order.md).

## Why

A run filled each sample without features with a constant fitted on the train split (the class prior, the mean value, or a risk of 0.0), flagged the row and scored it with the real predictions.

- It was silent: the filled rows moved the headline metric, and the only traces were one column, four metric keys and one line of the run panel.
- It was not neutral for ranking metrics. A risk of 0.0 lands wherever 0.0 falls among a model's risk scores, so its effect on the C-index is arbitrary.
- It rarely did what it was added for. Missing features come from tiling, so two models on the same features miss the same samples.

## Decisions

- **No invented predictions.** The fill, its flag column and `soma.evaluation.Baseline` are removed. Next to each test split's predictions, a run writes `not_predicted_<split>.parquet` (`sample_id`, `reason`), even when it has no row.
- **Refuse early.** The samples without features are known once the features are extracted. Under `error`, the run checks the tune and test splits of every fold before any fold trains, and stops with one message for all of them. Completed folds are checked too, so a resume under `error` refuses a run that an `exclude` launch completed.
- **Train samples are not scored.** A train sample without features is left out of training, as before, and the fold's log line counts it. The policy does not apply to it.
- **The policy is not part of the experiment identity.** A completed fold scores the same under either policy, because `error` stops a run that misses a sample before any fold trains. `evaluation.unpredicted` is left out of the identity and of the resume comparison, so a run that `error` refused resumes with `exclude`.
- **Set shape only.** Grid predictions cover every sample of their target source. A grid run, and grid predictions in `soma.evaluate` and `soma evaluate`, refuse `exclude`.
- **`compare` stays strict.** A paired permutation test needs the same samples in every set, so `soma.evaluation.compare` takes no policy. Its error says to filter the targets and the sets to the ids every set predicts.
- **The end of a run warns.** Under `exclude`, a warning names each split whose coverage is below 1.
- **The panel and the leaderboard show coverage.** The run panel prints each split's coverage with its counts (means per fold over folds). `soma leaderboard` gives each row the mean coverage of its runs, and adds a `coverage` column only when a row is below 1, so a board of full runs reads as before.

## Considered and rejected

- **Keep the fill as an option.** The number it produces stays easy to miss, and its effect on ranking metrics stays arbitrary.
- **`exclude` by default.** A metric over fewer samples than the cohort would be easy to miss. With `error` by default, leaving samples out is a choice the config states.
- **A `not_predicted` file for tune.** The tune split's predictions are not written. Its counts are in `metrics.json`, and the end-of-run warning names it.

## Consequences

- Breaking: a run with a tune or test sample without features stops by default. With `exclude`, its metrics change: the filled rows no longer count.
- `num_predicted` replaces `num_real_samples`, and `num_placeholder_samples` is gone. `num_samples` and `coverage` keep their names, and every set-shape evaluation reports the three counts. `Evaluation.num_samples` counts the targets.
- The canonical fold order loses its step 5, and the steps after it move up by one: 5. fit, 6. the tune-time step, 7. predict and evaluate per split, 8. save. The set parity scenario keeps its tile-less test sample and runs under `exclude` on both sides.
- No recorded benchmark row changes. The EVA patch tasks embed one image per sample, with no tiling; the EVA slide curator leaves slides without a foreground tile out of the manifest; the segmentation and detection benchmarks are grid runs; HEST uses its own runner; CRoMa runs are representation runs.
