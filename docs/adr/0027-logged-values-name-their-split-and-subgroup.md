# Logged values name their split and subgroup; monitors are pooled tune values or means over groups

Every value the Trainer logs at an evaluated epoch is named `<split>/<name>`, where the split is `tune`, or `train` for the eval-mode pass over the train split (`trainer.eval_train`). A pooled value is `<split>/<key>` (`tune/loss`, `tune/auroc`), with `<key>` a key of the learner's `validation_summary`. With `evaluation.subgroups`, one group's value is `<split>/<column>=<group>/<key>`, and the unweighted mean over a column's groups is `<split>/mean_over_<column>/<key>`, with `<key>_groups` counting the groups in the mean. `summary.json` uses the same names for the scores after the fit, with `_mean`, `_std` and `_nan_folds` over several folds. A `checkpoint` or `early_stopping` monitor names a pooled tune value or a tune mean over groups, and nothing else. This supersedes the monitor names of ADR 0013: bare `validation_summary` keys, and the default monitor `loss`.

## Why

A run that held its test split out and selected on tune could not see a value per subgroup, for example a C-index per centre when centres with different follow-up share the folds. The history held bare keys (`loss`, `auroc`) that were tune values by convention only, so a train value or a group's value had no name. The only train value was `train_loss`, a running mean over the epoch's batches in train mode while the weights change, so there was no clean train-vs-tune gap.

## Decisions

- **The split is part of the name.** The Trainer adds the prefix; learners keep returning bare keys, so a custom learner needs no change. `train_loss` keeps its name: it is the optimizer's running objective, auxiliary losses included, not an evaluation of the train split.
- **Groups are `<column>=<group>`, means are `mean_over_<column>`.** The first segment after the split tells a subgroup key from a pooled one, so a column or a group value that holds `/` or `=` is refused at plan time and by `Trainer.fit`. A group needs at least two samples, at every epoch as after the fit.
- **The mean over groups is unweighted and skips NaN groups.** A group with one class or no comparable pairs has no AUROC or C-index. Leaving it out keeps the mean defined, and `<key>_groups` says how many groups the mean covers. With no finite group, the mean is NaN.
- **Monitors are `tune/<key>` and `tune/mean_over_<column>/<key>`.** `tune/loss` is the default. A `train/` value is a diagnostic: selecting on it selects on the training data. One group's value is refused because a fold's tune split can lack the group, and the monitor would then not exist. A `mean_over` monitor must name an `evaluation.subgroups` column. A bare name is refused with an error that names the `tune/` form.
- **One loss over the joined split.** The set-shape learner computes `loss` once over the split's joined outputs, so a group's loss is the same function over the group's rows. For a Cox head, a group's loss is the partial likelihood over the group's own risk sets: a stratified value, so the groups' losses do not average to the pooled loss. A Cox evaluation loss over rows with no event is NaN, as the C-index is.
- **A NaN monitor fails fast**, in `Checkpoint` and in `EarlyStopping`. It can never improve, so selection would never move and early stopping would stop the run without saying why.
- **The train pass is opt-in.** It scores the train split in eval mode, over an unshuffled loader and never the training loader, at each tune evaluation, under a forked RNG, so the training is the same with it or without it. It is not part of the experiment identity, but a resume refuses a change to it.

## Considered and rejected

- **An alias from bare names to `tune/`.** Old configs would keep running, but one value would have two names. The rename is a clean break instead.
- **A monitor on one group.** A fold whose tune split lacks the group would have no monitor.
- **Train metrics from the `training_step` outputs.** They would cost no extra pass, but dropout, augmentation and the weights changing through the epoch bias them.

## Consequences

- Breaking: `monitor: loss` becomes `monitor: tune/loss`, and `monitor: auroc` becomes `monitor: tune/auroc`. The plan spells each callback's monitor, so a run made before the change cannot be resumed, and the same config now has another experiment id.
- `training_history.json` holds `tune/<key>` columns, and subgroup and `train/` columns when they are on. A bundle's `selection_monitor` records the prefixed name; a bundle saved before the change still loads, with the bare name.
- With `evaluation.subgroups`, every fold writes `subgroup_metrics_tune.json`, and each `subgroup_metrics_<split>.json` gains a `mean_over` block. `summary.json` holds the subgroup keys of each split it covers; it covers tune only with `evaluation.holdout_test: true`.
- The run-summary panel and `soma leaderboard` take a pooled metric as the primary metric, never a subgroup key.
- The progress panel and log lines show the pooled values and the monitors; per-group values go to `training_history.json` only.
- Subgroups and the train pass are for the set shape: the grid shape refuses both. A set-shape learner needs the `subgroup_summary` hook to run with subgroups.
- With the `mil` learner, `tune/loss` weighs every sample the same. For a mean-reduced loss it equals the old mean of batch means when all batches have the same size, and differs slightly when the last batch is smaller. A fully censored Cox tune split now fails fast on a loss monitor.
