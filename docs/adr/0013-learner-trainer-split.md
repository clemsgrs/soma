# Training is a subclass-first `Learner` driven by a generic `Trainer` with callbacks

soma's training loop is split in two. A `Learner` (`soma.learn.Learner`, an `nn.Module`) owns the model and the step logic: `training_step`, `validation_step`, `validation_summary`, `configure_optimizers`, `transfer_batch`, `trainable_state_dict`. The `Trainer` (`soma.learn.Trainer`) is a generic epoch/step-budgeted loop with gradient accumulation and callbacks; it knows nothing about tasks, heads, shapes or folds. `MILLearner` (set shape, `(N, D)`) and `DenseLearner` (grid shape, `(H, W, D)`) are soma's two learners; `soma.fit` is sugar over both. This replaces `soma.training.trainer.Trainer`, which switched on head flags inside one 1300-line class.

## Why

The old loop grew a branch for every variation: a windowed Cox epoch, a streaming dense evaluation, a full-cohort loss, a guessed auxiliary target, an `image` side input. A user who wanted an intermediate loss had to fork the loop. Extension is now one override of `training_step` that calls `forward_batch` and `compute_loss` and adds a term, and the loop stays under ~150 lines.

## Decisions

- **Subclass-first.** The object a user extends is the `Learner`, in Lightning's dialect (`LightningModule` in role, fastai's word). `MILLearner` and `DenseLearner` are the two defaults; the `training.learner` config leaf names a subclass by registry key or dotted import path.
- **Validation is two explicit methods.** `validation_step` per batch returns a `StepOutput`; `validation_summary` reduces the explicit list for the split and must return `"loss"`. The learner keeps no state between hooks, so an override cannot forget to reset. The three reductions the old loop switched on (mean of batch losses, one full-cohort loss, concatenated dense stat rows) live in the learners.
- **The Cox window is a batch.** The collate decides what a batch is, `forward_batch` decides how it is forwarded, the Trainer decides the update cadence. Gradient accumulation applies to windows like any batch; `accumulates_predictions` steers only the collate choice.
- **Aggregator auxiliary losses receive the whole target dict** (`compute_auxiliary_loss(auxiliary, targets, mask)`). CLAM, DSMIL and DTFD read `targets["label"]` and raise when a head provides none; the old loop guessed and silently dropped the term.
- **Optimizers and schedulers are `{name, params}` leaves** resolved through a registry with torch's classes pre-registered. The default `configure_optimizers` injects the `requires_grad` parameters and derives cosine `T_max` from an explicit `Budget`. A warm-up is an ordinary `constant` leaf at interval `epoch`; the dedicated `lr_warmup_*` knobs are gone. The Trainer steps each scheduler at its own interval and composes nothing.
- **Explicit `Budget`, no back-reference.** `configure_optimizers(budget)` receives `max_epochs`, `max_steps` and `updates_per_epoch` instead of reading `self.trainer`, so a learner is testable without a loop.
- **Callbacks have no loss authority.** Hooks are `on_fit_start`, `on_epoch_start`, `on_train_batch_end`, `on_validation_end`, `on_epoch_end`, `on_fit_end`; a callback stops the run with `trainer.should_stop = True`. `Checkpoint` owns selection (CPU snapshot on improvement, file only with a `dirpath`, restored at `on_fit_end`); `EarlyStopping` owns patience with the hold on an unmoved monitor; `History` is always installed; `Progress` is the panel and the plain log. Monitor names are `validation_summary` keys; the `tune_` prefix is gone from history, checkpoints and reports.
- **No Lightning.** GPU time goes to extraction upstream, cached-feature epochs are CPU-bound, variable-size bags and the custom samplers need distributed-aware variants before DDP is correct, and Lightning's API churn adds surface. Lightning **Fabric** inside `Trainer.fit` (`fabric.setup`, `fabric.backward`) is the named multi-GPU escape hatch; a later swap to `pl.Trainer` is mechanical because `Learner` already speaks its dialect.
- **Conformance over private tests.** `soma.testing.check_learner` is the contract test; soma runs it on `MILLearner` for every aggregator and on `DenseLearner` for segmentation and detection. Tests that pinned the old loop's private symbols were deleted, not ported.

## Consequences

- `soma/training/trainer.py` is deleted; the three fold functions in `pipeline.py` run on the new Trainer through `_build_learner` and `_fit_fold`. Datasets, collates and samplers stay in `soma/training/` until the layout rename.
- `best_model.pt` keeps its name and the keys `model_state_dict`, `epoch`, `selection`; `metrics` replaces `tune_metrics` / `tune_loss` and `optimizer_state_dict` is dropped. The live segmentation path's file holds the decoder and head only.
- `training_history.json` holds one row per epoch (`epoch`, `train_loss`, `lr`, `elapsed_seconds`, then one column per summary key, `null` on an epoch without an evaluation). The reporting reader adapts; there is no converter for old run dirs.
- `TrainingConfig.monitor` defaults to `loss`; `optimizer` / `scheduler` accept a name or a leaf (`scheduler` also a list); `learner` is a new optional leaf. `lr_warmup_epochs` / `lr_warmup_factor` are removed and fail as unknown keys.
- `fp32` stays the default so reproduction rows are unchanged; `precision="bf16"` is opt-in autocast without a grad scaler.
