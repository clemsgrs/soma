# The config mirrors the API; a run resolves it to a plan

An experiment is an `ExperimentConfig` of seven sections, each holding one public callable's keyword arguments: `run` is `Pipeline`'s, `cohort` the cohort's, `features` the feature source's (ADR 0019), `loaders` those of `soma.learn.loaders`, `learner` the `Learner` and its model, `trainer` the `Trainer`'s and `evaluation` those of `soma.evaluate`. A YAML file is the same dataclasses, hydrated generically: a mapping fills a dataclass, `{name, params}` is a `Component` leaf, a list fills a tuple. `soma.resolve(config, cohort)` turns a config into a `Plan`: the config with every component's default params spelled out and what the cohort decides filled in. `PipelineConfig`, `load_config`, `save_config`, `config_yaml_dict`, `TaskConfig`, `TrainingConfig`, `EvalConfig` and the YAML translation layer are deleted without aliases.

## Why

The YAML vocabulary (`data`, `task`, `training`, `aggregation`, `preprocessing`) had drifted from the Python API it configured: a knob could exist in YAML and not in Python or the reverse, `dataset_type` decided the code path although the model already did, and a translation module had to stay exact in both directions. A saved `config.yaml` held soma's defaults merged in at load time, so it could not tell what a user asked for from what soma chose, and nothing recorded what the cohort decided.

## Decisions

- **One section per callable.** A knob that cannot be passed in Python cannot appear in YAML, and every section is that callable's keywords with its defaults. The model's components (`aggregator` or `decoder`, `head`), the `optimizer` and `scheduler` and the feature adaptor are the learner's params; the checkpoint and early-stopping policy are `trainer.callbacks`; the training-batch sampler and the augmentation are `loaders` leaves; `tune_is_test` and `allow_missing_tune` are `cohort` options.
- **The model decides the path.** `dataset_type` is gone. The shape is the learner's (`dense` is the grid shape, any other learner the set shape), the task family is the head's, and a set run's sample unit is derived: `cohort.collapse: patient_id` makes patients, a tile encoder without an aggregator makes tiles, anything else is a slide.
- **`resolve` merges what the cohort decides under what the config states.** Each head's `infer` hook (ADR 0018) proposes values from the cohort; a stated value wins, and a contradiction (a stated class count the cohort does not match) raises. Feature dimensions are not in the plan; they come from the source at build time and are recorded in the bundle.
- **The run directory keeps both.** `config.yaml` is the config as given and `plan.yaml` the plan. The experiment identity digests the plan, without the `run` section. A resume compares plans, without where and how the launch runs (`run.output_dir`, `run.run_id`, `run.resume`, `run.mirror_root`, `run.folds`) and the execution resources slide2vec resolves for the machine; `run.seed` is compared. The manifest drift guard is unchanged.
- **Dotted paths stay as written.** A leaf named by import path keeps that path in the plan, with every param it was given; the bundle's `provenance.json` records the installed version of each top-level package such a path imports from. Versions are not part of the plan.
- **A head's leaf is its constructor keywords.** Head params are validated like every other component's. The survival `loss` selector is gone: the `cox_survival` head is the Cox objective.

## Considered and rejected

- **Keep a defaults YAML merged at load.** It is a second source of defaults that drifts from the dataclasses; the plan spells the defaults out instead, from the code.
- **Store only the plan.** A user's intent (what was left to the defaults or the cohort) would be lost; a replay from `config.yaml` re-resolves against the current code and cohort, a replay from `plan.yaml` pins every value.
- **Compare `run.folds` on resume.** Launches pinned to one run id split its folds between them; each launch takes on other folds of the same run, so the fold selection is launch-scoped like the run id.

## Consequences

- Every YAML file and every benchmark's config builder is rewritten; there is no converter from the old layout. Experiment identities change (identity version 9); feature cache keys do not.
- The plan includes the torch optimizer's and scheduler's defaults, read from their signatures, so a torch upgrade that changes a default changes the plan and the experiment id.
- `soma.ExperimentConfig`, `soma.resolve` and `soma.Plan` are the public surface; the sections live in `soma.config`.
- A config states only what differs from the defaults, and `soma plan.yaml` reruns a run with every value pinned.
