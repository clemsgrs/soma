# Presets are named after what their head trains on, and check their encoder's level

Each preset is named after what its head trains on: one tile image's embedding (`tile_*`), an aggregator's pooling of a slide's tiles (`mil_*`), or a pretrained slide encoder's embedding (`slide_*`). `segmentation` and `detection` decode each tile's token grid. A preset checks the level of its encoder when it is built, and refuses an encoder of another level by naming the preset to use.

| The head trains on | Encoder level | Classification | Regression | Survival |
|---|---|---|---|---|
| one tile image's embedding | tile | `tile_classification` | `tile_regression` | none |
| an aggregator's pooling of a slide's tiles | tile | `mil_classification` | `mil_regression` | `mil_survival` |
| a pretrained slide encoder's embedding | slide | `slide_classification` | `slide_regression` | `slide_survival` |

## Why

No preset trained on a slide encoder's embedding. A slide encoder without an aggregator makes each sample a slide, so `tile_classification("titan")` passed `check` and ran as a slide run, under a name that says tiles, at the tile probe's batch of 256 and with hs2p's default 1% tissue mask. There was no tile regression preset either: the HEST benchmark built from `tile_classification` and swapped in its regression head.

## Decisions

- **The name says what the head trains on.** The prefix picks the path and the suffix the task. `tile_regression` is `tile_classification`'s recipe with the `regression` head. The `slide_*` presets keep the `mil_*` presets' 10% tissue mask, so a slide encoder and a tile encoder with an aggregator see the same tissue, and leave the tiling unset, so the run takes the encoder's own: a slide encoder is tied to its tile encoder's tile size and spacing, so they take no `spacing` or `tile_size`. They train the linear head with no aggregator, in batches of 32 slides, with AdamW at lr 3e-4 and no schedule.
- **Cox on single embeddings uses padded mode.** `slide_survival(encoder, "cox")` sets no `cox_window`, so the risk set is the batch of 32 slides; `check` refuses `cox_window >= 2` without an aggregator. Both survival paths train on a risk set of 32.
- **The level check runs when the preset is built**, before any slide is tiled, with the resolver `check` uses (`soma.plan.encoder_level`). A slide encoder in a `tile_*` or `mil_*` preset is sent to the matching `slide_*` preset; a tile encoder in a `slide_*` preset is sent to the `mil_*` preset (pool its tiles with an aggregator) or the `tile_*` preset (tile images); a non-tile encoder in `segmentation` or `detection` is refused because they decode tile grids.
- **No preset covers patient-level encoders yet.** Every preset refuses one and says so. A patient-level run is written as a config with `cohort.collapse: patient_id`.
- **The presets list in table order.** `soma.components("presets")`, `soma list presets` and the CLI reference list them by row of the table above, then `segmentation` and `detection`; every other component kind still lists by name.

## Considered and rejected

- **One preset per task that picks its path from the encoder level**, such as `classification(encoder, aggregator=None)`. Leaving out `--aggregator` with `uni2` on a slide cohort would quietly become a tile run: a silent model choice of the kind that explicit preset arguments removed. The arguments also differ by path (`aggregator`, `spacing` and `tile_size` do not apply to a slide encoder), and so do the recipes: Cox needs padded mode on single embeddings and a window over MIL bags.

## Consequences

- `tile_classification("titan")` and `mil_classification("titan", "abmil")` now raise when built. A config written by hand still meets the same rules in `check`.
- The HEST benchmark builds from `tile_regression`. Its configs are unchanged: it overwrites the learner, loaders and trainer the two presets differ in.
- `soma.testing` registers `fake-slide`, a weight-free slide encoder (the mean of its tiles' `fake` embeddings), so the slide-encoder path runs end to end in the test suite.
