# Extraction progress is a snapshot hook beside slide2vec's reporters

`soma.extract(..., on_progress=fn)` and `Pipeline(config, on_extraction_progress=fn)` let a caller follow an extraction. `fn` takes one `soma.extract.ExtractionProgress`: the encoder, the phase (`checking_cache`, `tiling`, `loading_model`, `embedding`, `aggregating`, `finished`), and counters of samples and tiles. Each call holds the whole state of the extraction. One module, `soma.extract.progress`, folds slide2vec's and hs2p's progress events and soma's own into that state.

## Why

A caller that runs `soma.extract` away from a terminal (a batch job, a user interface that runs soma) could follow it only by activating a slide2vec reporter around the call. That reporter is not soma's API. Off a terminal soma had no reporter at all, and the events differ by mode: `tiling.*` and `embedding.*` on whole slides, `images.*` on pre-cropped images, `dense.*` on grids, and soma's own `embedding.*` for slide aggregation on several GPUs. Their totals are per slide2vec call, so a partial resume or a full cache hit reports something other than the cohort.

## Decisions

- **A snapshot, not event kinds.** Each call gets the whole current state, so the consumer keeps no state and needs no knowledge of the modes. The same hook covers tile encoders, hierarchical regions, slide encoders, pre-cropped images and dense grids.
- **A sample is one row of the image manifest**, in every mode, also for dense grids and for ROIs sampled from slides (each ROI counts as one tile). `samples_total` is the manifest. `samples_cached` is the rows whose features were in the final cache at the start (the slide cache for a slide encoder), `None` until the cache is checked.
- **Counters are per phase.** `samples_done` counts the pending rows that have completed the current phase and starts again from 0 in each phase. A slide encoder is two passes over the cohort, `embedding` then `aggregating`, so each pass has its own count. Tiles are counted only during `embedding`.
- **Phases only move forward.** A phase a mode does not have is absent. With several GPUs, `loading_model` lasts until the first worker encodes. A full cache hit is `checking_cache`, then `finished`. There is no failure phase: a failed extraction raises.
- **Additive beside slide2vec's reporters.** With a hook, an internal reporter folds every event and forwards it unchanged to the reporter that would have been active: the Rich display on a terminal, a reporter the caller activated, or none. soma's own events (the cohort, the cache check, the aggregation pass, the end) go to the fold only. Without a hook, nothing is installed.
- **Delivery is bounded and safe.** A counter-only change is delivered at most every 0.5 s, a fixed constant; the first snapshot of each phase and `finished` always are. Nothing is sent when nothing changed. The hook runs in the calling process, one call at a time. An `Exception` from it is logged once with its traceback and the hook keeps being called; `KeyboardInterrupt` and `SystemExit` pass through.
- **The pipeline passes the hook through, nothing more.** Like `callback_factory`, `on_extraction_progress` is not part of the config, the plan or the experiment identity. Extraction runs in the calling process, also with `devices`, so the hook is never pickled.

## Considered and rejected

- **Event kinds for the consumer** (`tile_done`, `slide_started`, ...). Every consumer would rebuild the state, and learn each mode's stream to do it.
- **Cohort-wide counters that run over the whole extraction.** A slide encoder's two passes would share one count, so a display could not tell how far each pass is.
- **A JSONL writer or a config field.** The consumer is a display, not a log; a config field would make a display part of the experiment.
- **Checking the cache before tiling, so `samples_cached` is known during `tiling`.** On whole slides the feature cache's key needs the tilings (their backend provenance and geometry), and sampled ROIs need sampling before their ids exist. Moving the check, or peeking at the cache before it, would duplicate cache logic. On those paths `samples_cached` stays `None` while the slides are tiled.

## Consequences

- On a terminal with a hook, one Rich display spans the whole `soma.extract` call, instead of one per slide2vec call on the image and grid paths.
- With a hook, a slide encoder on one GPU aggregates one slide per slide2vec call, so the hook can count slides; without one, the calls are unchanged.
- Some counts are coarser than others: on one GPU, slide2vec reports no batches of dense regions, so ROIs advance once per call; a slide encoder on several GPUs without a tile cache runs one pass, reported as `embedding`.
- A reporter the caller activates keeps working and stays undocumented; `on_progress` is the supported way.
