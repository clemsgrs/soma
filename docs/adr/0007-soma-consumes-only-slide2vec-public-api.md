# soma consumes only slide2vec's public API; extraction engines move upstream

soma must not import from `slide2vec.runtime.*` or other internal modules — where no public API covers a soma need, **slide2vec's public API is extended** rather than reached into. Concretely this deletes soma's own extraction engines (`tile_extraction.py`, `tile_extraction_spawn.py`, most of `dense_extraction.py`, and soma's dense write schema) in favour of public slide2vec entry points, leaving soma with dataset/labels, configuration, caching, training and evaluation.

## Why

soma's single hard break in slide2vec 5.4.0 came precisely from depending on a non-public function: `iter_regions_dense` (in `slide2vec.runtime.dense_regions`) was re-signatured without deprecation, because from slide2vec's side it is internal. Staying on it — and additionally mirroring its private `_build_dense_tiling_result`, which fabricates a 40-field hs2p `TilingResult` from placeholders — would not reduce drift; it schedules the next identical break and adds hs2p's field list as a second surface that can move underneath soma.

The duplication was larger than the dense call site. soma carried ~640 lines of feature-extraction engine for tile images, including a **second, independent implementation of multi-GPU sharding with atomic writes** — the same job `slide2vec/runtime/dense_shard.py` and `slide2vec/distributed/` already do, tested against nothing.

Persistence is not caching. slide2vec already writes the pooled artifacts soma's cache layer wraps; pointing `execution.output_dir` at soma's resolved cache directory keeps everything that actually constitutes caching in soma — the key, the completeness decision, `missing_sample_ids`, identity signatures, validation — while slide2vec writes the payloads. Nothing changes layers.

For in-process pooled slide encoding, soma calls the public `Model.embed_tiles` interface with its prepared slides and tiling results, then hands the returned artifacts unchanged to public `Model.aggregate_tiles`. slide2vec owns embedding progress and persistence in both the requested-output and temporary-artifact regimes; soma owns the outer cache gate and artifact lifetime.

## Upstream additions this requires

1. An explicit input contract at model load (ADR 0006).
2. That contract extended to the effective encoder input across pooled + dense (ADR 0006).
3. `embed_images` — given-geometry images on disk → embeddings, sharded across GPUs.
4. Dense over given images — the dense sibling of (3), replacing `soma/dense_extraction.py`.
5. A public home for the primitives soma's **live re-encode** path needs. Live re-encode is soma's own — it encodes inside the training loop, after augmentation, so no `embed_*` entry point covers it — but normalization, geometry validation, padding, device transfer, frozen encoding, windowing, attention, and precision must remain identical to extraction. slide2vec 5.7 provides that home through `Model.prepare_dense_encoder` and `DenseEncodeKit` (clemsgrs/slide2vec#267). soma now hands the kit augmented CPU RGB tensors and consumes its geometry and grids; the live path has no `slide2vec.runtime.*` imports (#322).

`embed_images` must apply the shipped transform **itemwise, then stack**: heterogeneously-sized inputs (BACH 2048×1536 alongside PCam 96²) cannot be stacked into one `(B, 3, H, W)` uint8 tensor before resizing, so slide2vec's batched transform spec is structurally inapplicable to the Given regime and stays exclusive to the uniform-size declared paths.

## Considered and rejected

- **Keep soma's dense writes and adapt slide2vec's artifacts into soma's schema.** Rejected: a translation layer keeps a second schema alive, and translation layers are where drift accumulates. slide2vec's sidecar already carries every field soma's does; `dense_input_mode` is derivable from `window_size is None`. soma's `DenseFeatureStore` reads slide2vec's layout natively instead.
- **Split the tile→vector API into a follow-up program.** Rejected in favour of one alignment: the Given regime has no home until it lands, and bundling it costs one re-verification pass rather than two.

## Consequences

- soma's ROI identity must stay explicit in the manifest rather than re-derived by string-splitting `<slide>__x<X>_y<Y>`; `SlideRegions.sample_id` carries the slide id and the ROI→(slide, x, y) mapping is recorded, not parsed.
- Dense extraction forwards `ExecutionConfig.num_gpus` to both public APIs. The boundary-sensitive gate in issue #305 compared 1 GPU with 2 GPUs for ragged image shards and a slide split across ROI shards in fp32 and fp16. All grids met slide2vec's cosine contract; see `docs/validation/dense-multigpu-parity-305.md`.
- The tile-path migration has no equivalent gate: `results/eva.csv` may be re-recorded within its 2 % relative band if numbers move.
- soma skipped slide2vec 5.4.0 entirely and aligned through 5.5/5.6. The dense source-spacing contract now requires slide2vec 5.7 and hs2p 4.4.1: soma declares `spacing_at_level_0` on source specs and consumes persisted `source_spacing_um` plus `effective_spacing_um` rather than reconstructing them from requested settings.
- With pooled embedding orchestration migrated in #288 and dense live encoding migrated in #322, soma's production extraction paths have no `slide2vec.runtime.*` imports.
