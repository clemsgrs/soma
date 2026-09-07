# PRISM reference provenance

The slide and tile embeddings in `gt/` come unchanged from
[slide2vec commit e1699ecc](https://github.com/clemsgrs/slide2vec/commit/e1699ecc43805d8d427c2ef7c8a73acc66fe9dc3),
`tests/fixtures/gt/test-wsi.pt` and `test-wsi.tiles.pt`.

That commit refreshed the upstream references after making WSI reads area-resize
the raw 444-pixel tile to the requested 224 pixels before encoder transforms.
The previous soma copies predated that change. They produce a slide cosine of
about 0.9872 against the current pipeline, below the existing 0.99 threshold.

Both input TIFFs are byte-identical to upstream. The 459 `x`, `y`, and `tile_index`
values also match; the coordinate files retain their original metadata schema.
Upstream generates the reference with ASAP, PRISM/Virchow at 0.5 µm/px, and FP16
precision; soma's integration test uses OpenSlide and the same numerical tolerance.
The upstream recipe is recorded in `tests/output_consistency_config.py` at the
linked commit.

Refresh from an independently validated upstream reference when its documented
input contract changes. Do not regenerate these files from soma merely to make
its regression pass, or relax the numerical threshold to hide a mismatch.
