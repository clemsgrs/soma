# Dense multi-GPU parity — issue #305

## Result

Both public slide2vec dense APIs passed the 1-GPU versus 2-GPU boundary gate. soma can
forward `ExecutionConfig.num_gpus` to `Model.embed_images_dense` and
`Model.embed_regions_dense` without a local numerical workaround.

The gate ran on 2026-08-11 with slide2vec 5.7.0 at commit
`7cfa6bf5adade8530fbad9aa70dd45f698c8deef`, PyTorch 2.7.1+cu128, CUDA 12.8, and two
NVIDIA GeForce RTX 2080 Ti GPUs. The fixed encoder was `lunit`, with 0.5 µm/px spacing,
224 px regions, whole-image dense encoding, no overlap, and batch size 2.

| Public API | Precision | Minimum cosine | Maximum `1 - cosine` | Maximum absolute delta | Mean absolute delta |
|---|---:|---:|---:|---:|---:|
| `embed_images_dense` | fp32 | 1.0000034571 | -3.4571e-06 | 2.6226e-05 | 9.1536e-07 |
| `embed_images_dense` | fp16 | 0.9999918938 | 8.1062e-06 | 3.1250e-02 | 1.1157e-03 |
| `embed_regions_dense` | fp32 | 1.0000030994 | -3.0994e-06 | 5.2929e-05 | 7.2057e-07 |
| `embed_regions_dense` | fp16 | 0.9999924898 | 7.5102e-06 | 3.9062e-02 | 8.6130e-04 |

Cosine accumulation can round slightly above 1, which explains the negative fp32
`1 - cosine` values. Every grid was finite and every comparison met cosine ≥ 0.9999.

## Boundary construction and assertions

- The image case used five ordered images. Contiguous 2-GPU shards contained 3 + 2
  images at batch size 2, giving rank 0 a partial tail.
- The region case flattened three slides with 3 + 3 + 1 ROIs. Contiguous 2-GPU shards
  contained 4 + 3 ROIs, so `slide-b` crossed the rank boundary after its first ROI. Its
  one-GPU batches were 2 + 1; its 2-GPU batches were 1 on rank 0 and 2 on rank 1.
- Image results were matched in caller order by `sample_id`. Region results were matched
  in caller order by `(slide_id, x, y)`.
- Both runs asserted identical semantic membership and order, tensor shape, tensor dtype,
  and persisted metadata. They also asserted finite values and cosine ≥ 0.9999 per grid.

Re-run the gate with `scripts/verify_dense_multigpu_parity.py`, a spacing-readable test
slide, and an output directory on local storage. The script writes the full payloads,
worker logs, generated inputs, and machine-readable report beneath that directory.

## Cache-key implication

GPU count remains outside soma's dense cache key. Rank-dependent batch composition can
change artifact bytes, as the fp32 and fp16 image deltas demonstrate, but the grids remain
semantically equivalent within slide2vec's public tolerance contract. A resumed cache may
therefore contain tolerance-equivalent grids produced by different rank batch shapes; it
must not be interpreted as byte-identical across GPU counts.
