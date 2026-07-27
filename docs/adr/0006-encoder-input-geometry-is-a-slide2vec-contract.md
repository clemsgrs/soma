# Encoder-input geometry is a slide2vec contract with two regimes

Whether an image is resized before it reaches a frozen encoder — and whether an off-preset size is an error, a resize, or an exact request — is **feature-extraction policy, so it lives in slide2vec**, not in soma. slide2vec exposes one contract with two regimes: **Declared geometry**, where the caller states the encoder input it wants (pooled `requested_tile_size_px`, dense `spacing_um` + `target_size`) and slide2vec must honor it exactly or raise; and **Given geometry**, where the caller supplies pixels it never requested (pre-cropped tile datasets) and the encoder's shipped transform is the contract. soma writes no geometry policy of its own — it only names which regime its input is in.

## Why two regimes is one policy, not two

The rule is *soma never silently substitutes a geometry it did not declare*. It lands differently in the two paths only because in one there is a declaration to honor and in the other there is nothing to honor:

- Pooled slide extraction declares physical extent — "224 px covering 112 µm of tissue". If slide2vec quietly feeds the encoder something else, the declaration was a lie, so an unsatisfiable declaration must raise.
- A tile dataset (`dataset_type="tile"`) declares nothing: the row is a path to a PNG whose dimensions belong to the upstream dataset (BACH 2048×1536, PCam 96²), frequently not square, varying per sample. There is no size soma could demand instead, so refusing the image gives the user nothing. Resizing to the encoder's input **is** the published protocol these datasets are evaluated under.

Applying the Declared rule to Given inputs would have made 4 of the 6 EVA datasets unrunnable and invalidated the `results/eva.csv` ledger.

## Scope: effective encoder input, not "tile size"

The contract is about the **Effective encoder input** — the geometry of the tensor immediately before `encode_tiles` / `encode_tiles_dense` — so it covers pooled *and* dense with one capability check:

| path | effective encoder input |
|---|---|
| pooled, declared | `requested_tile_size_px` |
| dense, whole-tile | padded `encoded_size` (target padded to the patch multiple) |
| dense, sliding | `window_size` (native, so the check passes trivially) |
| given | not declared; whatever the shipped transform produces, recorded after the fact |

One shared capability check (`supports_variable_input_size`) and one shared application of `variable_input_model_kwargs` then serve both paths.

## Considered and rejected

- **Resize-with-a-warning for fixed-input encoders on off-preset declared sizes.** Rejected: a fixed encoder fed a resized tile is a silent quality loss, and slide2vec 5.4.0 raises. soma inherits the behavior rather than building a third, more permissive policy.
- **Applying the Declared rule to `dataset_type="tile"`.** Rejected — breaks EVA (above).
- **Leaving the regime implicit** (slide2vec's `pooled_input_plan=None` meaning both "given" and "caller forgot"). Rejected: that overload is exactly how soma acquired a silent 1-GPU-vs-2-GPU feature divergence — `Pipeline` built the plan, the in-process `_embed_tiles` route did not, and no test failed. The contract is explicit and non-defaultable at the seam so omission is an error, not a fallback.
- **Passing `dynamic_img_size` by hand from soma** (status quo). Rejected: it bypasses the capability check entirely — phikon's constructor does not even accept the kwarg, so soma's `dynamic_img_size=True` is silently swallowed. Under the contract it is derived from a declaration plus registry metadata.

## Consequences

- slide2vec gains an explicit input contract at model load and extends it across pooled + dense; soma stops passing `dynamic_img_size`.
- Because the contract makes dense capability explicit, `Model.embed_regions_dense` becomes usable by soma (see ADR 0007) — the missing `dynamic_img_size` knob was the only blocker.
- A soma config with an off-preset declared size now behaves identically regardless of GPU count.
