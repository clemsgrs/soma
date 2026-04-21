# Slide Cache Refresh Output Variant

- [x] Reproduce the distributed slide-cache refresh key drift on slide-level runs.
- [x] Keep the runtime output variant separate from the cache identity output variant in the distributed helper.
- [x] Add a regression test that proves every slide-cache resolve in the distributed path uses the stable cache output variant.
- [x] Run the targeted extraction/cache tests and confirm the duplicate slide-cache miss is gone.
- [x] Update the cache notes in `docs/documentation.md`.
