# Dead Code Cleanup Plan

- [x] Remove the `sklearn` fallback implementations from `soma/evaluation/metrics.py`.
- [x] Keep the public metric functions working through the normal dependency path.
- [x] Run the focused evaluation test module to verify the cleanup.
- [x] Add a short documentation note describing the cleanup.
