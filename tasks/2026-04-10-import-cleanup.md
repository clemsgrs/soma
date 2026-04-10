# Import Cleanup Plan

- [x] Remove core-dependency import fallbacks from `soma/features.py`, `soma/cache.py`, and `soma/__init__.py`.
- [x] Move function-local core imports to module scope where practical.
- [x] Keep the touched modules behaving the same under the existing tests.
- [x] Add a short documentation note describing the cleanup.
