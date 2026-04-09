# Documentation Notes

## Active Design Decisions

### Tiling cache

The tiling cache is intentionally narrow and separate from the existing feature
cache.

The shared tiling cache will be the canonical storage location for cached
tiling artifacts. Run-local tiling directories will be lightweight stubs that
contain a `README.txt` plus a `process_list.csv` pointing at the shared cache
paths.

For `backend="auto"`, cache reuse should validate against the current runtime's
actual resolved backend by probing `hs2p.wsi.resolve_backend(...)` per sample
before accepting a cache hit.
