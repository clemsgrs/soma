# Benchmarking subsumes published-benchmark reproduction

soma treats **"benchmark foundation models on any dataset"** as the product, and **"reproduce a published benchmark"** as a *special case* of it: the same flow (curate → configure → run → leaderboard) pinned with a canonical protocol, recorded expected numbers, and a tolerance band.

Chosen over building a standalone "reproduction suite" for a fixed catalog of public benchmarks, because the general experimentation engine is soma's north star. A **Benchmark** is then just general benchmarking plus expected-numbers annotations, and both worlds — your runs on your data, and a reproduction run next to its published target — share **one Leaderboard artifact** rather than two parallel mechanisms.

## Consequences

- The central object is a *benchmarking sweep over a dataset that yields a ranked leaderboard*, not a benchmark catalog.
- OCELOT / EVA become *validation instances* of the engine (see [0002](0002-benchmarks-are-first-class-registered.md)), not bespoke one-offs.
