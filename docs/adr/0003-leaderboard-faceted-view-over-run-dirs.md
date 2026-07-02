# The Leaderboard is a faceted view over self-describing run dirs

The **Leaderboard** is a *rendered projection*, not a stored file, and its source of truth is the **self-describing run dirs**: every run already writes its own `config.yaml` + `summary.json` + `run.yaml`, and `experiment_id` is a sha256 fingerprint of the *entire* config modulo seed (changing an ABMIL `hidden_dim`, a learning rate, or a spacing all yield a new experiment). `indexes/experiments.csv` is demoted to a **rebuildable cache** — never authoritative, so a corrupt or racy index is a non-event, rebuilt by rescan.

A Leaderboard is a **facet**: a fixed `(dataset + splits + task)` plus a set of config axes *held fixed* and one axis *varied and ranked by*. So a cross-encoder leaderboard (fix the downstream recipe, vary `encoder`) and an encoder-specific leaderboard (fix `encoder`, vary the rest) are two facets of **one flat registry** — not separate tables.

## Considered and rejected

- **A central mutable index (the current CSV, or SQLite).** The CSV's `update_experiment_index` is an unlocked read-modify-**rewrite** of the whole file; two runs finishing concurrently silently lose a row — and a *sweep*, the leaderboard's primary feeder, is inherently concurrent. Making each run touch only its own dir *dissolves* the race instead of *managing* it with locks/WAL. SQLite is the right migration at 10⁵–10⁶ runs; at the 10²–10³ this domain sees it is more machinery than the problem warrants, and the run dirs stay the source of truth either way, so the migration remains open.
- **A pre-keyed leaderboard table (ranked by encoder).** Rejected: it forecloses the encoder-specific facet. A flat, fully-described registry is precisely what makes arbitrary faceting possible.

## Consequences

- The racy `update_experiment_index` write path is replaced/demoted; `soma leaderboard` scans-and-projects (and may cache the projection — the cache is disposable).
- Reproduction plugs in as a **reference row**: a Benchmark's `expected()` values render as pinned rows (a broad config-agnostic banner, or per-config aligned rows) beside the measured rows, and the tolerance check compares the two.
