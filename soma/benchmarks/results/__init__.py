"""Package marker so reproduced-results tables ship as importable package data.

Each ``<name>.csv`` here records soma's **own** produced numbers for a benchmark — the
counterpart to ``soma/benchmarks/reference/<name>.csv`` (the external target). A results
row shares the reference table's key columns (everything left of ``metric``) so the two
join, then adds the measurement (``measured``/``std``/``n_seeds``) and the provenance that
makes a reproduced number meaningful (``date``, ``soma_commit``, ``slide2vec_version``).

The table is an **append-only ledger**: re-running a cell at a new commit adds a row rather
than overwriting history, so drift across code/extractor versions stays visible. Rows are
written by ``soma reproduce --record`` and read by :func:`soma.benchmarks.load_results`.
"""
