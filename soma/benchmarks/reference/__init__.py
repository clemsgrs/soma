"""Package marker so bundled reference tables ship as importable package data.

Each ``<name>.csv`` here is the single source of truth for a benchmark's expected
numbers + per-row tolerance (columns ``key…, metric, expected, tolerance, source``).
The same bytes feed the ``soma reproduce`` tolerance check, and later the leaderboard
reference rows and the docs table (ADR 0002).
"""
