# Contributing

Thanks for helping make embed-playground better. This is a focused,
deterministic retrieval playground: **offline-testable, zero-runtime-
dependencies, honest about what each tier can and cannot do**. Please keep
those properties.

## Ground rules

- **No runtime dependencies.** The lexical tier and RRF hybrid run on pure
  Python stdlib. The dense tier (sentence-transformers) stays an optional
  extra (`pip install "embed-playground[dense]"`), lazily imported, never
  required.
- **Deterministic output.** Same query + same tier → same ranking, same
  scores. No randomness, no timestamps, no set-ordering dependence.
- **Golden tests for every behavior change.** If a ranking, score, or eval
  metric changes, update the golden numbers in `tests/` deliberately.
- **Tests stay offline.** No network, no model downloads, ever.
- **Corpus and queries are original.** No scraped text — the bundled corpus
  and benchmark are self-written to stay leakage-free.

## Daily Green

The repo commits one dated entry per day via `scripts/daily_update.py`
(pool: `scripts/tips_pool.json`). Add retrieval tips to the pool; never
edit `docs/daily-tips.md` by hand.

## PR process

1. Fork, branch, change, test: `python -m pytest tests/ -q` (all green).
2. `ruff check embed_playground tests scripts` clean.
3. CLI smoke: `embed-playground eval --tier lexical`.
4. Reference the golden numbers you changed (and why) in the PR body.

## Style

- Type hints on all public functions; `py3.10+`.
- Negative results are welcome: if a tier misses a query, add it to the
  benchmark and document it — that is the point of the playground.