# embed-playground tips of the day

> Maintained by `scripts/daily_update.py` (Daily Green automation) — one
> dated, non-empty retrieval/embeddings tip per day, rotated from the
> pool in `scripts/tips_pool.json`. Pause by creating a `.daily-pause`
> file in the repo root, or unload the scheduler job (see README,
> Daily Green).


## 2026-08-24 — Retrieval tip of the day: The corpus is the interface

Users never see your index, they see your docs. Clean, concrete, one-topic-per-doc writing improves every retrieval metric more than any model change.


## 2026-08-25 — Retrieval tip of the day: Determinism is a debugging tool

Random ordering or hash-order iteration makes ranking bugs nondeterministic. Sort by score, then id, and golden-test the top of the list.

> `embed-playground search flash loan mechanics --top-k 3`

