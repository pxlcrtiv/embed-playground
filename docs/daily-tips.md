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


## 2026-08-26 — Retrieval tip of the day: MRR gaming is real

Short queries with exactly one obvious doc inflate MRR. Mix multi-relevant queries (like this repo's cross-context items) or the metric flatters your demo.


## 2026-08-27 — Retrieval tip of the day: Fallbacks keep CI honest

A dense tier that needs a 90MB download cannot run in every CI. Keep the lexical tier as the offline golden path and treat dense as an optional accelerator.


## 2026-08-28 — Retrieval tip of the day: Watch the long tail of the corpus

Embedding quality concentrates on frequent concepts. Measure recall per category; the category with the worst recall is your data problem, not your model problem.

> `embed-playground eval --tier lexical --detail`


## 2026-08-29 — Retrieval tip of the day: Rerankers change the game

Cross-encoder rerankers on the top-50 retire most hybrid tuning fights. First retrieval breadth (recall@50), then rerank precision — never the reverse.


## 2026-08-30 — Retrieval tip of the day: Snippet quality is a ranking signal

A perfect hit with an unreadable snippet loses the user. Keep extraction clean: title, category, and a 1-2 sentence window, like this playground's table.


## 2026-08-31 — Retrieval tip of the day: Evaluation belongs in the repo

A benchmark script next to the code means every PR can regress retrieval. Golden numbers in tests turn silent ranking drift into a failed build.

> `pytest tests/ -q`


## 2026-09-01 — Retrieval tip of the day: Document your negative results

'Dense missed q03, lexical missed q02' is the most valuable paragraph in your README. It teaches users where each tier fails — that is how tools get trusted.


## 2026-09-02 — Retrieval tip of the day: Query rewriting is preprocessing

Users type fragments ('flash loan how'), models expect sentences. Cheap normalization — case, stopwords, expansion of common abbreviations — buys consistent gains.


## 2026-09-03 — Retrieval tip of the day: Latency is a benchmark metric

Dense search on 36 docs is instant; on 36 million it is an index problem. Record time-per-query in every eval run before scaling claims get made.


## 2026-09-04 — Retrieval tip of the day: TF-IDF is the baseline that never blinks

Before reaching for embeddings, score your problem with TF-IDF cosine. It is free, deterministic, and often embarrassingly competitive on keyword-shaped queries.

> `embed-playground eval --tier lexical`


## 2026-09-05 — Retrieval tip of the day: Recall@k is the question, MRR is the ranking

Recall@5 asks 'did the answer appear?', MRR asks 'how high?'. RAG needs both: a 100% recall with rank-50 retrieval is still a failed pipeline.

> `embed-playground eval --tier dense --detail`


## 2026-09-06 — Retrieval tip of the day: Paraphrase queries separate lexical from semantic

If every query shares words with its answer doc, lexical wins and you learn nothing. Write paraphrase queries ('bots steal my trades') to see what dense retrieval actually buys.

> `embed-playground compare how do bots steal money from my trades`


## 2026-09-07 — Retrieval tip of the day: Embeddings are not magic, they are priors

A sentence-transformer maps text to a learned prior space. The prior reflects its training data; domain jargon ('flash loan', 'MEV') is where generic models guess wrong.


## 2026-09-08 — Retrieval tip of the day: Hybrid is a fusion, not a feature

RRF fusion combines rankings, and it is only as good as its weakest tier. Benchmark each tier alone before fusing — a broken lexical tier quietly drags the hybrid down.

> `embed-playground eval --tier hybrid`


## 2026-09-09 — Retrieval tip of the day: Pin your embedding model version

A newer MiniLM checkpoint can silently re-rank your entire corpus. Record the model id and hash with the benchmark results or your leaderboard is untrustworthy.

