"""Retrieval evaluation: rank_query and EvalMetrics.

Single concern: ranking docs for a query across tiers and aggregating
Recall@5 / MRR benchmark results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .core import Doc, TfidfIndex, rank_by_score, rrf_score
from .data import load_queries
from .dense import DenseBackend


@dataclass
class EvalMetrics:
    tier: str
    queries: int
    recall_at_5: float
    mrr: float
    per_query: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "tier": self.tier,
            "queries": self.queries,
            "recall@5": round(self.recall_at_5, 4),
            "mrr": round(self.mrr, 4),
        }


def rank_query(
    tier: str,
    index: TfidfIndex,
    dense: DenseBackend | None,
    query: str,
    top_k: int = 50,
) -> list[tuple[Doc, float]]:
    """Rank docs for one query with the requested tier.

    - lexical: TF-IDF cosine
    - dense:   sentence-transformers cosine (requires the backend)
    - hybrid:  RRF fusion of lexical + dense; falls back to lexical alone
               when the dense tier is unavailable (deterministic).
    """
    if tier == "lexical":
        return index.search(query, top_k=top_k)
    if tier == "dense":
        if dense is None:
            raise RuntimeError("dense tier not available (backend not provided)")
        return dense.search(query, index.docs, top_k=top_k)
    if tier == "hybrid":
        lexical = [d.id for d, _ in index.search(query, top_k=50)]
        if dense is not None:
            dense_ranks = [d.id for d, _ in dense.search(query, index.docs, top_k=50)]
            fused = rrf_score([lexical, dense_ranks])
        else:
            fused = rrf_score([lexical])
        return rank_by_score(fused, {d.id: d for d in index.docs})[:top_k]
    raise ValueError(f"unknown tier: {tier}")


def evaluate(
    tier: str, index: TfidfIndex, dense: DenseBackend | None = None
) -> EvalMetrics:
    """Recall@5 and MRR over the bundled benchmark for one tier."""
    queries = load_queries()
    per: list[dict] = []
    tp = 0.0
    rr_sum = 0.0
    for q in queries:
        ranked = rank_query(tier, index, dense, q.query, top_k=50)
        ranked_ids = [d.id for d, _ in ranked]
        hits = [i for i, doc_id in enumerate(ranked_ids) if doc_id in q.relevant and i < 5]
        hit = 1 if hits else 0
        tp += hit
        rr = 1.0 / (hits[0] + 1) if hits else 0.0
        rr_sum += rr
        per.append({"id": q.id, "intent": q.intent, "hit@5": hit, "rr": round(rr, 4)})
    n = len(queries)
    return EvalMetrics(
        tier=tier,
        queries=n,
        recall_at_5=tp / n,
        mrr=rr_sum / n,
        per_query=per,
    )
