"""backends.py — backward-compatible facade.

All public names that lived here before the deep-module split are
re-exported from their new homes so existing imports keep working:

    from embed_playground.backends import DenseBackend, cosine_sim
    from embed_playground.backends import Query, load_corpus, load_queries
    from embed_playground.backends import EvalMetrics, rank_query, evaluate

Source of truth moved to:
    data.py   — corpus/queries/Query
    dense.py  — DenseBackend + cosine_sim
    eval.py   — EvalMetrics + rank_query + evaluate
"""

from __future__ import annotations

# ---- data seam ----
from .data import Query, load_corpus, load_queries

# ---- dense seam ----
from .dense import DenseBackend, cosine_sim

# ---- eval seam ----
from .eval import EvalMetrics, evaluate, rank_query

__all__ = [
    "DenseBackend",
    "EvalMetrics",
    "Query",
    "cosine_sim",
    "evaluate",
    "load_corpus",
    "load_queries",
    "rank_query",
]
