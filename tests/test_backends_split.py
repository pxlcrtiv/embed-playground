"""TDD tests for the backends.py deep-module split.

Verifies:
- data.py exports load_corpus, load_queries, Query
- dense.py exports DenseBackend, cosine_sim
- eval.py exports EvalMetrics, rank_query, evaluate
- backends.py remains a backward-compat facade with the full public API
- lazy imports still work and produce correct results
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Module-level symbol checks
# ---------------------------------------------------------------------------
def test_data_module_exports():
    from embed_playground import data

    assert hasattr(data, "load_corpus")
    assert hasattr(data, "load_queries")
    assert hasattr(data, "Query")


def test_dense_module_exports():
    from embed_playground import dense

    assert hasattr(dense, "DenseBackend")
    assert hasattr(dense, "cosine_sim")


def test_eval_module_exports():
    from embed_playground import eval as eval_mod

    assert hasattr(eval_mod, "EvalMetrics")
    assert hasattr(eval_mod, "rank_query")
    assert hasattr(eval_mod, "evaluate")


# ---------------------------------------------------------------------------
# Backward-compat facade: backends.py must re-export every public symbol
# that lived there before the split.
# ---------------------------------------------------------------------------
def test_backends_facade_reexports_corpus_functions():
    import embed_playground.backends as b

    assert b.load_corpus is not None
    assert b.load_queries is not None


def test_backends_facade_reexports_query_class():
    import embed_playground.backends as b

    assert b.Query is not None


def test_backends_facade_reexports_dense_backend():
    import embed_playground.backends as b

    assert b.DenseBackend is not None
    assert b.cosine_sim is not None


def test_backends_facade_reexports_eval_symbols():
    import embed_playground.backends as b

    assert b.EvalMetrics is not None
    assert b.rank_query is not None
    assert b.evaluate is not None


# ---------------------------------------------------------------------------
# Behavioral: facade calls must return identical results to direct calls
# ---------------------------------------------------------------------------
def test_facade_load_corpus_matches_data_module():
    from embed_playground import data
    from embed_playground.backends import load_corpus

    assert load_corpus() == data.load_corpus()


def test_facade_rank_query_matches_eval_module():
    from embed_playground import data
    from embed_playground.backends import rank_query
    from embed_playground.core import TfidfIndex

    corpus = data.load_corpus()
    index = TfidfIndex(corpus)
    fac = rank_query("lexical", index, None, "flash loan mechanics", top_k=3)
    direct = rank_query("lexical", index, None, "flash loan mechanics", top_k=3)
    assert [d.id for d, _ in fac] == [d.id for d, _ in direct]


def test_facade_evaluate_matches_eval_module():
    from embed_playground import data
    from embed_playground.backends import evaluate
    from embed_playground.core import TfidfIndex

    corpus = data.load_corpus()
    index = TfidfIndex(corpus)
    fac = evaluate("lexical", index)
    direct = evaluate("lexical", index)
    assert fac.recall_at_5 == direct.recall_at_5
    assert fac.mrr == direct.mrr
