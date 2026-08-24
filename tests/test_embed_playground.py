"""Offline, deterministic tests for embed-playground. No network, no models."""

from __future__ import annotations

import json

import pytest

from embed_playground.backends import evaluate, load_corpus, load_queries, rank_query
from embed_playground.core import TfidfIndex, rank_by_score, rrf_score, tokenize

CORPUS = load_corpus()
INDEX = TfidfIndex(CORPUS)  # shared deterministic index
QUERIES = load_queries()


# ---------------------------------------------------------------- tokenizer
def test_tokenize_basic():
    assert tokenize("Flash loans! ERC-20, don't.") == ["flash", "loans", "erc-20", "don't"]


def test_tokenize_strips_stopwords():
    assert tokenize("this is the way to do it") == ["this", "way", "do"]


def test_tokenize_empty():
    assert tokenize("!!! ...") == []


# ---------------------------------------------------------------- corpus
def test_corpus_schema():
    assert len(CORPUS) == 36
    ids = [d.id for d in CORPUS]
    assert len(set(ids)) == len(ids)
    cats = sorted({d.category for d in CORPUS})
    assert cats == ["defi", "layer2", "mev", "nft", "security", "wallets"]
    assert all(d.title and d.text for d in CORPUS)


def test_corpus_stats_golden():
    assert INDEX.stats() == {"docs": 36, "terms": 972}


def test_queries_schema():
    assert len(QUERIES) == 16
    known = {d.id for d in CORPUS}
    ids = [q.id for q in QUERIES]
    assert len(set(ids)) == len(ids)
    for q in QUERIES:
        assert all(r in known for r in q.relevant)
        assert q.intent in {"definitional", "paraphrase", "cross-context"}


# ---------------------------------------------------------------- lexical tier
def test_index_deterministic_across_instances():
    index2 = TfidfIndex(CORPUS)
    a = INDEX.search("flash loan mechanics", top_k=3)
    b = index2.search("flash loan mechanics", top_k=3)
    assert [(d.id, round(s, 12)) for d, s in a] == [(d.id, round(s, 12)) for d, s in b]


def test_lexical_search_flash_loan_golden():
    top = INDEX.search("flash loan mechanics", top_k=3)
    assert [d.id for d, _ in top] == ["d02", "d15", "d01"]
    assert top[0][1] == pytest.approx(0.336, abs=1e-3)


def test_lexical_search_deterministic_rerun():
    assert [d.id for d, _ in INDEX.search("flash loan mechanics")] == [
        d.id for d, _ in INDEX.search("flash loan mechanics")
    ]


def test_lexical_multisig_query():
    top = INDEX.search("startup treasury wallet with multiple signers", top_k=3)
    assert top[0][0].id == "d21"  # multisig doc on top


def test_rare_term_outranks_common_term():
    # 'flash' appears in few docs -> high idf; 'token' appears everywhere.
    common = INDEX.search("token token token token", top_k=1)
    rare = INDEX.search("flash flash flash flash", top_k=1)
    assert rare[0][1] > common[0][1]


def test_empty_query_returns_empty():
    assert INDEX.search("!!! ...") == []


# ---------------------------------------------------------------- RRF hybrid
def test_rrf_scores_golden():
    fused = rrf_score([["a", "b", "c"], ["b", "a", "c"]])
    k = 60
    assert fused["a"] == pytest.approx(1 / (k + 1) + 1 / (k + 2), abs=1e-9)
    assert fused["b"] == pytest.approx(1 / (k + 2) + 1 / (k + 1), abs=1e-9)
    assert fused["c"] == pytest.approx(2 / (k + 3), abs=1e-9)


def test_rrf_single_list_is_identity_ranking():
    fused = rrf_score([["d02", "d01", "d03"]])
    ranked = [doc_id for doc_id, _ in sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))]
    assert ranked == ["d02", "d01", "d03"]


def test_rank_by_score_ties_break_by_doc_id():
    docs = {d.id: d for d in CORPUS}
    # synthetic equal scores
    scores = {d.id: 0.5 for d in CORPUS[:5]}
    ranked = rank_by_score(scores, docs)
    assert [d.id for d, _ in ranked] == sorted(scores)


# ---------------------------------------------------------------- eval benchmark
def test_eval_lexical_golden():
    m = evaluate("lexical", INDEX)
    assert m.recall_at_5 == pytest.approx(0.75, abs=1e-9)  # 12/16
    assert m.mrr == pytest.approx(0.625, abs=1e-9)
    misses = {p["id"] for p in m.per_query if p["hit@5"] == 0}
    assert misses == {"q02", "q06", "q07", "q14"}  # the four paraphrase misses


def test_eval_hybrid_without_dense_falls_back_to_lexical():
    m_lex = evaluate("lexical", INDEX)
    m_hyb = evaluate("hybrid", INDEX, dense=None)  # no dense backend
    assert m_hyb.recall_at_5 == m_lex.recall_at_5
    assert m_hyb.mrr == m_lex.mrr


def test_hybrid_fallback_ranking_equals_lexical():
    # Single-list RRF is an identity fusion: the hybrid must rank exactly
    # like the lexical tier, with no phantom docs floating to the top.
    lex = [d.id for d, _ in INDEX.search("flash loan mechanics", top_k=5)]
    hyb = [d.id for d, _ in rank_query("hybrid", INDEX, None, "flash loan mechanics", top_k=5)]
    assert hyb == lex


def test_eval_metrics_as_dict_shape():
    m = evaluate("lexical", INDEX)
    d = m.as_dict()
    assert d["tier"] == "lexical" and d["queries"] == 16
    assert "recall@5" in d and "mrr" in d


# ---------------------------------------------------------------- dense tier (graceful)
def test_dense_requires_backend_for_ranking():
    with pytest.raises(RuntimeError):
        rank_query("dense", INDEX, None, "flash loan mechanics")


# ---------------------------------------------------------------- CLI
def test_cli_search(capsys):
    from embed_playground.cli import main

    assert main(["search", "flash loan", "mechanics"]) == 0
    out = capsys.readouterr().out
    assert "d02" in out and "tier: lexical" in out


def test_cli_corpus(capsys):
    from embed_playground.cli import main

    assert main(["corpus"]) == 0
    out = capsys.readouterr().out
    assert "docs: 36" in out and "benchmark queries: 16" in out


def test_cli_eval(capsys):
    from embed_playground.cli import main

    assert main(["eval", "--tier", "lexical"]) == 0
    out = capsys.readouterr().out
    assert "recall@5: 75.0%" in out and "mrr:" in out


def test_cli_eval_json(capsys):
    from embed_playground.cli import main

    assert main(["eval", "--tier", "lexical", "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recall@5"] == pytest.approx(0.75, abs=1e-6)


def test_cli_search_dense_unavailable(monkeypatch, capsys):
    from embed_playground import cli

    def no_dense():
        raise RuntimeError("dense tier not available")

    monkeypatch.setattr(cli, "_dense", no_dense)
    assert cli.main(["search", "flash loan", "--tier", "dense"]) == 2
    assert "dense" in capsys.readouterr().err


def test_cli_compare_falls_back_without_dense(monkeypatch, capsys):
    from embed_playground import cli

    class Broken:
        def __init__(self):
            raise RuntimeError("no sentence-transformers")

    monkeypatch.setattr(cli, "DenseBackend", Broken)
    assert cli.main(["compare", "flash loan"]) == 0
    err = capsys.readouterr().err
    assert "dense" in err and "lexical vs hybrid" in err


def test_cli_empty_query_message(capsys):
    from embed_playground.cli import main

    assert main(["search", "!!!", "..."]) == 0
    assert "no results" in capsys.readouterr().out