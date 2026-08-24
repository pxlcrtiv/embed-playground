"""Rendering: deterministic search tables and evaluation reports."""

from __future__ import annotations

import json

from .backends import EvalMetrics
from .core import Doc


def render_results(query: str, tier: str, results: list[tuple[Doc, float]], top_k: int) -> str:
    """Stable, copy-pasteable result table."""
    lines = [
        f"query: {query!r}",
        f"tier: {tier}  top-k: {min(top_k, len(results))}",
        "",
    ]
    if not results:
        lines.append("(no results — query has no in-vocabulary terms for the lexical tier)")
        return "\n".join(lines)
    lines.append(f"{'rank':<5}{'doc':<6}{'score':<10}{'category':<10}title")
    for rank, (doc, score) in enumerate(results, start=1):
        lines.append(f"{rank:<5}{doc.id:<6}{score:<10.4f}{doc.category:<10}{doc.title}")
    return "\n".join(lines)


def render_snippet(doc: Doc, width: int = 110) -> str:
    text = doc.text.replace("\n", " ")
    return text if len(text) <= width else text[: width - 1] + "…"


def render_comparison(query: str, rows: list[tuple[str, list[tuple[Doc, float]]]]) -> str:
    """Lexical vs dense vs hybrid side-by-side (per tier, ranked)."""
    out = [f"query: {query!r}", ""]
    for tier, results in rows:
        out.append(f"  [{tier}]")
        for rank, (doc, score) in enumerate(results[:5], start=1):
            out.append(f"    {rank}. {doc.id:<6} {score:<8.4f} {doc.category:<10} {doc.title}")
        out.append("")
    return "\n".join(out).rstrip()


def render_eval(metrics: EvalMetrics, detail: bool = False) -> str:
    lines = [
        f"tier: {metrics.tier}",
        f"recall@5: {metrics.recall_at_5:.1%} ({round(metrics.recall_at_5 * metrics.queries)}/{metrics.queries})",
        f"mrr:      {metrics.mrr:.3f}",
    ]
    if detail:
        lines.append("")
        lines.append("per-query (hit@5, reciprocal rank):")
        for q in metrics.per_query:
            lines.append(f"  {q['id']}  {q['intent']:<14} hit={q['hit@5']}  rr={q['rr']}")
    return "\n".join(lines)


def render_eval_csv(metrics: EvalMetrics) -> str:
    rows = ["tier,queries,recall@5,mrr"]
    m = metrics.as_dict()
    rows.append(f"{m['tier']},{m['queries']},{m['recall@5']},{m['mrr']}")
    return "\n".join(rows) + "\n"


def render_eval_json(metrics: EvalMetrics, detail: bool = False) -> str:
    payload = metrics.as_dict()
    if detail:
        payload["per_query"] = metrics.per_query
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"