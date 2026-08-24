"""embed-playground CLI — search, compare, and evaluate, keyless by default."""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache

from . import __version__
from .backends import DenseBackend, evaluate, load_corpus, rank_query
from .core import TfidfIndex
from .report import (
    render_comparison,
    render_eval,
    render_eval_csv,
    render_eval_json,
    render_results,
    render_snippet,
)

TIERS = ("lexical", "dense", "hybrid")
EXIT_OK, EXIT_NO_BACKEND, EXIT_ERROR = 0, 2, 1


@lru_cache(maxsize=1)
def _index() -> TfidfIndex:
    return TfidfIndex(load_corpus())


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="embed-playground",
        description="Compare lexical (TF-IDF) and dense (embeddings) search on a bundled DeFi/Web3 corpus.",
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("search", help="search the bundled corpus")
    s.add_argument("query", nargs="+", help="search terms (a phrase)")
    s.add_argument("--tier", choices=TIERS, default="lexical")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--snippet", action="store_true", help="show a text snippet per hit")

    c = sub.add_parser("compare", help="lexical vs dense vs hybrid side-by-side")
    c.add_argument("query", nargs="+")
    c.add_argument("--top-k", type=int, default=5)

    e = sub.add_parser("eval", help="run the bundled retrieval benchmark")
    e.add_argument("--tier", choices=TIERS, default="lexical")
    e.add_argument("--detail", action="store_true")
    e.add_argument("--format", choices=("text", "csv", "json"), default="text")

    a = sub.add_parser("app", help="launch the Streamlit playground (needs streamlit)")
    a.add_argument("--port", type=int, default=8501)

    co = sub.add_parser("corpus", help="bundled corpus stats")
    co.add_argument("--json", action="store_true")
    return p


def _dense() -> DenseBackend | None:
    """Instantiate the dense backend only when requested."""
    try:
        return DenseBackend()
    except Exception:  # noqa: BLE001 - lazy; CLI decides how to report
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    index = _index()
    docs = index.docs

    if args.command == "corpus":
        from .backends import load_queries

        stats = index.stats()
        stats["benchmark_queries"] = len(load_queries())
        cats = sorted({d.category for d in docs})
        stats["categories"] = cats
        if args.json:
            import json

            print(json.dumps(stats, sort_keys=True))
        else:
            print(
                f"docs: {stats['docs']} | terms: {stats['terms']} | "
                f"benchmark queries: {stats['benchmark_queries']} | "
                f"categories: {', '.join(cats)}"
            )
        return EXIT_OK

    if args.command == "search":
        query = " ".join(args.query)
        try:
            dense = _dense() if args.tier in ("dense", "hybrid") else None
            results = rank_query(args.tier, index, dense, query, top_k=args.top_k)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NO_BACKEND
        print(render_results(query, args.tier, results, args.top_k))
        if args.snippet:
            print()
            for doc, _score in results[:3]:
                print(f"  [{doc.id}] {render_snippet(doc)}")
                print()
        return EXIT_OK

    if args.command == "compare":
        query = " ".join(args.query)
        rows: list[tuple[str, list]] = []
        try:
            dense = DenseBackend()
            rows.append(("lexical", rank_query("lexical", index, None, query, top_k=5)))
            rows.append(("dense", rank_query("dense", index, dense, query, top_k=5)))
            rows.append(("hybrid", rank_query("hybrid", index, dense, query, top_k=5)))
        except RuntimeError as exc:
            # dense unavailable: fall back to lexical vs hybrid(lexical-only)
            rows.append(("lexical", rank_query("lexical", index, None, query, top_k=5)))
            rows.append(("hybrid", rank_query("hybrid", index, None, query, top_k=5)))
            print(f"note: dense tier unavailable ({exc}) — comparing lexical vs hybrid", file=sys.stderr)
            return EXIT_OK
        print(render_comparison(query, rows))
        return EXIT_OK

    if args.command == "eval":
        try:
            dense = _dense() if args.tier in ("dense", "hybrid") else None
            metrics = evaluate(args.tier, index, dense)
        except RuntimeError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return EXIT_NO_BACKEND
        if args.format == "csv":
            print(render_eval_csv(metrics), end="")
        elif args.format == "json":
            print(render_eval_json(metrics, detail=args.detail), end="")
        else:
            print(render_eval(metrics, detail=args.detail))
        return EXIT_OK

    if args.command == "app":
        import importlib.util

        if importlib.util.find_spec("streamlit") is None:
            print("error: streamlit not installed — pip install 'embed-playground[app]'", file=sys.stderr)
            return EXIT_NO_BACKEND
        import subprocess

        return subprocess.run(
            [sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(args.port)],
            check=False,
        ).returncode

    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())