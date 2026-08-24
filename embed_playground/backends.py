"""Dense embedding backend (optional) and the retrieval evaluation engine.

The dense tier uses sentence-transformers (all-MiniLM-L6-v2 by default) —
keyless, CPU, cached after first download. It is imported lazily so the
default install needs nothing beyond the stdlib.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

from .core import Doc, TfidfIndex, rank_by_score, rrf_score

DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class DenseBackend:
    """Optional sentence-transformers retriever (CPU, keyless)."""

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or DEFAULT_MODEL
        self._model = None
        self._doc_vecs: list[list[float]] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - environment-dependent
            raise RuntimeError(
                "The 'dense' tier needs the optional dependency: "
                "pip install 'embed-playground[dense]' (installs sentence-transformers)"
            ) from exc
        try:
            self._model = SentenceTransformer(self.model_name)
            self._model.eval()
        except Exception as exc:  # pragma: no cover - network/model dependent
            raise RuntimeError(f"failed to load {self.model_name!r} (offline? wrong id?): {exc}") from exc

    def encode(self, texts: list[str]) -> list[list[float]]:
        self._load()
        import torch

        with torch.no_grad():
            return [v.tolist() for v in self._model.encode(texts, convert_to_numpy=True)]

    def index_docs(self, docs: list[Doc]) -> None:
        self._doc_vecs = self.encode([f"{d.title}. {d.text}" for d in docs])

    def search(self, query: str, docs: list[Doc], top_k: int = 5) -> list[tuple[Doc, float]]:
        if self._doc_vecs is None:
            self.index_docs(docs)
        qv = self.encode([query])[0]
        sims = [cosine_sim(qv, dv) for dv in self._doc_vecs or []]
        ranked = sorted(zip(docs, sims), key=lambda pair: (-pair[1], pair[0].id))
        return ranked[:top_k]


def cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


@dataclass(frozen=True)
class Query:
    id: str
    query: str
    relevant: tuple[str, ...]
    intent: str

    @classmethod
    def from_dict(cls, d: dict) -> Query:
        rel = tuple(str(r) for r in d["relevant"])
        if not rel:
            raise ValueError(f"{d['id']}: at least one relevant doc required")
        return cls(id=str(d["id"]), query=str(d["query"]), relevant=rel, intent=str(d.get("intent", "definitional")))


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


def load_corpus() -> list[Doc]:
    raw = _read_json(DATA_DIR / "corpus.json")
    docs = [Doc(id=str(d["id"]), title=str(d["title"]), category=str(d["category"]), text=str(d["text"])) for d in raw["docs"]]
    ids = [d.id for d in docs]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate doc ids in corpus")
    return docs


def load_queries() -> list[Query]:
    raw = _read_json(DATA_DIR / "bench" / "queries.json")
    known = {d.id for d in load_corpus()}
    queries = [Query.from_dict(d) for d in raw["queries"]]
    for q in queries:
        for rel in q.relevant:
            if rel not in known:
                raise ValueError(f"{q.id}: relevant doc {rel!r} not in corpus")
    return queries


def _read_json(path: Path) -> dict:
    import json

    return json.loads(path.read_text(encoding="utf-8"))


def rank_query(tier: str, index: TfidfIndex, dense: DenseBackend | None, query: str, top_k: int = 50) -> list[tuple[Doc, float]]:
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
        # RRF scores exist only for docs that at least one tier retrieved;
        # everything else is explicitly not ranked for this query.
        return rank_by_score(fused, {d.id: d for d in index.docs})[:top_k]
    raise ValueError(f"unknown tier: {tier}")


def evaluate(tier: str, index: TfidfIndex, dense: DenseBackend | None = None) -> EvalMetrics:
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