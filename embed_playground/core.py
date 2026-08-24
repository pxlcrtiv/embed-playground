"""Deterministic TF-IDF retrieval core: tokenizer, vectors, cosine, RRF."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_STOP = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "has", "have", "i", "if", "in", "is", "it", "its", "my", "of", "on",
    "or", "that", "the", "to", "was", "we", "what", "when", "which", "who",
    "will", "with", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, stopwords removed — deterministic."""
    return [m.group(0) for m in _WORD_RE.finditer(text.lower()) if m.group(0) not in _STOP]


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    category: str
    text: str


class TfidfIndex:
    """L2-normalized TF-IDF index over a list of docs (pure stdlib).

    idf(w) = ln((1 + N) / (1 + df(w))) + 1 — the standard scikit-learn
    smooth form, implemented here with zero dependencies so the numbers are
    reproducible anywhere Python runs.
    """

    def __init__(self, docs: list[Doc]):
        self.docs = list(docs)
        self.n = len(self.docs)
        self._df: dict[str, int] = {}
        self._tf: list[dict[str, float]] = []
        for d in self.docs:
            terms = tokenize(d.text + " " + d.title)
            tf: dict[str, float] = {}
            for t in terms:
                tf[t] = tf.get(t, 0) + 1.0
            self._tf.append(tf)
            for t in set(tf):
                self._df[t] = self._df.get(t, 0) + 1
        self._idf = {t: math.log((1 + self.n) / (1 + df)) + 1 for t, df in self._df.items()}
        self._norm: list[float] = []
        for tf in self._tf:
            s = math.sqrt(sum(
                (tf[t] * self._idf[t]) ** 2
                for t in tf if t in self._idf
            ))
            self._norm.append(s if s > 0 else 1.0)

    def query_vector(self, query: str) -> dict[str, float]:
        q = tokenize(query)
        counts: dict[str, float] = {}
        for t in q:
            counts[t] = counts.get(t, 0) + 1.0
        max_c = max(counts.values()) if counts else 0.0
        vec: dict[str, float] = {}
        for t, c in counts.items():
            if t in self._idf:
                vec[t] = (c / max_c) * self._idf[t]
        return vec

    def cosine(self, qvec: dict[str, float], doc_i: int) -> float:
        tf = self._tf[doc_i]
        dot = 0.0
        for t, qw in qvec.items():
            if t in tf:
                dot += qw * tf[t] * self._idf[t]
        qn = math.sqrt(sum(v * v for v in qvec.values()))
        if qn == 0:
            return 0.0
        return dot / (qn * self._norm[doc_i])

    def search(self, query: str, top_k: int = 5) -> list[tuple[Doc, float]]:
        """Rank docs by cosine similarity, desc; ties by doc id (stable)."""
        qvec = self.query_vector(query)
        if not qvec:
            return []
        scored = [(d, self.cosine(qvec, i)) for i, d in enumerate(self.docs)]
        scored.sort(key=lambda pair: (-pair[1], pair[0].id))
        return scored[:top_k]

    def stats(self) -> dict:
        return {"docs": self.n, "terms": len(self._idf)}


def rrf_score(ranks: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over ranked doc-id lists (Cormack et al.).

    score(d) = sum over lists of 1 / (k + rank(d)). Deterministic; a single
    list fused with itself is its own ranking (fallback behavior when only
    one tier is available).
    """
    fused: dict[str, float] = {}
    for lst in ranks:
        for pos, doc_id in enumerate(lst):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + pos + 1)
    return fused


def rank_by_score(scores: dict[str, float], doc_by_id: dict[str, Doc]) -> list[tuple[Doc, float]]:
    """Deterministic ordering: score desc, then doc id asc."""
    items = [(doc_by_id[doc_id], s) for doc_id, s in scores.items() if doc_id in doc_by_id]
    items.sort(key=lambda pair: (-pair[1], pair[0].id))
    return items