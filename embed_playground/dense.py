"""Optional sentence-transformers dense retrieval backend.

Single concern: loading a transformer model, encoding texts, and
performing cosine-similarity search over pre-built document vectors.
"""

from __future__ import annotations

import math

from .core import Doc

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def cosine_sim(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError("vector length mismatch")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


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
            raise RuntimeError(
                f"failed to load {self.model_name!r} (offline? wrong id?): {exc}"
            ) from exc

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
