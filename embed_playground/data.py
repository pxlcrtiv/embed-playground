"""Corpus and benchmark query loading.

Single concern: reading bundled JSON data files and producing typed
Python objects (Doc and Query).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .core import Doc

DATA_DIR = Path(__file__).resolve().parent / "data"


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
        return cls(
            id=str(d["id"]),
            query=str(d["query"]),
            relevant=rel,
            intent=str(d.get("intent", "definitional")),
        )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_corpus() -> list[Doc]:
    raw = _read_json(DATA_DIR / "corpus.json")
    docs = [
        Doc(
            id=str(d["id"]),
            title=str(d["title"]),
            category=str(d["category"]),
            text=str(d["text"]),
        )
        for d in raw["docs"]
    ]
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
