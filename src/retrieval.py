"""Dense, TF-IDF and rank-fused retrieval over the Day 6 B/C chunks."""

import gzip
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from starter.baseline import TfidfIndex
from .day6_pipeline.embeddings import MyEmbedder
from .day6_pipeline.store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
EMBEDDING_MODEL = "openai/text-embedding-3-small"
DIMENSIONS = 1536


def load_environment() -> None:
    load_dotenv(ROOT / ".env", override=False)
    # Reuse an existing local secret without copying it into the Day 7 repo.
    day6 = Path(os.getenv("DAY6_ROOT", str(ROOT.parent / "K5-L3-Day06-DataIngestionEngineering")))
    load_dotenv(day6 / "starter/.env", override=False)


def load_chunks() -> list[dict[str, Any]]:
    chunks = json.loads((ROOT / "data/chunks.json").read_text(encoding="utf-8"))
    if len(chunks) != 340 or len({c["id"] for c in chunks}) != len(chunks):
        raise ValueError("Expected 340 unique Day 6 B/C chunks")
    return chunks


def _unit(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0 or len(vector) != DIMENSIONS:
        raise ValueError("Invalid embedding")
    return [x / norm for x in vector]


class Retriever:
    def __init__(self, chunks: list[dict[str, Any]], backend: str = "local"):
        if backend not in {"local", "pgvector"}:
            raise ValueError("backend must be local or pgvector")
        load_environment()
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self.lexical = TfidfIndex(chunks)
        self.backend = backend
        self.embedder = None
        self.store = None
        self.vectors = None
        self.query_cache: dict[str, list[float]] = {}
        if backend == "local":
            path = ROOT / "data/vectors.json.gz"
            if path.exists():
                with gzip.open(path, "rt", encoding="utf-8") as stream:
                    saved = json.load(stream)
                if saved["model"] != EMBEDDING_MODEL or saved["ids"] != [c["id"] for c in chunks]:
                    raise ValueError("Local vectors do not match the current Day 6 chunks/model")
                self.vectors = [_unit(v) for v in saved["vectors"]]
        else:
            url = os.getenv("DATABASE_URL")
            if not url:
                raise RuntimeError("DATABASE_URL is required for pgvector backend")
            self.store = PgVectorStore(url, DIMENSIONS, EMBEDDING_MODEL)
            if self.store.count("C") != len(chunks):
                raise RuntimeError("Day 6 strategy C has no matching 340-chunk index")

    def index_local(self) -> int:
        if self.backend != "local":
            raise ValueError("index_local is only for local backend")
        self.embedder = MyEmbedder(EMBEDDING_MODEL, DIMENSIONS)
        vectors = self.embedder.embed_documents([c["text"] for c in self.chunks])
        path = ROOT / "data/vectors.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as stream:
            json.dump({"model": EMBEDDING_MODEL, "ids": [c["id"] for c in self.chunks],
                       "vectors": vectors}, stream)
        self.vectors = [_unit(v) for v in vectors]
        return self.embedder.prompt_tokens

    def _query_vector(self, query: str) -> list[float]:
        if query not in self.query_cache:
            if self.embedder is None:
                self.embedder = MyEmbedder(EMBEDDING_MODEL, DIMENSIONS)
            self.query_cache[query] = self.embedder.embed_query(query)
        return self.query_cache[query]

    def dense(self, query: str, top_k: int, filters: dict | None = None) -> list[dict]:
        q = self._query_vector(query)
        if self.backend == "pgvector":
            rows = self.store.search("C", q, top_k=top_k, filters=filters)
            return [{"id": r["chunk_id"], "source_id": r["document_id"],
                     "source": r["source"], "page": r["page"], "text": r["text"],
                     "metadata": r["metadata"], "score": r["score"], "strategy": "dense"}
                    for r in rows]
        if self.vectors is None:
            raise RuntimeError("Run `python -m src.run index` to create real embeddings first")
        q_unit = _unit(q)
        scored = []
        for chunk, vector in zip(self.chunks, self.vectors, strict=True):
            if filters and chunk["metadata"].get("section") != filters.get("section"):
                continue
            score = sum(a * b for a, b in zip(q_unit, vector, strict=True))
            scored.append({**chunk, "score": score, "strategy": "dense"})
        return sorted(scored, key=lambda c: (-c["score"], c["id"]))[:top_k]

    def tfidf(self, query: str, top_k: int, filters: dict | None = None) -> list[dict]:
        # The starter TF-IDF has no metadata filter; rank a section subset when needed.
        if filters:
            subset = [c for c in self.chunks if c["metadata"].get("section") == filters.get("section")]
            return TfidfIndex(subset).search(query, top_k=top_k) if subset else []
        return self.lexical.search(query, top_k=top_k)

    def search(self, method: str, query: str, top_k: int = 3,
               filters: dict | None = None, candidate_k: int = 12) -> list[dict]:
        if filters and (set(filters) != {"section"} or not isinstance(filters["section"], str)):
            raise ValueError("Only a user/question-derived section filter is allowed")
        if method == "dense":
            rows = self.dense(query, top_k, filters)
        elif method == "tfidf":
            rows = self.tfidf(query, top_k, filters)
        elif method == "hybrid":
            dense = self.dense(query, candidate_k, filters)
            lexical = self.tfidf(query, candidate_k, filters)
            scores: Counter[str] = Counter()
            for ranking in (dense, lexical):
                for rank, row in enumerate(ranking, 1):
                    scores[row["id"]] += 1 / (60 + rank)
            rows = [{**self.by_id[cid], "score": score, "strategy": "hybrid"}
                    for cid, score in scores.items()]
            rows = sorted(rows, key=lambda c: (-c["score"], c["id"]))[:top_k]
        else:
            raise ValueError("Unknown retrieval method")
        return [{**row, "rank": rank} for rank, row in enumerate(rows, 1)]
